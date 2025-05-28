from fastapi import FastAPI, HTTPException, WebSocket, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import base64
import cv2
import face_recognition
import concurrent.futures
from typing import List, Tuple

# ==== FastAPI App ====
app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ==== Conexión MongoDB ====
client = MongoClient("mongodb://rostrosDb:123rostros123@ac-htsmw9f-shard-00-00.7gbe7s8.mongodb.net:27017,ac-htsmw9f-shard-00-01.7gbe7s8.mongodb.net:27017,ac-htsmw9f-shard-00-02.7gbe7s8.mongodb.net:27017/?ssl=true&replicaSet=atlas-hzhi1a-shard-0&authSource=admin&retryWrites=true&w=majority&appName=RostrosDb")
db = client["RostrosDb"]
coleccion = db["rostros"]
sesiones = db["sesiones"]

# ==== Cargar rostros desde MongoDB ====
def cargar_registros():
    rostros = list(coleccion.find({}, {"_id": 0}))
    encodings = [np.array(r["encoding"]) for r in rostros]
    nombres = [r["nombre"] for r in rostros]
    return encodings, nombres

# ==== Endpoints REST ====

@app.get("/rostros")
def obtener_rostros():
    try:
        rostros = list(coleccion.find({}, {"_id": 0}))
        return {"rostros": rostros}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rostros")
def registrar_rostro(rostro: dict):
    try:
        coleccion.insert_one(rostro)
        return {"mensaje": "Rostro registrado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rostros/{nombre}")
def borrar_rostro(nombre: str):
    try:
        resultado = coleccion.delete_one({"nombre": nombre})
        if resultado.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Rostro no encontrado.")
        return {"mensaje": "Rostro borrado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return FileResponse("frontend/lobby/lobby.html")

@app.get("/static/lobby/lobby.html")
def redirect_to_root():
    return RedirectResponse("/")

@app.get("/sesiones")
def obtener_sesiones():
    try:
        lista = []
        for sesion in sesiones.find({}):
            sesion["_id"] = str(sesion["_id"])
            lista.append(sesion)
        return {"sesiones": lista}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sesiones")
def crear_sesion(sesion: dict):
    try:
        # Asegura que la sesión tenga los campos requeridos
        nueva_sesion = {
            "nombre": sesion.get("nombre"),
            "profesor": sesion.get("profesor"),
            "asistentes": []
        }
        sesiones.insert_one(nueva_sesion)
        return {"mensaje": "Sesión creada exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sesiones/{sesion_id}")
def obtener_sesion(sesion_id: str):
    sesion = sesiones.find_one({"_id": ObjectId(sesion_id)})
    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    sesion["_id"] = str(sesion["_id"])
    return sesion

@app.post("/sesiones/{sesion_id}/asistencia")
def registrar_asistencia(sesion_id: str, data: dict):
    nombre = data.get("nombre")
    fecha = data.get("fecha")
    if not nombre:
        raise HTTPException(status_code=400, detail="Nombre requerido")
    sesion = sesiones.find_one({"_id": ObjectId(sesion_id)})
    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    # Evitar duplicados
    if any(a["nombre"] == nombre for a in sesion.get("asistentes", [])):
        return {"mensaje": "Ya registrado"}
    sesiones.update_one(
        {"_id": ObjectId(sesion_id)},
        {"$push": {"asistentes": {"nombre": nombre, "fecha": fecha}}}
    )
    return {"mensaje": "Asistencia registrada"}

@app.delete("/sesiones/{sesion_id}/asistencia")
def eliminar_asistencia(sesion_id: str, data: dict):
    nombre = data.get("nombre")
    if not nombre:
        raise HTTPException(status_code=400, detail="Nombre requerido")
    sesion = sesiones.find_one({"_id": ObjectId(sesion_id)})
    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    sesiones.update_one(
        {"_id": ObjectId(sesion_id)},
        {"$pull": {"asistentes": {"nombre": nombre}}}
    )
    return {"mensaje": "Asistencia eliminada"}

# ==== WebSocket para detección en tiempo real desde navegador ====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("🔌 WebSocket conectado")
    await websocket.accept()
    encodings_conocidos, nombres_conocidos = cargar_registros()
    
    # Crear un pool de hilos para procesar reconocimientos faciales en paralelo
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)  # Ajusta según tu CPU
    
    # Función que procesa un rostro individual en un hilo separado
    def procesar_rostro(face_encoding, face_location, encodings_conocidos, nombres_conocidos, accion, nombre_nuevo) -> dict:
        top, right, bottom, left = face_location
        nombre = "Desconocido"
        
        # Comparar con rostros conocidos
        match = face_recognition.compare_faces(encodings_conocidos, face_encoding, 0.5)
        if True in match:
            index = match.index(True)
            nombre = nombres_conocidos[index]
        
        resultado = {
            "nombre": nombre,
            "coordenadas": [top, right, bottom, left]
        }
        
        # Manejar acciones de registro/borrado por separado en cada hilo
        acciones_pendientes = []
        if accion == "registrar" and nombre == "Desconocido" and nombre_nuevo:
            acciones_pendientes.append(("registrar", nombre_nuevo, face_encoding))
        elif accion == "borrar" and nombre != "Desconocido":
            acciones_pendientes.append(("borrar", nombre, None))
            
        return resultado, acciones_pendientes
    
    while True:
        try:
            data = await websocket.receive_json()
            img_data = data["image"].split(",")[1]
            accion = data.get("accion", None)
            nombre_nuevo = data.get("nombre", "").strip()
            
            # Procesar imagen y detectar rostros (esto debe permanecer secuencial)
            img_bytes = base64.b64decode(img_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            
            detecciones = []
            todas_acciones_pendientes = []
            
            # No hay rostros, enviar respuesta vacía inmediatamente
            if len(face_locations) == 0:
                await websocket.send_json(detecciones)
                continue
                
            # Procesar cada rostro en hilos paralelos
            futures = []
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                future = executor.submit(
                    procesar_rostro,
                    face_encoding,
                    face_location,
                    encodings_conocidos,
                    nombres_conocidos,
                    accion,
                    nombre_nuevo if i == 0 and accion == "registrar" else ""  # Solo registrar el primer rostro si hay múltiples
                )
                futures.append(future)
            
            # Recolectar resultados cuando todos los hilos terminen
            for future in concurrent.futures.as_completed(futures):
                resultado, acciones = future.result()
                detecciones.append(resultado)
                todas_acciones_pendientes.extend(acciones)
            
            # Procesar acciones pendientes (registro/borrado)
            debe_recargar = False
            for accion_tipo, nombre, encoding in todas_acciones_pendientes:
                if accion_tipo == "registrar":
                    print(f"➡️ Registrando rostro como {nombre}")
                    coleccion.insert_one({
                        "nombre": nombre,
                        "encoding": encoding.tolist()
                    })
                    debe_recargar = True
                elif accion_tipo == "borrar":
                    print(f"➡️ Borrando rostro {nombre}")
                    coleccion.delete_one({"nombre": nombre})
                    debe_recargar = True
            
            # Recargar las codificaciones si se realizaron cambios en la base de datos
            if debe_recargar:
                encodings_conocidos, nombres_conocidos = cargar_registros()
            
            # Enviar resultados al cliente
            await websocket.send_json(detecciones)

        except Exception as e:
            print("❌ WebSocket error:", e)
            break
