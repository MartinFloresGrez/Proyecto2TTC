from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import base64
import cv2
import face_recognition

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

# ==== WebSocket para detección en tiempo real desde navegador ====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("🔌 WebSocket conectado")
    await websocket.accept()
    encodings_conocidos, nombres_conocidos = cargar_registros()
    while True:
        try:
            data = await websocket.receive_json()
            img_data = data["image"].split(",")[1]
            accion = data.get("accion", None)
            nombre_nuevo = data.get("nombre", "").strip()
            img_bytes = base64.b64decode(img_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            detecciones = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                nombre = "Desconocido"
                match = face_recognition.compare_faces(encodings_conocidos, face_encoding)
                if True in match:
                    index = match.index(True)
                    nombre = nombres_conocidos[index]

                detecciones.append({
                    "nombre": nombre,
                    "coordenadas": [top, right, bottom, left]
                })

                if accion == "registrar" and nombre == "Desconocido" and nombre_nuevo:
                    print(f"➡️ Registrando rostro como {nombre_nuevo}")
                    coleccion.insert_one({
                        "nombre": nombre_nuevo,
                        "encoding": face_encoding.tolist()
                    })
                    encodings_conocidos, nombres_conocidos = cargar_registros()

                elif accion == "borrar" and nombre != "Desconocido":
                    print(f"➡️ Borrando rostro {nombre}")
                    coleccion.delete_one({"nombre": nombre})
                    encodings_conocidos, nombres_conocidos = cargar_registros()

            await websocket.send_json(detecciones)

        except Exception as e:
            print("❌ WebSocket error:", e)
            break
