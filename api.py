from pymongo import MongoClient
from fastapi import FastAPI, HTTPException


# Conexión a MongoDB (ajusta la URI si estás usando Atlas o una IP diferente)
client = MongoClient("mongodb+srv://rostrosDb:123rostros123@rostrosdb.7gbe7s8.mongodb.net/?retryWrites=true&w=majority&appName=RostrosDb")
db = client["RostrosDb"]
coleccion = db["rostros"]



# Inicializar FastAPI
app = FastAPI()


@app.get("/rostros")
def obtener_rostros():
    """
    Obtener todos los rostros registrados en la base de datos.
    """
    try:
        rostros = list(coleccion.find({}, {"_id": 0}))  # Excluir el campo _id
        return {"rostros": rostros}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

@app.post("/rostros")
def registrar_rostro(rostro: dict):
    """
    Registrar un nuevo rostro en la base de datos.
    """
    try:
        coleccion.insert_one(rostro)
        return {"mensaje": "Rostro registrado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/rostros/{nombre}")
def borrar_rostro(nombre: str):
    """
    Borrar un rostro registrado en la base de datos por nombre.
    """
    try:
        resultado = coleccion.delete_one({"nombre": nombre})
        if resultado.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Rostro no encontrado.")
        return {"mensaje": "Rostro borrado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


