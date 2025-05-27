# main.py
import cv2
import face_recognition
import numpy as np
import requests

def cargar_registros():
    try:
        response = requests.get("http://localhost:8000/rostros")
        data = response.json()["rostros"]
        encodings = [np.array(item["encoding"]) for item in data]
        nombres = [item["nombre"] for item in data]
        return encodings, nombres
    except Exception as e:
        print(f"Error al cargar desde la API: {e}")
        return [], []

def borrar_rostro_api(nombre):
    try:
        response = requests.delete(f"http://localhost:8000/rostros/{nombre}")
        if response.status_code == 200:
            print(f"✅ {response.json()['mensaje']}")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

def detectar_y_registrar():
    encodings_conocidos, nombres_conocidos = cargar_registros()

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            small_coords = (top // 2, right // 2, bottom // 2, left // 2)

            encoding = face_recognition.face_encodings(small_rgb, [small_coords])
            nombre_detectado = "Desconocido"

            if encoding:
                resultados = face_recognition.compare_faces(encodings_conocidos, encoding[0])
                if True in resultados:
                    nombre_detectado = nombres_conocidos[resultados.index(True)]

            cv2.putText(frame, nombre_detectado, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Reconocimiento facial', frame)

        key = cv2.waitKey(1)

        if key == ord('r') and len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                top, right, bottom, left = y, x + w, y + h, x
                small_coords = (top // 2, right // 2, bottom // 2, left // 2)
                face_encoding = face_recognition.face_encodings(small_rgb, [small_coords])

                if face_encoding:
                    cv2.destroyAllWindows()
                    nombre = input(f"Ingrese el nombre para el rostro {i+1}: ")
                    payload = {
                        "nombre": nombre,
                        "encoding": face_encoding[0].tolist()
                    }
                    requests.post("http://localhost:8000/rostros", json=payload)
                    print("✅ Rostro enviado al backend.")
                    return  # Finaliza después de registrar

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
