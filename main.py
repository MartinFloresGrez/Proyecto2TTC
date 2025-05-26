import cv2
import face_recognition
import numpy as np
import requests

# Función para cargar registros desde MongoDB API
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

# Función para borrar un rostro por nombre
def borrar_rostro_api(nombre):
    try:
        response = requests.delete(f"http://localhost:8000/rostros/{nombre}")
        if response.status_code == 200:
            print(f"✅ {response.json()['mensaje']}")
            return True
        elif response.status_code == 404:
            print("❌ Rostro no encontrado en la base de datos.")
        else:
            print(f"❌ Error al borrar: {response.text}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
    return False

# Cargar registros desde la API al iniciar
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        top = y
        right = x + w
        bottom = y + h
        left = x

        small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        small_top = top // 2
        small_right = right // 2
        small_bottom = bottom // 2
        small_left = left // 2

        encoding = face_recognition.face_encodings(small_rgb, [(small_top, small_right, small_bottom, small_left)])
        nombre_detectado = "Desconocido"

        if encoding:
            resultados = face_recognition.compare_faces(encodings_conocidos, encoding[0])
            if True in resultados:
                index = resultados.index(True)
                nombre_detectado = nombres_conocidos[index]

        cv2.putText(frame, nombre_detectado, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow('Reconocimiento facial', frame)

    key = cv2.waitKey(1)

    # Registrar nuevo rostro
    if key == ord('r') and len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            print(f"[INFO] Registrando rostro {i+1}...")

            top = y
            right = x + w
            bottom = y + h
            left = x

            small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            small_top = top // 2
            small_right = right // 2
            small_bottom = bottom // 2
            small_left = left // 2

            face_encoding = face_recognition.face_encodings(small_rgb, [(small_top, small_right, small_bottom, small_left)])

            if face_encoding:
                cv2.destroyAllWindows()
                nombre = input(f"Ingresa el nombre para rostro {i+1}: ")

                payload = {
                    "nombre": nombre,
                    "encoding": face_encoding[0].tolist()
                }

                try:
                    response = requests.post("http://localhost:8000/rostros", json=payload)
                    if response.status_code == 200:
                        print("✅ Rostro registrado en MongoDB.")
                        encodings_conocidos, nombres_conocidos = cargar_registros()
                    else:
                        print(f"❌ Error al registrar: {response.text}")
                except Exception as e:
                    print(f"❌ Error de conexión: {e}")

                cap = cv2.VideoCapture(1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('Reconocimiento facial', frame)
            else:
                print("❌ No se pudo generar encoding facial.")

    # Borrar rostro
    elif key == ord('d'):
        cv2.destroyAllWindows()
        print("\n=== Rostros registrados ===")
        for i, nombre in enumerate(nombres_conocidos):
            print(f"{i}: {nombre}")

        try:
            index = int(input("\nIngresa el número del rostro a borrar (o -1 para cancelar): "))
            if 0 <= index < len(nombres_conocidos):
                nombre_a_borrar = nombres_conocidos[index]
                if borrar_rostro_api(nombre_a_borrar):
                    encodings_conocidos, nombres_conocidos = cargar_registros()
            else:
                print("❎ Cancelado.")
        except ValueError:
            print("❎ Entrada inválida.")

        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
