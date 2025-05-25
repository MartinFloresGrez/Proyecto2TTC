import cv2
import face_recognition
import pickle
import os

# Crear carpeta si no existe
if not os.path.exists("encodings"):
    os.makedirs("encodings")

# Función para cargar registros
def cargar_registros():
    encodings = []
    nombres = []
    for archivo in os.listdir("encodings"):
        if archivo.endswith(".pkl"):
            with open(f'encodings/{archivo}', 'rb') as f:
                data = pickle.load(f)
                encodings.append(data["encoding"])
                nombres.append(data["nombre"])
    return encodings, nombres

encodings_conocidos, nombres_conocidos = cargar_registros()

# Abrir cámara
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
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

        # Coordenadas del rostro actual
        top = y
        right = x + w
        bottom = y + h
        left = x

        # Redimensionar imagen y coordenadas
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

    # Registrar rostros
    if key == ord('r') and len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            print(f"[INFO] Registrando rostro {i+1} de {len(faces)}...")

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
                cv2.destroyAllWindows()  # Pausar para input
                nombre = input(f"Ingresa el nombre de la persona para rostro {i+1}: ")

                data = {"nombre": nombre, "encoding": face_encoding[0]}
                with open(f'encodings/{nombre}.pkl', 'wb') as f:
                    pickle.dump(data, f)
                print(f"Rostro de {nombre} registrado.")

                # Recargar registros
                encodings_conocidos, nombres_conocidos = cargar_registros()

                cap = cv2.VideoCapture(2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                print(f"No se pudo generar encoding facial para rostro {i+1}. Intenta nuevamente.")

    # Borrar registros
    elif key == ord('d'):
        cv2.destroyAllWindows()
        print("=== Registros guardados ===")
        for i, nombre in enumerate(nombres_conocidos):
            print(f"{i}: {nombre}")

        try:
            index_borrar = int(input("Ingresa el número del registro a borrar (o -1 para cancelar): "))
            if 0 <= index_borrar < len(nombres_conocidos):
                nombre_a_borrar = nombres_conocidos[index_borrar]
                archivo_a_borrar = f'encodings/{nombre_a_borrar}.pkl'

                if os.path.exists(archivo_a_borrar):
                    os.remove(archivo_a_borrar)
                    print(f"Registro de {nombre_a_borrar} borrado.")
                else:
                    print("Archivo no encontrado.")

                # Recargar registros
                encodings_conocidos, nombres_conocidos = cargar_registros()
            else:
                print("Cancelado.")
        except ValueError:
            print("Entrada inválida. Cancelado.")

        cap = cv2.VideoCapture(2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
