import cv2

# Cargar clasificadores pre-entrenados de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Abrir cámara (0 = cámara por defecto)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Dibujar rectángulo en el rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detección de ojos dentro del rostro
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Mostrar imagen
    cv2.imshow('Detección de rostros y ojos', frame)

    # Salir con la tecla ESC (27)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
