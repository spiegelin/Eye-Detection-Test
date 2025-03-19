import cv2
import numpy as np

# Inicia la captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Cargar clasificadores Haar
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('./haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error al cargar clasificadores Haar.")
    exit()

# Rango HSV aproximado para colores de ojos
color_ranges = {
    'Marrones': ((0,   50,  50), (20, 255, 255)),
    'Avellana': ((20,  50,  50), (30, 255, 255)),
    'Verdes':   ((30,  50,  50), (85, 255, 255)),
    'Azules':   ((90,  50,  50), (130,255, 255)),
    'Grises':   ((0,    0,  50), (180, 50, 200))
}

def classify_color(bgr_color):
    """Convierte un color BGR a HSV y lo compara con los rangos de color."""
    hsv = cv2.cvtColor(np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]]),
                       cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = hsv
    print(hsv)

    # Caso especial: grises => saturación baja, valor intermedio
    if sat < 50 and 50 <= val <= 200:
        return "Grises"

    # Compara con el resto de rangos
    for name, (lower, upper) in color_ranges.items():
        if (lower[0] <= hue <= upper[0] and
            lower[1] <= sat <= upper[1] and
            lower[2] <= val <= upper[2]):

            return name
    return "Indeterminado"

def classify_iris_color(bgr_color):
    """Convierte un color BGR a HSV y lo compara con los rangos de color."""
    hsv = cv2.cvtColor(np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]]),
                       cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = hsv
    print(hsv)
    if hue < 2 or sat < 150 or val < 120:
        return "cafe"
    if hue <= 3 or 150 <= sat <= 180 or 120 <= val <= 170:
        return "verde"        
    else:
        return "azul"


while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80,80))

    for (x, y, w, h) in faces:
        # Dibuja la cara
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30,30))
        for (ex, ey, ew, eh) in eyes:
            # Dibuja rectángulo de ojo
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color= roi_color[ey:ey+eh, ex:ex+ew]

            # Difumina para reducir ruido
            eye_gray = cv2.GaussianBlur(eye_gray, (5,5), 0)

            # === Detectar círculo del iris con HoughCircles ===
            # Ajusta param1, param2, minRadius, maxRadius
            circles = cv2.HoughCircles(eye_gray, 
                                       cv2.HOUGH_GRADIENT, 
                                       dp=1.2, 
                                       minDist=ew//4,
                                       param1=80,  # umbral para canny
                                       param2=20,  # umbral para acumulador
                                       minRadius=ew//8,
                                       maxRadius=ew//2)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Toma la primera (o la más grande si prefieres)
                # Aquí tomamos la primera
                x_c, y_c, r_c = circles[0][0]

                # Dibuja el círculo en la imagen de depuración
                cv2.circle(eye_color, (x_c, y_c), r_c, (0, 255, 255), 2)

                # Crea máscara circular para el iris
                iris_mask = np.zeros_like(eye_gray, dtype=np.uint8)
                cv2.circle(iris_mask, (x_c, y_c), r_c, 255, -1)

                # === Aplica Sobel solo dentro del círculo ===
                sobelx = cv2.Sobel(eye_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(eye_gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel  = cv2.magnitude(sobelx, sobely)
                sobel  = np.uint8(np.clip(sobel, 0, 255))

                # Filtra sobel con la máscara del iris
                sobel_masked = cv2.bitwise_and(sobel, sobel, mask=iris_mask)

                # Threshold
                _, thresh = cv2.threshold(sobel_masked, 30, 255, cv2.THRESH_BINARY)

                # Busca contornos en la parte sobel del iris
                conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if conts:
                    cnt = max(conts, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    if area > 30:
                        cv2.drawContours(eye_color, [cnt], -1, (0,0,255), 2)

                        # Crea una máscara a partir del contorno
                        cont_mask = np.zeros_like(iris_mask)
                        cv2.drawContours(cont_mask, [cnt], -1, 255, -1)

                        # Color promedio
                        mean_bgr = cv2.mean(eye_color, mask=cont_mask)
                        b,g,r = mean_bgr[:3]

                        # Clasifica color
                        color_iris = classify_iris_color((b,g,r))
                        cv2.putText(frame, color_iris,
                                    (x+ex, y+ey-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,255,255),2)
            else:
                # Si no hay círculo, tal vez fallback
                pass

    cv2.imshow("Deteccion Ojos - HoughCircles", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
