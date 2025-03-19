import cv2
import numpy as np

# Inicia la captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Cargar clasificadores Haar para rostro y ojos
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('./haarcascade_eye.xml')
if face_cascade.empty() or eye_cascade.empty():
    print("Error al cargar clasificadores Haar.")
    exit()

# Definir rangos HSV para los 4 colores de ojos
color_ranges = {
    'Marrones': ((0,   50,  50), (25, 255, 100)),
    'Verdes':   ((20,  50,  50), (30, 255, 150)),
    'Azules':   ((31,  50,  50), (130,255, 255))
}

def classify_color(bgr_color):
    """Convierte BGR a HSV y clasifica según los rangos definidos."""
    hsv = cv2.cvtColor(np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]]), 
                       cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = hsv
    print(hsv)
    for name, (lower, upper) in color_ranges.items():
        if lower[0] <= hue <= upper[0] and lower[1] <= sat <= upper[1] and lower[2] <= val <= upper[2]:
            return name
    return "Indeterminado"

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80,80))
    
    for (x, y, w, h) in faces:
        # Dibuja el rectángulo del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Detecta ojos dentro del rostro
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10, minSize=(30,30))
        for (ex, ey, ew, eh) in eyes:
            # Dibuja el rectángulo del ojo (opcional)
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
            # Recortar una subregión del ojo que se espera contenga el iris
            # Se descarta la parte superior (posible ceja) y se toma la zona central-inferior.
            iris_x1 = int(ew * 0.25)
            iris_x2 = int(ew * 0.75)
            iris_y1 = int(eh * 0.4)
            iris_y2 = int(eh * 0.8)
            if iris_y2 - iris_y1 <= 0 or iris_x2 - iris_x1 <= 0:
                continue  # Evita recortes no válidos
            iris_roi_gray = face_roi_gray[ey+iris_y1:ey+iris_y2, ex+iris_x1:ex+iris_x2]
            iris_roi_color = face_roi_color[ey+iris_y1:ey+iris_y2, ex+iris_x1:ex+iris_x2]
            
            # Suaviza la imagen para reducir ruido
            iris_gray_blurred = cv2.GaussianBlur(iris_roi_gray, (5,5), 0)
            
            # Detecta el círculo del iris con HoughCircles sobre la subregión
            circles = cv2.HoughCircles(iris_gray_blurred,
                                       cv2.HOUGH_GRADIENT,
                                       dp=1.2,
                                       minDist=iris_roi_gray.shape[1]//4,
                                       param1=80,
                                       param2=15,   # Ajusta este parámetro según calidad
                                       minRadius=int(iris_roi_gray.shape[1]*0.1),
                                       maxRadius=int(iris_roi_gray.shape[1]*0.5))
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Tomamos la primera circunferencia detectada
                cx, cy, radius = circles[0][0]
                # Convertir las coordenadas del subrecorte a las coordenadas originales del rostro
                circle_center_x = ex + iris_x1 + cx
                circle_center_y = ey + iris_y1 + cy
                cv2.circle(face_roi_color, (circle_center_x, circle_center_y), radius, (0,255,255), 2)
                
                # Crea una máscara circular en la subregión
                iris_mask = np.zeros_like(iris_roi_gray, dtype=np.uint8)
                cv2.circle(iris_mask, (cx, cy), radius, 255, -1)
                
                # Calcula el color promedio en la zona del iris
                mean_color = cv2.mean(iris_roi_color, mask=iris_mask)[:3]
                color_name = classify_color(mean_color)
                # Muestra el nombre del color sobre el rostro
                cv2.putText(frame, color_name, (x+ex, y+ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
    cv2.imshow("Deteccion de Iris y Color", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
