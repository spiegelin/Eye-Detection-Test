import cv2
import numpy as np

# Inicia la captura de video desde la cámara predeterminada (ID 0).
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # Si no se puede abrir la cámara, se muestra un mensaje de error y se finaliza el programa.
    print("No se pudo abrir la cámara.")
    exit()

# Carga los clasificadores Haar para la detección de rostros y ojos.
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('./haarcascade_eye.xml')
if face_cascade.empty() or eye_cascade.empty():
    # Verifica que los clasificadores se hayan cargado correctamente.
    print("Error al cargar clasificadores Haar.")
    exit()

# Define los rangos de color en formato HSV para clasificar los colores de ojos.
# Se utiliza un diccionario para organizar los rangos de cada color.
color_ranges = {
    'Marrones': ((0,   50,  50), (25, 255, 100)),
    'Verdes':   ((20,  50,  50), (30, 255, 150)),
    'Azules':   ((31,  50,  50), (130,255, 255))
}

def classify_color(bgr_color):
    """
    Convierte un color de formato BGR a HSV y lo clasifica según los rangos definidos.

    Args:
        bgr_color (tuple): Color en formato BGR (Blue, Green, Red).

    Returns:
        str: Nombre del color clasificado o "Indeterminado" si no coincide con ningún rango.
    """
    # Convierte el color BGR a HSV para facilitar la comparación.
    hsv = cv2.cvtColor(np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]]),
                       cv2.COLOR_BGR2HSV)[0][0]

    # Recorre cada rango definido y verifica si el color se encuentra dentro de los límites.
    for name, (lower, upper) in color_ranges.items():
        if lower[0] <= hsv[0] <= upper[0] and lower[1] <= hsv[1] <= upper[1] and lower[2] <= hsv[2] <= upper[2]:
            print(f"{hsv} : {name}")
            return name
    # Retorna "Indeterminado" si el color no coincide con ninguno de los rangos.
    # Imprime el valor HSV para fines de depuración.
    print(f"{hsv} : Indeterminado")
    return "Indeterminado"

while True:
    # Lee un frame de la captura de video.
    ret, frame = cap.read()
    if not ret:
        # Si no se puede leer el frame, se muestra un mensaje y se termina el bucle.
        print("No se pudo leer el frame.")
        break

    # Convierte el frame a escala de grises para mejorar la detección de rostros y ojos.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostros en la imagen usando el clasificador Haar.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
    
    # Itera sobre cada rostro detectado.
    for (x, y, w, h) in faces:
        # Dibuja un rectángulo azul alrededor del rostro.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Define las regiones de interés (ROI) para el rostro en escala de grises y en color.
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Detecta ojos dentro de la región del rostro.
        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 10, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes:
            # Dibuja un rectángulo verde alrededor del ojo (opcional).
            cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Define la subregión del ojo que probablemente contenga el iris.
            # Se descarta la parte superior para evitar incluir la ceja.
            iris_x1 = int(ew * 0.25)
            iris_x2 = int(ew * 0.75)
            iris_y1 = int(eh * 0.4)
            iris_y2 = int(eh * 0.8)
            # Se verifica que las dimensiones del recorte sean válidas.
            if iris_y2 - iris_y1 <= 0 or iris_x2 - iris_x1 <= 0:
                continue  # Omite este ojo si el recorte no es correcto.
            
            # Extrae la subregión del iris en las imágenes en escala de grises y en color.
            iris_roi_gray = face_roi_gray[ey + iris_y1:ey + iris_y2, ex + iris_x1:ex + iris_x2]
            iris_roi_color = face_roi_color[ey + iris_y1:ey + iris_y2, ex + iris_x1:ex + iris_x2]
            
            # Aplica un suavizado Gaussiano para reducir el ruido en la imagen del iris.
            iris_gray_blurred = cv2.GaussianBlur(iris_roi_gray, (5, 5), 0)
            
            # Detecta el círculo correspondiente al iris usando la transformación de Hough.
            circles = cv2.HoughCircles(iris_gray_blurred,
                                       cv2.HOUGH_GRADIENT,
                                       dp=1.2,
                                       minDist=iris_roi_gray.shape[1] // 4,
                                       param1=80,
                                       param2=15,   # Umbral para el acumulador, ajustar según la calidad de la imagen.
                                       minRadius=int(iris_roi_gray.shape[1] * 0.1),
                                       maxRadius=int(iris_roi_gray.shape[1] * 0.5))
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Selecciona la primera circunferencia detectada como el iris.
                cx, cy, radius = circles[0][0]
                # Calcula las coordenadas del centro del círculo en relación a la imagen original.
                circle_center_x = ex + iris_x1 + cx
                circle_center_y = ey + iris_y1 + cy
                # Dibuja el círculo del iris en la imagen del rostro.
                cv2.circle(face_roi_color, (circle_center_x, circle_center_y), radius, (0, 255, 255), 2)
                
                # Crea una máscara circular para aislar la región del iris en la imagen.
                iris_mask = np.zeros_like(iris_roi_gray, dtype=np.uint8)
                cv2.circle(iris_mask, (cx, cy), radius, 255, -1)
                
                # Calcula el color promedio del iris usando la máscara.
                mean_color = cv2.mean(iris_roi_color, mask=iris_mask)[:3]
                # Clasifica el color del iris basándose en los rangos definidos.
                color_name = classify_color(mean_color)
                # Escribe el nombre del color detectado sobre la imagen del rostro.
                cv2.putText(frame, color_name, (x + ex, y + ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Muestra la imagen resultante con las detecciones en tiempo real.
    cv2.imshow("Deteccion de Iris y Color", frame)
    
    # Sale del bucle si se presiona la tecla 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura de video y cierra todas las ventanas abiertas.
cap.release()
cv2.destroyAllWindows()
