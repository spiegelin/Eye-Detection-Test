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

def classify_color(bgr_color):
    """
    Convierte un color BGR a HSV y clasifica el color del iris según una lógica
    basada en el valor (V) y el tono (H).
    
    Este método usa el espacio de color HSV para determinar el color basado en:
      - El tono (H) y la saturación (S).
      - El valor (V), que indica la luminosidad o intensidad del color.
    
    Se asume que:
      - Para tonos con H < 10 se consideran Marrones.
      - Para 10 <= H < 25:
          • Si V >= 150 se clasifica como Azules.
          • Si V < 140 se clasifica como Marrones.
          • Valores intermedios se asignan como Verdes.
      - Para 25 <= H < 40:
          • Si V >= 120 se clasifica como Azules.
          • Si V < 120 se clasifica como Verdes.
      - Para 40 <= H < 100 se asigna Azules.
      - Para otros casos se retorna "Indeterminado".
    
    Args:
        bgr_color (tuple): Color en formato BGR.
    
    Returns:
        str: "Marrones", "Verdes", "Azules" o "Indeterminado".
    """
    # Convierte el color BGR a HSV para facilitar la clasificación.
    hsv = cv2.cvtColor(np.uint8([[[bgr_color[0], bgr_color[1], bgr_color[2]]]]),
                       cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Clasificación de ojos marrones si el tono (H) es muy bajo.
    if h < 10:
        return "Marrones"
    # Clasificación de ojos azules o marrones con tonos intermedios.
    elif 10 <= h < 25:
        if v >= 150:
            return "Azules"
        elif v < 140:
            return "Marrones"
        else:
            return "Verdes"
    # Clasificación de ojos azules o verdes según el valor de (V).
    elif 25 <= h < 40:
        if v >= 120:
            return "Azules"
        else:
            return "Verdes"
    # Clasificación de ojos azules en tonos más elevados de (H).
    elif 40 <= h < 100:
        return "Azules"
    else:
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
