# 📷 Detección de Rostros y Clasificación de Color de Iris en Tiempo Real

Este proyecto utiliza Python y OpenCV para detectar rostros, ojos y clasificar el color del iris en tiempo real desde la cámara. El código implementa la detección de rostros y ojos utilizando clasificadores Haar, y luego clasifica el color del iris en "Marrones", "Verdes", "Azules" o "Indeterminado" basado en el espacio de color HSV.

## 🚀 Requisitos

- Python 3.x
- OpenCV
- NumPy

## 🛠️ Instrucciones de Instalación

Sigue estos pasos para configurar el entorno virtual y las dependencias necesarias:

### 1. Crear un entorno virtual
Abre tu terminal o línea de comandos y navega a la carpeta donde quieres clonar el repositorio. Luego, ejecuta el siguiente comando para crear un entorno virtual:

```bash
python -m venv venv
```

### 2. Activar el entorno virtual
- **En Windows**:
  ```bash
  .\venv\Scripts\activate
  ```
- **En macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar las dependencias
Con el entorno virtual activado, instala las bibliotecas necesarias:

```bash
pip install opencv-python numpy
```

### 4. Descarga los clasificadores Haar
Este proyecto utiliza clasificadores Haar preentrenados para la detección de rostros y ojos. Asegúrate de descargar los archivos necesarios desde los siguientes enlaces y guardarlos en el directorio del proyecto:

- [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- [haarcascade_eye.xml](https://github.com/opencv/opencv/tree/master/data/haarcascades)

Coloca estos archivos en la misma carpeta que tu script.

### 5. Ejecutar el código
Con el entorno virtual activado y las dependencias instaladas, ejecuta el script:

```bash
python nombre_del_script.py
```

Esto abrirá la cámara de tu dispositivo y comenzará la detección de rostros y ojos, mientras clasifica el color del iris en tiempo real. Para salir, presiona la tecla `q`.

## 🧑‍💻 Funcionalidad

- **Detección de Rostros y Ojos**: Utiliza clasificadores Haar para detectar rostros y ojos en tiempo real desde la cámara.
- **Clasificación del Color del Iris**: Después de detectar el ojo, el programa extrae el iris y lo clasifica como marrón, verde o azul, basándose en el valor promedio del color en el espacio HSV.
- **Interfaz Visual**: Muestra la cámara en vivo con rectángulos alrededor de los rostros y ojos detectados, y el nombre del color del iris sobre el rostro.

## ⚙️ Descripción del Código

1. **Captura de Video**: El script usa `cv2.VideoCapture` para acceder a la cámara del sistema y capturar frames en tiempo real.
2. **Clasificadores Haar**: Se cargan los clasificadores para detectar rostros y ojos.
3. **Función `classify_color`**: Convierte el color del iris de BGR a HSV y lo clasifica basado en el tono (H) y valor (V) del color.
4. **Detección de Iris**: Después de detectar los ojos, el código aplica la Transformación de Hough para identificar círculos, que corresponden al iris.
5. **Visualización**: Se dibujan rectángulos alrededor de los rostros y ojos, y se muestra el color clasificado sobre el rostro.

## 💡 Mejoras Futuras

- Mejorar la precisión de la detección en condiciones de baja luz.
- Implementar un sistema de clasificación de colores más avanzado para el iris.
- Agregar soporte para otros tipos de clasificación facial (como emociones).

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
