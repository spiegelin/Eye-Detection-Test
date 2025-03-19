# Iris Detection and Color Classification

This project uses Python and OpenCV to capture live video, detect faces and eyes in real-time, and identify the iris within each detected eye. It then analyzes the iris using various image processing techniques to determine its color. This can serve as a starting point for biometric analysis or interactive eye-based systems.

## Features

- **Real-Time Video Capture**: Connects to a specified camera.
- **Face & Eye Detection**: Uses Haar Cascade classifiers to locate faces and eyes.
- **Iris Detection**: Employs the Hough Circle Transform to detect the iris within an eye.
- **Edge Detection & Analysis**: Applies the Sobel filter to extract edges and refine iris boundaries.
- **Color Classification**: Converts BGR to HSV to compare with defined HSV ranges and classify iris color.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install dependencies with:

```bash
pip install opencv-python numpy
```

## How It Works

### 1. Video Capture Initialization
- **Camera Setup**: The script starts by opening a video stream from a specified camera (e.g., `cv2.VideoCapture(1)`). If the camera cannot be accessed, the program exits.

### 2. Loading Haar Cascade Classifiers
- **Face Detection**: Loads the `haarcascade_frontalface_default.xml` to detect faces.
- **Eye Detection**: Loads the `haarcascade_eye.xml` to detect eyes.
- The program checks if these classifiers are loaded correctly before proceeding.

### 3. HSV Color Range Definition
- A dictionary of HSV ranges is defined to classify iris colors into categories such as Brown, Hazel, Green, Blue, and Grey.
- These ranges are used later to compare the average color of the iris.

### 4. Color Classification Functions
- **`classify_color(bgr_color)`**: Converts a given BGR color to HSV and checks which predefined HSV range it falls into.
- **`classify_iris_color(bgr_color)`**: Applies specific HSV thresholds to classify the iris as green, brown, or blue.

### 5. Processing Each Video Frame
- **Frame Capture**: Reads the video frame-by-frame.
- **Grayscale Conversion**: Converts each frame to grayscale, which simplifies face and eye detection.

### 6. Face and Eye Detection
- **Face Detection**: Uses the face cascade to locate faces in the grayscale image.
- **ROI Extraction**: For each detected face, extracts a region of interest (ROI) from both the grayscale and color frames.
- **Eye Detection**: Within the face ROI, the eye cascade detects eyes and draws a rectangle around each detected eye.

### 7. Iris Detection Using HoughCircles
- **Preprocessing**: Applies a Gaussian Blur to the eye region to reduce noise.
- **Hough Circle Transform**: Detects circles in the blurred eye image, which is expected to correspond to the iris.
- If a circle is detected, it is drawn on the image.

### 8. Iris Analysis with Edge Detection
- **Mask Creation**: A circular mask is created based on the detected iris circle to isolate the iris region.
- **Sobel Filter**: Applies the Sobel filter (both x and y directions) to the iris region to detect edges.
- **Thresholding**: Converts the Sobel result into a binary image to highlight significant edges.
- **Contour Detection**: Finds the largest contour within the thresholded image, which likely corresponds to the iris boundary.

### 9. Color Averaging and Iris Color Classification
- **Color Averaging**: Computes the average BGR color inside the detected iris contour.
- **Classification**: The average color is then processed (converted to HSV) and classified into one of the predefined iris color categories.
- **Display**: The determined iris color is overlaid as text on the frame.

### 10. Display and Cleanup
- **Output Window**: Displays the processed frame with annotations (rectangles, circles, text).
- **Termination**: The program continues processing until the user presses the 'q' key, after which it releases the camera and closes all windows.

## Filters Used

- **Gaussian Blur**: Reduces noise in the eye region before edge detection.
- **Sobel Filter**: Computes the gradient magnitude in both x and y directions to detect edges within the iris.
- **Thresholding**: Converts the Sobel output to a binary image, making contour detection more effective.

## Running the Code

1. Ensure you have the required dependencies installed.
2. Place the Haar cascade XML files (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`) in the same directory as the script or update the paths accordingly.
3. Run the script using:

    ```bash
    python detection.py
    ```

Press 'q' to exit the video stream.

## License

This project is licensed under the MIT License.
```

Feel free to modify or expand the README to better suit your needs!