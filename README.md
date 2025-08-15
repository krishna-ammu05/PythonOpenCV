# OpenCV – Open Source Computer Vision Library

##  Introduction
**OpenCV** (Open Source Computer Vision Library) is an open-source, cross-platform computer vision and machine learning software library.  
It provides tools for **real-time image processing**, **video analysis**, **object detection**, and more.

- **Written in:** C/C++ (with Python, Java, and other bindings)
- **License:** Apache 2.0 (Free for commercial and academic use)
- **Platforms:** Windows, Linux, macOS, Android, iOS
- **Website:** [https://opencv.org](https://opencv.org)

---

##  Features
- Image and video I/O
- Image processing (filters, transformations, morphology)
- Object detection and tracking
- Machine learning algorithms
- Deep learning (DNN module)
- Camera calibration and 3D vision
- GPU acceleration support

---

##  Applications
- Facial recognition
- Gesture recognition
- License plate detection
- Medical image analysis
- Industrial automation
- Augmented reality
- Motion tracking
- Robotics vision

---

##  Installation
```bash
# Basic version
pip install opencv-python

# With extra contributed modules (SIFT, SURF, etc.)
pip install opencv-contrib-python
```

---

##  Basic Usage
```python
import cv2

# Read an image
image = cv2.imread('sample.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges
edges = cv2.Canny(blurred, 50, 150)

# Display images
cv2.imshow('Original', image)
cv2.imshow('Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 🛠 OpenCV Functions Reference

### 1️ Image I/O
```python
cv2.imread(filename, flags)
cv2.imwrite(filename, img)
cv2.imdecode(buf, flags)
cv2.imencode(ext, img)
```

### 2️ Display
```python
cv2.imshow(winname, img)
cv2.waitKey(delay)
cv2.destroyAllWindows()
```

### 3️ Color Conversions
```python
cv2.cvtColor(img, code)
cv2.inRange(src, lower, upper)
```

### 4️ Geometric Transformations
```python
cv2.resize(img, dsize, fx, fy)
cv2.flip(img, flipCode)
cv2.rotate(img, rotateCode)
cv2.getRotationMatrix2D(center, angle, scale)
cv2.warpAffine(img, M, dsize)
cv2.warpPerspective(img, M, dsize)
```

### 5️ Thresholding
```python
cv2.threshold(src, thresh, maxval, type)
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
```

### 6️ Blurring & Filtering
```python
cv2.blur(img, ksize)
cv2.GaussianBlur(img, ksize, sigmaX)
cv2.medianBlur(img, ksize)
cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
cv2.filter2D(img, ddepth, kernel)
```

### 7️ Morphological Operations
```python
cv2.erode(img, kernel, iterations)
cv2.dilate(img, kernel, iterations)
cv2.morphologyEx(img, op, kernel)
```

### 8️ Edge Detection
```python
cv2.Canny(img, threshold1, threshold2)
cv2.Sobel(src, ddepth, dx, dy)
cv2.Laplacian(src, ddepth)
```

### 9️ Contours
```python
cv2.findContours(img, mode, method)
cv2.drawContours(img, contours, contourIdx, color, thickness)
cv2.contourArea(contour)
cv2.arcLength(contour, closed)
cv2.approxPolyDP(curve, epsilon, closed)
cv2.boundingRect(contour)
cv2.minEnclosingCircle(contour)
cv2.convexHull(points)
```

### 10 Drawing
```python
cv2.line(img, pt1, pt2, color, thickness)
cv2.rectangle(img, pt1, pt2, color, thickness)
cv2.circle(img, center, radius, color, thickness)
cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
```

### 1️1️ Video I/O
```python
cap = cv2.VideoCapture(index or filename)
cap.read()
cap.release()

cv2.VideoWriter(filename, fourcc, fps, frameSize)
cv2.VideoWriter_fourcc(*'XVID')
```

### 1️2️ Feature Detection
```python
cv2.cornerHarris(src, blockSize, ksize, k)
cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)

sift = cv2.SIFT_create()  # contrib module
```

### 1️3️ Object Detection
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
```

### 1️4️ Machine Learning
```python
cv2.ml.KNearest_create()
cv2.ml.SVM_create()
cv2.ml.DTrees_create()
```

### 1️5️ Deep Learning (DNN)
```python
net = cv2.dnn.readNetFromONNX('model.onnx')
net.setInput(blob)
output = net.forward()

cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop)
```

### 1️6️ Utilities
```python
cv2.split(img)
cv2.merge(channels)
cv2.add(img1, img2)
cv2.subtract(img1, img2)
cv2.bitwise_and(img1, img2)
cv2.bitwise_or(img1, img2)
cv2.bitwise_not(img)
```

---

##  Additional Resources
- [OpenCV Documentation](https://docs.opencv.org)
- [OpenCV GitHub Repository](https://github.com/opencv/opencv)
- [Learn OpenCV Blog](https://www.learnopencv.com)

---

##  License
This project uses **OpenCV**, which is licensed under the Apache 2.0 License.
