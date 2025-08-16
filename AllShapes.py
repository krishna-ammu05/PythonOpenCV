import cv2
import numpy as np

# Black canvas
img = np.zeros((400, 500, 3), dtype=np.uint8)

# Line
cv2.line(img, (50, 50), (450, 50), (0, 255, 0), 3)

# Rectangle
cv2.rectangle(img, (50, 80), (200, 150), (255, 0, 0), 2)

# Circle
cv2.circle(img, (350, 120), 40, (0, 0, 255), -1)

# Ellipse
cv2.ellipse(img, (120, 250), (80, 40), 0, 0, 360, (0,255,255), 2)

# Polygon
pts = np.array([[300,200],[400,200],[450,300],[350,350]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (255,255,0), 2)

# Filled Polygon
cv2.fillPoly(img, [pts], (0,128,255))

# Arrowed Line
cv2.arrowedLine(img, (50, 200), (200, 200), (255, 0, 255), 3)

# Marker
cv2.drawMarker(img, (250, 300), (0,255,0), cv2.MARKER_STAR, 40, 2)

# Text
cv2.putText(img, "OpenCV Shapes", (120, 380), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2)

# Show
cv2.imshow("All Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

