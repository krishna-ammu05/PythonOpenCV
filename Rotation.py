import cv2 as cv
import numpy as np

# Read image
img = cv.imread("pictures/dog.jpg")
cv.imshow("Image", img)

# Function to rotate
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

# Rotate by 45 degrees
rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)

cv.waitKey(0)
cv.destroyAllWindows()
