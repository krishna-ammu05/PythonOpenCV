import cv2 as cv
import numpy as np

img = cv.imread('pictures/dog.jpg') 
cv.imshow("Dog",img)

#Translation
def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions =( img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)

# -x --> Left
# -Y --> up
# x --> Right
# y -->Down
translated = translate(img, -100,100)
# translated1= translate(img, -100,-100)
# translated2= translate(img, 100, -100)
# translated3=translate(img, 100,100)
cv.imshow('Translated',translated)
# cv.imshow('Translated',translated1)
# cv.imshow('Translated',translated2)
# cv.imshow('Translated',translated3)

cv.waitKey(30)