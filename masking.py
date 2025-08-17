import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show(title,image,cmap = None):
    plt.figure()
    if cmap:
        plt.imshow(image,cmap = cmap)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

img = cv.imread("pictures/dog.jpg")
img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)

mask = np.zeros(img.shape[:2],dtype="uint8")

center = (img.shape[1]//3+150 , img.shape[0]//3 )
radius = 200
cv.circle(mask,center,radius,255,-1)

masked = cv.bitwise_and(img_rgb,img_rgb,mask = mask)

show("Original Image",img_rgb)
show("mask",mask,cmap="gray")
show("Masked image",masked)