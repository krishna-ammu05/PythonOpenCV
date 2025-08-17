import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create two simple images
img1 = np.zeros((300,300,3), dtype='uint8')
img2 = np.zeros((300,300,3), dtype='uint8')

# Draw shapes on them
cv.rectangle(img1, (50,50), (250,250), (255,0,0), -1)   # Blue rectangle
cv.circle(img2, (150,150), 100, (0,255,0), -1)          # Green circle

# Perform arithmetic operations
add_img = cv.add(img1, img2)
subtract_img = cv.subtract(img1, img2)
multiply_img = cv.multiply(img1, img2)
divide_img = cv.divide(img1, cv.add(img2, 1))  # add 1 to avoid division by zero

# Show all results using matplotlib
titles = ["Image 1 (Rectangle)", "Image 2 (Circle)", 
          "Addition", "Subtraction", "Multiplication", "Division"]

images = [img1, img2, add_img, subtract_img, multiply_img, divide_img]

plt.figure(figsize=(12,8))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
