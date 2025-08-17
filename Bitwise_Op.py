import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create blank canvas
blank = np.zeros((400,400),dtype='uint8')

# Draw rectangle and circle
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

# Perform bitwise operations
bitwise_and = cv.bitwise_and(rectangle, circle)
bitwise_or = cv.bitwise_or(rectangle, circle)
bitwise_xor = cv.bitwise_xor(rectangle, circle)
bitwise_not = cv.bitwise_not(rectangle)

# Show all results using matplotlib
titles = ["Rectangle", "Circle", "Bitwise AND", "Bitwise OR", "Bitwise XOR", "Bitwise NOT"]
images = [rectangle, circle, bitwise_and, bitwise_or, bitwise_xor, bitwise_not]

plt.figure(figsize=(12,8))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
