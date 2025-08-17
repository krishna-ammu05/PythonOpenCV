import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load image
    img = cv2.imread("pictures/dog.jpg")
    if img is None:
        raise FileNotFoundError("Image not found. Please check the path.")

    # Apply filters
    gaussian = cv2.GaussianBlur(img, (7, 7), sigmaX=1.5)
    median = cv2.medianBlur(img, 7)
    average = cv2.blur(img, (7, 7))
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    kernel_sharpen = np.array([[0, -1,  0],
                               [-1,  5, -1],
                               [0, -1,  0]], np.float32)
    custom = cv2.filter2D(img, -1, kernel_sharpen)

    # List of results
    titles = [
        "Original",
        "Gaussian Blur (7x7, σ=1.5)",
        "Median Blur (ksize=7)",
        "Average Blur (7x7)",
        "Bilateral Filter (d=9, σColor=75, σSpace=75)",
        "Sharpening (filter2D)"
    ]
    images = [img, gaussian, median, average, bilateral, custom]

    # Plot all in one window
    plt.figure(figsize=(15, 10))
    for i, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(2, 3, i+1)   # 2 rows, 3 columns
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    # Adjust spacing (hspace controls row-gap)
    plt.subplots_adjust(hspace=0.4)  # Increase gap between rows

    plt.show()

if __name__ == "__main__":
    main()
