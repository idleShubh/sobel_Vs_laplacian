import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Compute gradient magnitude

# Apply Laplacian filter
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

# Convert back to uint8 for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)
laplacian = cv2.convertScaleAbs(laplacian)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 3, 2), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(2, 3, 3), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(2, 3, 4), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.subplot(2, 3, 5), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
plt.tight_layout()
plt.show()

# i find sobel to be better than laplacian
# sobel is better at detecting edges in both directions
# laplacian is better at detecting edges in one direction
