import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar original dalam grayscale
original_image = cv2.imread('D:/ss/charlie.jpeg', cv2.IMREAD_GRAYSCALE)

# Filter Prewitt di arah X
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_x = cv2.filter2D(original_image, -1, prewitt_kernel_x)

# Filter Prewitt di arah Y
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_y = cv2.filter2D(original_image, -1, prewitt_kernel_y)

# Gabungkan hasil Prewitt X dan Y
prewitt_combined = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))

# Menampilkan gambar
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(prewitt_x, cmap='gray')
plt.title('Prewitt X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(prewitt_y, cmap='gray')
plt.title('Prewitt Y')
plt.axis('off')

plt.figure()
plt.imshow(prewitt_combined, cmap='gray')
plt.title('Prewitt Combined')
plt.axis('off')

plt.show()
