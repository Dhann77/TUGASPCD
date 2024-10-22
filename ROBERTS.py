import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar asli dalam grayscale
original_image = cv2.imread('D:/ss/charlie.jpeg', cv2.IMREAD_GRAYSCALE)

# Kernel Roberts
roberts_kernel_1 = np.array([[1, 0], [0, -1]])
roberts_kernel_2 = np.array([[0, 1], [-1, 0]])

# Terapkan kedua kernel Roberts
roberts_edge_1 = cv2.filter2D(original_image, -1, roberts_kernel_1)
roberts_edge_2 = cv2.filter2D(original_image, -1, roberts_kernel_2)

# Gabungkan hasil
roberts_combined = cv2.magnitude(roberts_edge_1.astype(float), roberts_edge_2.astype(float))

# Tampilkan gambar
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(roberts_combined, cmap='gray')
plt.title('Roberts Edge Detection')
plt.axis('off')

plt.show()
