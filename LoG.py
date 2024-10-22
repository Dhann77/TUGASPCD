import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar asli dalam grayscale
original_image = cv2.imread('D:/ss/charlie.jpeg', cv2.IMREAD_GRAYSCALE)

# Terapkan Gaussian Blur untuk mengurangi noise
gaussian_blur = cv2.GaussianBlur(original_image, (5, 5), 0)

# Terapkan Laplacian untuk mendeteksi tepi
laplacian_edges = cv2.Laplacian(gaussian_blur, cv2.CV_64F)

# Ambil nilai absolut untuk menghindari nilai negatif
laplacian_edges = np.abs(laplacian_edges)

# Tampilkan gambar
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_edges, cmap='gray')
plt.title('Laplacian of Gaussian (LoG)')
plt.axis('off')

plt.show()
