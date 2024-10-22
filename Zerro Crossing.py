import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar asli dalam grayscale
original_image = cv2.imread('D:/ss/charlie.jpeg', cv2.IMREAD_GRAYSCALE)

# Terapkan Gaussian Blur untuk mengurangi noise
gaussian_blur = cv2.GaussianBlur(original_image, (5, 5), 0)

# Terapkan Laplacian untuk mendeteksi tepi
laplacian_edges = cv2.Laplacian(gaussian_blur, cv2.CV_64F)

# Ambil nilai absolut dari Laplacian
laplacian_edges = np.abs(laplacian_edges)

# Buat gambar untuk Zero-Crossing
zero_crossing = np.zeros_like(laplacian_edges, dtype=np.uint8)

# Deteksi zero-crossing
for i in range(1, laplacian_edges.shape[0] - 1):
    for j in range(1, laplacian_edges.shape[1] - 1):
        # Periksa jika ada perubahan tanda
        if (laplacian_edges[i, j] > 0 and 
            (laplacian_edges[i-1, j] < 0 or 
             laplacian_edges[i+1, j] < 0 or 
             laplacian_edges[i, j-1] < 0 or 
             laplacian_edges[i, j+1] < 0)):
            zero_crossing[i, j] = 255  # Tandai titik zero-crossing

# Tampilkan gambar
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(zero_crossing, cmap='gray')
plt.title('Zero-Crossing Edge Detection')
plt.axis('off')

plt.show()
