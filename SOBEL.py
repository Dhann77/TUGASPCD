import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar original dalam grayscale
original_image = cv2.imread('D:/ss/charlie.jpeg', cv2.IMREAD_GRAYSCALE)

# Filter Sobel (deteksi tepi berdasarkan gradien)
sobel_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel di arah x
sobel_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel di arah y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Menggabungkan kedua gradien
    
# Menampilkan gambar
plt.figure(figsize=(10,5))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.figure()
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Combined (Magnitude)')
plt.axis('off')

plt.show()
