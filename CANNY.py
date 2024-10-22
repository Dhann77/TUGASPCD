import cv2
import matplotlib.pyplot as plt

# Baca gambar original dalam grayscale
original_image = cv2.imread('D:/ss/charlie.jpeg', cv2.IMREAD_GRAYSCALE)

# Filter Canny (deteksi tepi)
canny_edges = cv2.Canny(original_image, 100, 200)

# Menampilkan gambar
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.show()
