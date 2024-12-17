import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Memuat gambar dari file
image_path = 's'
src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Memeriksa apakah gambar berhasil dimuat
if src is None:
    print("Gagal memuat gambar.")
    exit()

# Sobel Edge Detection
sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)  # Tepi horizontal
sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)  # Tepi vertikal
sobel = cv2.magnitude(sobelx, sobely)  # Menghitung magnitudo dari Sobel

# Canny Edge Detection
canny = cv2.Canny(src, 100, 200)

# Prewitt Edge Detection
prewittx = ndimage.prewitt(src, axis=0).astype(np.float64)  # Tepi horizontal
prewitty = ndimage.prewitt(src, axis=1).astype(np.float64)  # Tepi vertikal
prewitt = cv2.magnitude(prewittx, prewitty)  # Menghitung magnitudo dari Prewitt

# Roberts Edge Detection
robertsx = np.array([[1, 0], [0, -1]])  # Kernel Roberts X
robertsy = np.array([[0, 1], [-1, 0]])  # Kernel Roberts Y
roberts_x = ndimage.convolve(src.astype(float), robertsx)
roberts_y = ndimage.convolve(src.astype(float), robertsy)
roberts = np.sqrt(roberts_x**2 + roberts_y**2)

# Laplacian of Gaussian (LoG) Edge Detection
log = cv2.GaussianBlur(src, (3, 3), 0)
log = cv2.Laplacian(log, cv2.CV_64F)

# Zero Crossing Detection
log_abs = np.absolute(log)
zero_crossing = np.zeros_like(log_abs)
zero_crossing[log_abs < 0.1] = 255  # Thresholding untuk mencari nol

# Menampilkan gambar
plt.figure(figsize=(14, 10))

# Gambar asli
plt.subplot(3, 3, 1)
plt.imshow(src, cmap='gray')
plt.title('Gambar Asli')
plt.axis('off')

# Deteksi tepi Sobel
plt.subplot(3, 3, 2)
plt.imshow(sobel, cmap='gray')
plt.title('Deteksi Tepi Sobel')
plt.axis('off')

# Deteksi tepi Canny
plt.subplot(3, 3, 3)
plt.imshow(canny, cmap='gray')
plt.title('Deteksi Tepi Canny')
plt.axis('off')

# Deteksi tepi Prewitt
plt.subplot(3, 3, 4)
plt.imshow(prewitt, cmap='gray')
plt.title('Deteksi Tepi Prewitt')
plt.axis('off')

# Deteksi tepi Roberts
plt.subplot(3, 3, 5)
plt.imshow(roberts, cmap='gray')
plt.title('Deteksi Tepi Roberts')
plt.axis('off')

# Deteksi tepi Laplacian of Gaussian (LoG)
plt.subplot(3, 3, 6)
plt.imshow(log, cmap='gray')
plt.title('Deteksi Tepi LoG (Laplacian of Gaussian)')
plt.axis('off')

# Deteksi tepi Zero Crossing
plt.subplot(3, 3, 7)
plt.imshow(zero_crossing, cmap='gray')
plt.title('Deteksi Tepi Zero Crossing')
plt.axis('off')

# Menampilkan semua gambar
plt.tight_layout()
plt.show()
