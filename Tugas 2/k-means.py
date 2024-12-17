import numpy as np
import matplotlib.pyplot as plt
import cv2

# Membaca gambar
image_path = r'C:/Users/a516j/Downloads/download (1).jpg'
image = cv2.imread(image_path)

# Mengonversi BGR ke RGB (OpenCV membaca gambar dalam format BGR, sementara matplotlib mengharapkan format RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Menampilkan gambar asli
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')

# Mengubah bentuk gambar menjadi array 2D dengan nilai-nilai 3 komponen warna (RGB)
pixel_vals = image.reshape((-1, 3))

# Mengonversi tipe data ke float untuk k-means
pixel_vals = np.float32(pixel_vals)

# Mendefinisikan kriteria penghentian untuk k-means:
# Penghentian jika 100 iterasi telah dijalankan, atau akurasi (epsilon) mencapai 0,85
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Melakukan k-means clustering dengan jumlah cluster yang didefinisikan sebagai 3 (untuk segmentasi)
k = 3  # Jumlah cluster (dapat diubah sesuai kebutuhan)
_, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Mengonversi pusat (centers) ke nilai 8-bit
centers = np.uint8(centers)

# Memetakan label ke pusat (warna) untuk setiap pixel
segmented_data = centers[labels.flatten()]

# Mengubah bentuk data yang telah disegmentasi ke dimensi gambar asli
segmented_image = segmented_data.reshape((image.shape))

# Menampilkan gambar hasil segmentasi
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmentasi gambar dengan K-Means')

# Menampilkan kedua gambar
plt.show()
