import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r'C:/Users/Asus/Downloads/images.jpeg'  # Ganti dengan jalur gambar Anda
image = cv2.imread(image_path)

# Cek apakah gambar berhasil dimuat
if image is None:
    print("Error: Could not read the image. Please check the file path.")
    exit()

# Resize the image for easier processing
image = cv2.resize(image, (640, 480))

# Convert image to RGB (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Convert to float32
pixels = np.float32(pixels)

# Define criteria and apply KMeans
k = 3  # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to uint8 and reshape labels to the original image shape
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image_rgb.shape)

# Plot the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image (K-means)')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()
