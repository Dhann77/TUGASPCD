import cv2
import numpy as np
import matplotlib.pyplot as plt

img_color = cv2.imread('C:\\Users\\a516j\\Downloads\\WhatsApp Image 2024-10-08 at 21.59.47_ca82c59d.jpg', 1)
img_gray = cv2.imread('C:\\Users\\a516j\\Downloads\\WhatsApp Image 2024-10-08 at 21.59.47_ca82c59d.jpg', 0)

filter_median = cv2.medianBlur(img_gray, 3)
filter_gaussian = cv2.GaussianBlur(img_gray, (3, 3), 0)
kernel = np.ones((3, 3), np.uint8)
filter_min = cv2.erode(img_gray, kernel)
filter_max = cv2.dilate(img_gray, kernel)

# Fungsi untuk menampilkan gambar dan histogram
def plot_image_and_histogram(image, title, is_color=False):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
        
    plt.title(f'{title} Image', fontsize=14)
    plt.axis('off')  

    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title(f"Histogram of {title}", fontsize=14)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

# Menampilkan gambar dan histogram dari hasil filter
plot_image_and_histogram(img_color,"Original",is_color=True)
plot_image_and_histogram(filter_min, "Minimum Filter")
plot_image_and_histogram(filter_max, "Maximum Filter")
plot_image_and_histogram(filter_gaussian, "Gaussian Filter")
plot_image_and_histogram(filter_median, "Median Filter")

cv2.waitKey(0)
cv2.destroyAllWindows()
