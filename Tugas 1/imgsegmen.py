import cv2
import numpy as np

threshold = 0.5
kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
x_co = 0
y_co = 0
hsv = None
H = 125
S = 220
V = 170
thr_H = 180 * threshold
thr_S = 255 * threshold
thr_V = 255 * threshold

# Fungsi callback mouse untuk menangkap warna saat klik pada gambar
def on_mouse(event, x, y, flag, param):
    global x_co, y_co, H, S, V, hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        x_co = x
        y_co = y
        p_sel = hsv[y_co][x_co]
        H = p_sel[0]
        S = p_sel[1]
        V = p_sel[2]

cv2.namedWindow("image", 1)
cv2.namedWindow("mask", 2)
cv2.namedWindow("segmented", 3)

# Memuat gambar dari file
image_path = 'C:/Users/a516j/Downloads/download (1).jpg'  # Ganti dengan path gambar Anda
src = cv2.imread(image_path)

# Memeriksa apakah gambar berhasil dimuat
if src is None:
    print("Gagal memuat gambar.")
    exit()

# Menerapkan blur untuk menghaluskan gambar
src = cv2.blur(src, (3, 3))

# Konversi ke ruang warna HSV
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# Mengatur callback mouse untuk pemilihan warna
cv2.setMouseCallback("image", on_mouse, 0)

while True:
    # Mendefinisikan rentang threshold warna
    min_color = np.array([H - thr_H, S - thr_S, V - thr_V])
    max_color = np.array([H + thr_H, S + thr_S, V + thr_V])
    mask = cv2.inRange(hsv, min_color, max_color)

    # Transformasi morfologi untuk menutup lubang kecil
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)

    # Menampilkan nilai HSV dari titik yang dipilih
    cv2.putText(mask, f"H:{H} S:{S} V:{V}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), thickness=1)

    # Menampilkan mask dan gambar asli
    cv2.imshow("mask", mask)
    cv2.imshow("image", src)

    # Melakukan segmentasi pada area warna yang dipilih
    src_segmented = cv2.add(src, src, mask=mask)
    cv2.imshow("segmented", src_segmented)

    # Menghentikan loop saat tombol 'ESC' ditekan
    if cv2.waitKey(10) == 27:
        break

# Menutup semua jendela
cv2.destroyAllWindows()
