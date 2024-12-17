Pelacakan Aliran Optik Lucas-Kanade

Gambaran Umum
Proyek ini mengimplementasikan sistem deteksi wajah dan pelacakan aliran optik menggunakan OpenCV. Sistem ini mendeteksi wajah dalam aliran video (dari file atau kamera), melacak titik-titik pada wajah yang terdeteksi menggunakan Lucas-Kanade Optical Flow, dan menampilkan hasil pelacakan secara real-time.

Fitur
- Mendeteksi wajah menggunakan Haar Cascade Classifier.
- Melacak fitur wajah menggunakan Lucas-Kanade Optical Flow.
- Mendukung reset titik pelacakan secara dinamis.
- Mendukung input dari file video atau kamera langsung.

Persyaratan
- Python 3.x
- OpenCV (cv2)
- NumPy

Instalasi
1. Clone repositori:
   bash
   git clone https://github.com/your-repository/lucas-kanade-tracking.git
   cd lucas-kanade-tracking
   

2. Instal dependensi yang diperlukan:
   ```bash
   pip install numpy opencv-python
   ```

3. Pastikan file Haar Cascade `haarcascade_frontalface_default.xml` ada di direktori yang sama atau tentukan jalurnya di dalam skrip.

Penggunaan
Jalankan skrip dengan opsi berikut:

```bash
python script.py [-v VIDEO] [-n MAX_CORNERS]
```

Opsi
- `-v, --video`: Jalur ke file video. Jika tidak disebutkan, skrip akan menggunakan kamera.
- `-n, --max_corners`: Jumlah maksimum titik sudut yang akan dilacak (default: 100).

Contoh
- Jalankan dengan input kamera:
  ```bash
  python script.py
  ```
- Jalankan dengan file video:
  ```bash
  python script.py -v path/to/video.mp4
  ```
- Jalankan dengan jumlah titik sudut tertentu untuk dilacak:
  ```bash
  python script.py -n 50
  ```

Kontrol
- `q`: Keluar dari program.
- `r`: Reset titik pelacakan.

Cara Kerja
1. **Deteksi Wajah**: Program menggunakan Haar Cascade Classifier untuk mendeteksi wajah terbesar dalam frame.
2. **Pelacakan Aliran Optik**: Setelah wajah terdeteksi, fitur-fitur baik (sudut-sudut) untuk dilacak diambil menggunakan detektor sudut Shi-Tomasi. Fitur-fitur ini dilacak dari frame ke frame menggunakan algoritma Lucas-Kanade Optical Flow.
3. **Visualisasi**: Titik-titik pelacakan digambar pada frame, dan overlay teks menampilkan status saat ini.

Masalah yang Diketahui
- File Haar Cascade harus ada di direktori kerja atau ditentukan dengan jalur yang benar.
- Pelacakan mungkin gagal jika wajah keluar dari frame atau berubah orientasi secara signifikan.

Referensi
- [Dokumentasi OpenCV](https://docs.opencv.org/)
- [Lucas-Kanade Optical Flow](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)

Lisensi
Proyek ini dilisensikan di bawah Lisensi MIT.

