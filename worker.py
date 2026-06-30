import redis
import cv2
import numpy as np

# 1. Koneksi ke Redis lokal
r = redis.Redis(host='localhost', port=6379, db=0)

# 2. Setup Pub/Sub
pubsub = r.pubsub()
pubsub.subscribe('urken:frame:raw')

print("🚀 Python Image Worker berjalan. Menunggu frame di 'urken:frame:raw'...")

def enhance_image(img):
    # 1. Konversi BGR ke ruang warna LAB (L = Lightness, A, B = Color)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit: ambang batas kontras (3.0 adalah nilai standar yang bagus)
    # tileGridSize: membagi gambar jadi 8x8 kotak kecil untuk analisis lokal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 3. Gabungkan kembali channel L (yang sudah bersih) dengan channel A & B (warna asli)
    merged = cv2.merge((cl, a, b))
    
    # 4. Konversi kembali ke BGR agar bisa dikirim ke browser
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced

try:
    for message in pubsub.listen():
        if message['type'] == 'message':
            # a. Terima data biner JPEG dari Gateway
            raw_bytes = message['data']
            
            # b. Konversi biner ke format array NumPy yang bisa dibaca OpenCV
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is not None:
                # c. Lakukan proses hitung & perbaikan
                enhanced_img = enhance_image(img)

                # d. Encode kembali ke JPEG (Kualitas 70% agar ringan dikirim ke frontend)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                _, buffer = cv2.imencode('.jpg', enhanced_img, encode_param)
                
                # e. Publish kembali gambar yang sudah bersih ke Redis
                r.publish('urken:frame:enhanced', buffer.tobytes())

except KeyboardInterrupt:
    print("\nWorker dihentikan.")
finally:
    pubsub.close()