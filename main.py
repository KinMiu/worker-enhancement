import redis
import cv2
import numpy as np
import os

# HARDCODE: Langsung arahkan ke localhost dan port 6380 sesuai setup Golang kamu
r = redis.Redis(host='127.0.0.1', port=6380, db=0)

def enhance_image(img):
    # 1. Denoising menggunakan Bilateral Filter
    # Menghilangkan bintik/noise ruangan redup, tapi tetap menjaga garis wajah/objek agar tidak blur
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=35)
    
    # 2. Konversi BGR ke LAB untuk memisahkan kecerahan (L) dan warna (A, B)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. Terapkan CLAHE pada layer kecerahan (L)
    # clipLimit disesuaikan ke 2.0 agar kontras naik tapi tidak memicu noise baru
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 4. Satukan kembali layer LAB
    merged = cv2.merge((cl, a, b))
    
    # 5. Kembalikan format ke BGR OpenCV
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return result

def run_worker():
    pubsub = r.pubsub()
    pubsub.subscribe('urken:frame:raw')
    print("🚀 Worker started on 127.0.0.1:6380. Waiting for frames...")

    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                raw_bytes = message['data']
                
                # Cek jika payload kosong
                if not raw_bytes:
                    print("⚠️ Menerima payload kosong.")
                    continue
                
                # Ubah biner dari Redis menjadi numpy array
                nparr = np.frombuffer(raw_bytes, np.uint8)
                
                # Decode numpy array menjadi gambar matriks OpenCV
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # print(f"📸 Sukses decode frame! Ukuran: {img.shape}")
                    
                    # Eksekusi fungsi perbaikan kualitas gambar
                    enhanced_img = enhance_image(img)
                    
                    # Encode kembali ke JPEG dengan kualitas 70% untuk dikirim ke Viewer
                    _, buffer = cv2.imencode('.jpg', enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    
                    # Kirim hasil olahan ke channel urken:frame:enhanced
                    r.publish('urken:frame:enhanced', buffer.tobytes())
                else:
                    print("❌ OpenCV Gagal men-decode biner menjadi gambar (img is None).")
                    
            except Exception as e:
                print(f"💥 Error saat memproses frame: {e}")

if __name__ == "__main__":
    run_worker()