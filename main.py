import redis
import cv2
import numpy as np
import os

# HARDCODE: Langsung arahkan ke localhost dan port 6380 sesuai setup Golang kamu
r = redis.Redis(host='127.0.0.1', port=6380, db=0)

def enhance_image(img):
    # 1. Denoising menggunakan Bilateral Filter
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=25, sigmaSpace=25)
    
    # 2. GAMMA CORRECTION (Menurunkan kecerahan yang overexposed)
    # Nilai gamma > 1.0 akan menggelapkan gambar secara non-linear (menyelamatkan area silau)
    gamma = 1.5 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(denoised, table)
    
    # 3. Konversi ke LAB Color Space
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 4. Terapkan CLAHE yang lebih lembut (Gentle CLAHE)
    # clipLimit diturunkan dari 2.0 ke 1.2 agar area wajah tidak terlalu putih/silau
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 5. Satukan kembali layer LAB
    merged = cv2.merge((cl, a, b))
    
    # 6. Kembalikan format ke BGR OpenCV
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