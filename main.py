import redis
import cv2
import numpy as np
import os

# Gunakan env variable untuk host, default ke 'redis' (nama service docker)
# REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
r = redis.Redis(host='127.0.0.1', port=6380, db=0)
def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def run_worker():
    pubsub = r.pubsub()
    pubsub.subscribe('urken:frame:raw')
    print(f"🚀 Worker started on local. Waiting for frames...")

    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                raw_bytes = message['data']
                
                # Pastikan data biner tidak kosong
                if not raw_bytes:
                    print("⚠️ Menerima payload kosong.")
                    continue
                
                # Convert ke numpy array
                nparr = np.frombuffer(raw_bytes, np.uint8)
                
                # Decode ke gambar OpenCV
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    print(f"📸 Sukses decode frame! Ukuran: {img.shape}")
                    
                    # Proses enhancement
                    enhanced_img = enhance_image(img)
                    _, buffer = cv2.imencode('.jpg', enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    
                    # Publish kembali ke Redis
                    r.publish('urken:frame:enhanced', buffer.tobytes())
                else:
                    print("❌ OpenCV Gagal men-decode biner menjadi gambar (img is None).")
                    
            except Exception as e:
                print(f"💥 Error saat memproses frame: {e}")

if __name__ == "__main__":
    run_worker()