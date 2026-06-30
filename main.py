import redis
import cv2
import numpy as np
import os
import requests
import time
import threading
from datetime import datetime

# --- CONFIGURATION (Membaca dari Environment Variables) ---
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6380))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Menggunakan variabel env untuk inisialisasi Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Motion Detection Config
THRESHOLD_SENSITIVITY = 25    # Semakin kecil, semakin sensitif terhadap perubahan pixel
MIN_CONTOUR_AREA = 800        # Ukuran minimal objek bergerak (dalam pixel) agar dianggap "motion"
COOLDOWN_TELEGRAM = 60        # Jeda waktu 1 menit agar tidak spam

# Variable Global untuk menyimpan status
prev_gray = None
last_telegram_time = 0

def _send_telegram_worker(message, image_bytes):
    """Fungsi internal yang berjalan di background thread untuk mengirim ke Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Kredensial Telegram tidak ditemukan di environment variable!")
        return

    try:
        if image_bytes is not None:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            files = {'photo': ('motion.jpg', image_bytes, 'image/jpeg')}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, data=data, files=files, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, data=data, timeout=10)
            
        if response.status_code == 200:
            print("🚀 Notifikasi Telegram berhasil dikirim lewat background thread!")
        else:
            print(f"⚠️ Gagal kirim Telegram: {response.text}")
    except Exception as e:
        print(f"💥 Error Telegram API: {e}")

def send_telegram_alert(message, image_bytes=None):
    """Membungkus pengiriman Telegram ke dalam Thread terpisah agar streaming utama tidak lag"""
    thr = threading.Thread(target=_send_telegram_worker, args=(message, image_bytes))
    thr.start()

def detect_motion(current_frame):
    """Fungsi mendeteksi gerakan menggunakan Frame Differencing"""
    global prev_gray
    motion_detected = False

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray is None:
        prev_gray = gray
        return False

    frame_delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_delta, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            motion_detected = True
            break 

    prev_gray = gray
    return motion_detected

def enhance_image(img):
    denoised = cv2.bilateralFilter(img, d=7, sigmaColor=25, sigmaSpace=25)
    
    gamma = 1.5 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(denoised, table)
    
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return result

def run_worker():
    global last_telegram_time
    pubsub = r.pubsub()
    pubsub.subscribe('urken:frame:raw')
    print(f"🚀 Worker started. Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    print(f"🔒 Anti-Spam Active: Maksimal 1 pesan per {COOLDOWN_TELEGRAM} detik.")

    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                raw_bytes = message['data']
                if not raw_bytes:
                    continue
                
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    is_moving = detect_motion(img)
                    enhanced_img = enhance_image(img)
                    _, buffer = cv2.imencode('.jpg', enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    
                    r.publish('urken:frame:enhanced', buffer.tobytes())
                    
                    if is_moving:
                        current_time = time.time()
                        if current_time - last_telegram_time > COOLDOWN_TELEGRAM:
                            last_telegram_time = current_time
                            
                            waktu_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            caption = f"⚠️ *UrKen Alert!*\nTerdeteksi gerakan pada `{waktu_sekarang}`."
                            
                            print(f"🚨 Motion Detected! Mengirim alert terbaru ke Telegram...")
                            send_telegram_alert(caption, buffer.tobytes())
                            
                else:
                    print("❌ OpenCV Gagal men-decode biner.")
                    
            except Exception as e:
                print(f"💥 Error: {e}")

if __name__ == "__main__":
    run_worker()