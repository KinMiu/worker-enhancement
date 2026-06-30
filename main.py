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

# Inisialisasi Redis client
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Motion Detection Config
THRESHOLD_SENSITIVITY = 20    # Diturunkan sedikit agar lebih sensitif di kondisi terang
MIN_CONTOUR_AREA = 1000       # Ditingkatkan ke 1000 biar tidak gampang false alarm karena bayangan kecil
COOLDOWN_TELEGRAM = 60        # Jeda waktu 1 menit antar notifikasi Telegram

# Variable Global untuk menyimpan state tracking gerakan
prev_gray = None
last_telegram_time = 0

def _send_telegram_worker(message, image_bytes):
    """Fungsi internal di background thread untuk kirim alert Telegram"""
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
            print("🚀 Notifikasi Telegram berhasil dikirim!")
        else:
            print(f"⚠️ Gagal kirim Telegram: {response.text}")
    except Exception as e:
        print(f"💥 Error Telegram API: {e}")

def send_telegram_alert(message, image_bytes=None):
    """Membungkus pengiriman Telegram ke dalam Thread agar tidak mengganggu aliran stream utama"""
    thr = threading.Thread(target=_send_telegram_worker, args=(message, image_bytes))
    thr.start()

def detect_motion(current_frame):
    """Fungsi mendeteksi gerakan menggunakan Frame Differencing"""
    global prev_gray
    motion_detected = False

    # 1. Convert ke Grayscale & Blur ringan (Sangat cepat)
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    if prev_gray is None:
        prev_gray = gray
        return False

    # 2. Hitung perbedaan antar frame
    frame_delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_delta, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 3. Cari kontur gerakan
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            motion_detected = True
            break 

    prev_gray = gray
    return motion_detected

def optimize_bright_mode(img):
    """Fungsi super ringan untuk memperjelas & mempertajam kondisi ruangan terang"""
    # 1. Gunakan Unsharp Masking untuk menaikkan ketajaman (Sharpening) detail wajah/benda
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    
    # 2. Konversi ke LAB untuk menaikkan kontras lokal secara instan & ringan
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Clip limit kecil (1.2) agar warna tidak pecah/lebay, proses super cepat
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return result

def run_worker():
    global last_telegram_time
    pubsub = r.pubsub()
    pubsub.subscribe('urken:frame:raw')
    print(f"🚀 High-Speed Worker Started! Terhubung ke Redis.")
    print(f"☀️ Mode Terang Aktif. Mengoptimalkan FPS Real-Time.")

    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                raw_bytes = message['data']
                if not raw_bytes:
                    continue
                
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # A. DETEKSI GERAKAN (Real-time tracking)
                    is_moving = detect_motion(img)
                    
                    # B. OPTIMASI GAMBAR MODE TERANG (Super cepat, tanpa bottleneck)
                    enhanced_img = optimize_bright_mode(img)
                    
                    # Encode kualitas ditingkatkan ke 80 karena prosesnya sudah sangat enteng
                    _, buffer = cv2.imencode('.jpg', enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    
                    # Broadcast langsung ke Frontend viewer
                    r.publish('urken:frame:enhanced', buffer.tobytes())
                    
                    # C. TELEGRAM NOTIFICATION
                    if is_moving:
                        current_time = time.time()
                        if current_time - last_telegram_time > COOLDOWN_TELEGRAM:
                            last_telegram_time = current_time
                            
                            waktu_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            caption = f"⚠️ *UrKen Alert!*\nTerdeteksi gerakan pada `{waktu_sekarang}`."
                            
                            print(f"🚨 Motion Detected! Mengirim snapshot bening ke Telegram...")
                            send_telegram_alert(caption, buffer.tobytes())
                            
                else:
                    print("❌ OpenCV Gagal men-decode biner.")
                    
            except Exception as e:
                print(f"💥 Error pada pemrosesan frame: {e}")

if __name__ == "__main__":
    run_worker()