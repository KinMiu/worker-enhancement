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
THRESHOLD_SENSITIVITY = 25    # Semakin kecil, semakin sensitif terhadap perubahan pixel
MIN_CONTOUR_AREA = 800        # Ukuran minimal objek bergerak (dalam pixel) agar dianggap "motion"
COOLDOWN_TELEGRAM = 60        # Jeda waktu 1 menit agar tidak banjir spam ke Telegram

# Variable Global untuk menyimpan state tracking gerakan
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
    """Membungkus pengiriman Telegram ke dalam Thread terpisah agar streaming utama tidak lag/patah"""
    thr = threading.Thread(target=_send_telegram_worker, args=(message, image_bytes))
    thr.start()

def detect_motion(current_frame):
    """Fungsi mendeteksi gerakan menggunakan Frame Differencing"""
    global prev_gray
    motion_detected = False

    # 1. Convert ke Grayscale & Blur tebal untuk meminimalkan false alarm dari bintik noise low-light
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray is None:
        prev_gray = gray
        return False

    # 2. Hitung perbedaan absolut antar frame
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

def enhance_image(img):
    """Fungsi khusus mengoptimalkan kondisi ruangan gelap gulita / low-light tanpa Infrared"""
    # 1. PEREKAYASAAN LOW-LIGHT (Denoising tingkat lanjut untuk memotong noise grain/semut)
    # Kita pakai h=10 untuk meredam grain/bintik-bintik di area pekat
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    
    # 2. LOW-LIGHT GAMMA CORRECTION (gamma < 1.0 untuk mendongkrak kegelapan secara non-linear)
    gamma = 0.5 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_boosted = cv2.LUT(denoised, table)
    
    # 3. Konversi ke LAB Color Space untuk optimasi kontras lokal
    lab = cv2.cvtColor(gamma_boosted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 4. Terapkan CLAHE adaptif dengan clipLimit sedang (1.5)
    # Ini supaya sisa siluet objek (seperti bantal/helm) bisa dipertegas bentuknya
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 5. Satukan kembali layer LAB
    merged = cv2.merge((cl, a, b))
    
    # 6. Kembalikan format ke BGR OpenCV
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
                    # A. DETEKSI GERAKAN (Gunakan frame asli bawaan kamera agar akurat)
                    is_moving = detect_motion(img)
                    
                    # B. JALANKAN LOGIKA LOW-LIGHT ENHANCEMENT
                    enhanced_img = enhance_image(img)
                    _, buffer = cv2.imencode('.jpg', enhanced_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    
                    # Publish hasil jernih ke Viewer Frontend lewat Redis channel
                    r.publish('urken:frame:enhanced', buffer.tobytes())
                    
                    # C. LOGIKA PENGIRIMAN ALERT KE TELEGRAM
                    if is_moving:
                        current_time = time.time()
                        # Cek apakah sudah melewati masa cooldown 1 menit (60 detik)
                        if current_time - last_telegram_time > COOLDOWN_TELEGRAM:
                            last_telegram_time = current_time
                            
                            waktu_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            caption = f"⚠️ *UrKen Alert!*\nTerdeteksi gerakan pada `{waktu_sekarang}`."
                            
                            print(f"🚨 Motion Detected! Mengirim gambar hasil optimasi low-light ke Telegram...")
                            # Kirim foto yang sudah di-brighten ke Telegram HP kamu
                            send_telegram_alert(caption, buffer.tobytes())
                            
                else:
                    print("❌ OpenCV Gagal men-decode biner.")
                    
            except Exception as e:
                print(f"💥 Error pada pemrosesan frame: {e}")

if __name__ == "__main__":
    run_worker()