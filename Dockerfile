# Gunakan base image yang ringan
FROM python:3.10-slim

# Install dependencies sistem yang dibutuhkan OpenCV
# libgl1 untuk fungsi dasar, libglib2.0-0 untuk dependensi runtime
RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements terlebih dahulu agar proses build lebih cepat (caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sisa kode program
COPY . .

# Jalankan worker
CMD ["python", "main.py"]