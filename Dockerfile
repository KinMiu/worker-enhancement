FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Install dependensi OS yang dibutuhkan oleh OpenCV jika menggunakan standard image
# (Sebagai antisipasi jika headless tetap membutuhkan beberapa shared library dasar)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Copy file requirements terlebih dahulu (manfaatkan Docker layer caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code worker ke dalam container
COPY . .

# Jalankan script utama worker
CMD ["python", "worker.py"]