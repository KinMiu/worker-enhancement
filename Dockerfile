FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Perbaikan: Tambahkan libgomp1 untuk kestabilan internal multithreading OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
  libglib2.0-0 \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy file requirements terlebih dahulu (manfaatkan Docker layer caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code worker ke dalam container
COPY . .

# Jalankan script utama worker
CMD ["python", "main.py"]