"""
Konfigurasi Terpusat Sistem Absensi Pengenalan Wajah
Membaca variabel lingkungan dari file .env secara aman.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat file .env jika tersedia
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# ==================== KONEKSI POSTGRESQL ====================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "zaky123")

# ==================== AMBANG BATAS AI / FACE RECOGNITION ====================
# Threshold Cosine Similarity untuk membedakan wajah dikenal vs tidak dikenal
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
MIN_DIFF_THRESHOLD = float(os.getenv("MIN_DIFF_THRESHOLD", "0.05"))
DETECTION_SIZE = int(os.getenv("DETECTION_SIZE", "640"))

# ==================== GEOFENCING (RADIUS KELAS) ====================
KELAS_LAT = float(os.getenv("KELAS_LAT", "-6.9828"))
KELAS_LON = float(os.getenv("KELAS_LON", "110.4091"))
BATAS_RADIUS_METER = float(os.getenv("BATAS_RADIUS_METER", "50.0"))

# Mockup GPS mahasiswa saat pengujian
GPS_MOCK_LAT = float(os.getenv("GPS_MOCK_LAT", "-6.9828"))
GPS_MOCK_LON = float(os.getenv("GPS_MOCK_LON", "110.4091"))

# ==================== COOLDOWN & ANTI-SPAM ABSENSI ====================
COOLDOWN_ABSEN_DETIK = int(os.getenv("COOLDOWN_ABSEN_DETIK", "15"))

# ==================== LOKASI DIREKTORI & FILE ====================
DATA_UJI_DIR = BASE_DIR / "data_wajah"
FOTO_UJI_DIR = BASE_DIR / "foto_uji"
MODELS_CACHE_DIR = BASE_DIR / "models_cache"
MODEL_ANTISPOOF = BASE_DIR / "antispoof.onnx"
WEIGHTS_YOLO = BASE_DIR / "weights" / "yolov11n-face.onnx"


# ==================== FUNGSI PEMBANTU UNICODE OPEN_CV ====================
import cv2
import numpy as np


def imread_unicode(file_path):
    """
    Membaca gambar dengan aman pada path bersimbol non-ASCII / Unicode di Windows
    (seperti OneDrive/ドキュメント).
    """
    try:
        with open(file_path, "rb") as f:
            bytes_data = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
    except Exception:
        return None


