"""
Skrip Inisialisasi Database PostgreSQL
Membangun 6 tabel skema terstruktur sesuai Foreign Key
dan menjalankan registrasi wajah mahasiswa dari dataset sah.
"""

import sys
import config
from database import test_connection, create_tables
from registrasi_wajah import registrasi_dataset_wajah


def inisialisasi_database(folder_dataset="data_wajah"):
    print("=" * 68)
    print(" SETUP STRUKTUR DATABASE POSTGRESQL (6 TABEL SKEMA RESMI)")
    print("=" * 68)

    # 1. Uji koneksi ke PostgreSQL
    terhubung, pesan_koneksi = test_connection()
    if not terhubung:
        print(f"\n[ERROR] Gagal terhubung ke database PostgreSQL!")
        print(f"Detail: {pesan_koneksi}")
        print("\nSilakan periksa konfigurasi di file .env:")
        print(f"  DB_HOST = {config.DB_HOST}")
        print(f"  DB_PORT = {config.DB_PORT}")
        print(f"  DB_NAME = {config.DB_NAME}")
        print(f"  DB_USER = {config.DB_USER}")
        print(f"  DB_PASSWORD = ********")
        sys.exit(1)

    print(f" [+] Status Koneksi: {pesan_koneksi}")

    # 2. Buat skema 6 tabel
    print("\n[1/2] Membuat struktur 6 tabel PostgreSQL sesuai urutan Foreign Key...")
    sukses_skema, pesan_skema = create_tables(drop_existing=False)
    print(f" [+] {pesan_skema}")

    # 3. Registrasi dataset wajah mahasiswa sah
    print("\n[2/2] Memproses pendaftaran mahasiswa dan ekstraksi wajah ke PostgreSQL...")
    registrasi_dataset_wajah(folder_dataset=folder_dataset, simpan_ke_db=True)

    print("\n" + "=" * 68)
    print(" INISIALISASI DATABASE & REGISTRASI DATASET SELESAI DENGAN SUKSES!")
    print("=" * 68)


if __name__ == "__main__":
    inisialisasi_database()
