"""
Modul Registrasi Wajah Mahasiswa ke Database PostgreSQL
Menerapkan Prinsip: SATU FOTO VALID = SATU EMBEDDING = SATU BARIS DATABASE

Membaca foto dari folder data_wajah / data_uji secara individual,
mengekstrak face embedding berdimensi 512 via InsightFace, dan menyimpannya
ke tabel t_datawajah tanpa menghapus data wajah yang sudah ada.
"""

import os
import csv
import json
import argparse
from pathlib import Path
import numpy as np
from insightface.app import FaceAnalysis

import config
from database import (
    get_connection,
    close_connection,
    get_mahasiswa,
    save_face_data,
    test_connection
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def muat_metadata_identitas(folder_path: Path):
    """
    Mencari dan membaca file metadata identitas mahasiswa (CSV atau JSON)
    jika tersedia di folder data atau root project.
    Format return: {nim: {"nama": str, "tingkat": str, "prodi": str}}
    """
    metadata = {}

    # 1. Cek file CSV (mahasiswa.csv)
    opsi_csv = [
        folder_path / "mahasiswa.csv",
        config.BASE_DIR / "mahasiswa.csv",
        folder_path.parent / "mahasiswa.csv",
        config.BASE_DIR / "data_wajah" / "mahasiswa.csv",
        config.BASE_DIR / "data_uji" / "mahasiswa.csv"
    ]
    for path_csv in opsi_csv:
        if path_csv.exists():
            print(f" [INFO] Memuat metadata identitas dari CSV: {path_csv.name}")
            with open(path_csv, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nim = row.get("NIM") or row.get("nim")
                    nama = row.get("nama") or row.get("Nama")
                    tingkat = row.get("tingkat") or row.get("Tingkat", "Tingkat 3")
                    prodi = row.get("Program_studi") or row.get("program_studi") or row.get("prodi", "Rekayasa Perangkat Lunak")
                    if nim and nama:
                        metadata[str(nim).strip()] = {
                            "nama": str(nama).strip(),
                            "tingkat": str(tingkat).strip(),
                            "prodi": str(prodi).strip()
                        }
            if metadata:
                return metadata

    # 2. Cek file JSON jika CSV tidak ditemukan
    opsi_json = [
        folder_path / "mahasiswa.json",
        config.BASE_DIR / "data_wajah" / "mahasiswa.json",
        config.BASE_DIR / "mahasiswa.json"
    ]
    for path_json in opsi_json:
        if path_json.exists():
            print(f" [INFO] Memuat metadata identitas dari JSON: {path_json.name}")
            with open(path_json, mode="r", encoding="utf-8") as f:
                data_json = json.load(f)
                for nim, val in data_json.items():
                    metadata[str(nim).strip()] = {
                        "nama": val.get("nama", "").strip(),
                        "tingkat": val.get("tingkat", "Tingkat 3").strip(),
                        "prodi": val.get("prodi", "Rekayasa Perangkat Lunak").strip()
                    }
            if metadata:
                return metadata

    return metadata


def inisialisasi_insightface():
    """Inisialisasi model ekstraksi fitur wajah InsightFace (buffalo_l)."""
    print(" [AI] Memuat InsightFace (buffalo_l)...")
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(config.MODELS_CACHE_DIR),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=-1, det_size=(config.DETECTION_SIZE, config.DETECTION_SIZE))
    return app


def pastikan_mahasiswa_terdaftar(nim, metadata):
    """
    Memastikan data mahasiswa terdaftar di tabel master t_mahasiswa.
    Jika belum ada, menggunakan data sah dari CSV/JSON.
    DILARANG membuat data identitas fiktif jika tidak ada di sumber sah.
    """
    mhs_db = get_mahasiswa(nim)
    if mhs_db:
        return True, mhs_db["nama"], mhs_db["tingkat"], mhs_db["program_studi"]

    if nim in metadata:
        identitas = metadata[nim]
        nama = identitas["nama"]
        tingkat = identitas["tingkat"]
        prodi = identitas["prodi"]

        conn = get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO t_mahasiswa (nim, nama, tingkat, program_studi)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (nim) DO NOTHING;
                    """, (nim, nama, tingkat, prodi))
            return True, nama, tingkat, prodi
        finally:
            close_connection(conn)

    return False, None, None, None


def registrasi_dataset_wajah(folder_dataset: str = None, simpan_ke_db: bool = True, izinkan_duplikat_foto: bool = False):
    """
    Memproses setiap foto mahasiswa secara INDIVIDUAL:
    SATU FOTO VALID = SATU EMBEDDING = SATU BARIS DATABASE

    - Tidak menggabungkan embedding dengan np.mean()
    - Tidak menghapus baris lama milik NIM tersebut
    - Menolak foto yang memiliki 0 atau >1 wajah
    - Melaporkan 7 metrik rekapitulasi secara rinci
    """
    target_dir = Path(folder_dataset) if folder_dataset else None

    if not target_dir or not target_dir.exists():
        opsi_dir = [
            config.BASE_DIR / "data_wajah",
            config.BASE_DIR / "data_uji",
            Path("data_wajah"),
            Path("data_uji")
        ]
        for d in opsi_dir:
            if d.exists() and d.is_dir():
                target_dir = d
                break

    if not target_dir or not target_dir.exists():
        raise FileNotFoundError("Direktori dataset foto (data_wajah / data_uji) tidak ditemukan.")

    print("=" * 70)
    print(f" REGISTRASI DATASET INDIVIDUAL WAJAH DARI: {target_dir.resolve()}")
    print("=" * 70)

    metadata_identitas = muat_metadata_identitas(target_dir)
    app = inisialisasi_insightface()

    # Metrik Rekapitulasi
    total_mahasiswa_diproses = 0
    total_mahasiswa_gagal = 0
    total_foto_ditemukan = 0
    total_foto_valid = 0
    total_foto_gagal = 0
    total_data_baru_tersimpan = 0
    total_data_duplikat_dilewati = 0

    daftar_tersimpan = []
    daftar_gagal = []

    subfolders = [d for d in target_dir.iterdir() if d.is_dir()]
    if not subfolders:
        print(f" [!] Tidak ditemukan subfolder NIM di {target_dir}")
        return {}

    for subfolder in subfolders:
        nim = subfolder.name.strip()
        total_mahasiswa_diproses += 1

        terdaftar, nama, tingkat, prodi = pastikan_mahasiswa_terdaftar(nim, metadata_identitas)
        if not terdaftar:
            total_mahasiswa_gagal += 1
            daftar_gagal.append({
                "nim": nim,
                "alasan": f"NIM {nim} tidak memiliki data identitas yang sah di CSV/JSON (Registrasi ditolak)."
            })
            print(f"\n [DITOLAK] NIM: {nim} -> Tidak memiliki data identitas sah.")
            continue

        print(f"\n[*] Memproses: {nim} - {nama} [{prodi}]")

        file_foto_list = [f for f in subfolder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        total_foto_ditemukan += len(file_foto_list)

        if not file_foto_list:
            print(f"  [-] Tidak ada file foto valid (.jpg/.png) untuk {nim}")
            continue

        for foto_path in file_foto_list:
            # 1. Baca gambar dengan aman
            gambar = config.imread_unicode(foto_path)
            if gambar is None:
                total_foto_gagal += 1
                print(f"  [X] Gagal membaca citra: {foto_path.name}")
                continue

            # 2. Deteksi wajah individual
            wajah_list = app.get(gambar)

            # 3. Validasi jumlah wajah (wajib tepat 1 wajah)
            if len(wajah_list) == 0:
                total_foto_gagal += 1
                print(f"  [X] {foto_path.name}: Tidak ada wajah terdeteksi (Ditolak)")
                continue

            if len(wajah_list) > 1:
                total_foto_gagal += 1
                print(f"  [X] {foto_path.name}: Terdeteksi {len(wajah_list)} wajah (Ditolak - Multipel)")
                continue

            # 4. Foto valid -> Ambil normalized embedding 512 dimensi
            wajah = wajah_list[0]
            normed_embedding = wajah.normed_embedding  # 512-dimensi float32
            total_foto_valid += 1

            # 5. Baca biner citra asli untuk referensi_foto BYTEA
            with open(foto_path, "rb") as f_img:
                raw_bytes_foto = f_img.read()

            print(f"  [+] {foto_path.name}: 1 Wajah valid (det_score: {wajah.det_score:.3f})")

            # 6. Simpan setiap foto secara individual ke t_datawajah
            if simpan_ke_db:
                try:
                    sukses, id_dw, status_save = save_face_data(
                        nim=nim,
                        foto_bytes=raw_bytes_foto,
                        embedding_array=normed_embedding,
                        cek_duplikat_foto=not izinkan_duplikat_foto
                    )
                    if sukses:
                        total_data_baru_tersimpan += 1
                        daftar_tersimpan.append({
                            "id_datawajah": id_dw,
                            "nim": nim,
                            "nama": nama,
                            "file": foto_path.name
                        })
                        print(f"      -> [SIMPAN BARIS BARU] t_datawajah ID: {id_dw} ({foto_path.name})")
                    else:
                        total_data_duplikat_dilewati += 1
                        print(f"      -> [DUPLIKAT DILEWATI] Foto {foto_path.name} sudah ada (ID: {id_dw})")

                except Exception as e:
                    total_foto_gagal += 1
                    print(f"      -> [ERROR DATABASE]: {e}")
            else:
                total_data_baru_tersimpan += 1

    # ==================== LAPORAN REKAPITULASI RESMI ====================
    print("\n" + "=" * 70)
    print(" LAPORAN REKAPITULASI PENYIMPANAN DATASET WAJAH INDIVIDUAL")
    print("=" * 70)
    print(f" 1. Jumlah mahasiswa diproses       : {total_mahasiswa_diproses}")
    print(f" 2. Jumlah foto ditemukan           : {total_foto_ditemukan}")
    print(f" 3. Jumlah foto valid               : {total_foto_valid}")
    print(f" 4. Jumlah foto gagal               : {total_foto_gagal}")
    print(f" 5. Jumlah data baru tersimpan      : {total_data_baru_tersimpan}")
    print(f" 6. Jumlah data duplikat dilewati   : {total_data_duplikat_dilewati}")
    print(f" 7. Jumlah mahasiswa gagal          : {total_mahasiswa_gagal}")
    print("=" * 70)
    print(f" Rincian Baris Baru Tersimpan: {len(daftar_tersimpan)}")
    for item in daftar_tersimpan:
        print(f"  [+] ID: {item['id_datawajah']} | NIM: {item['nim']} | Nama: {item['nama']} | File: {item['file']}")
    print("=" * 70)

    return {
        "mahasiswa_diproses": total_mahasiswa_diproses,
        "foto_ditemukan": total_foto_ditemukan,
        "foto_valid": total_foto_valid,
        "foto_gagal": total_foto_gagal,
        "data_baru_tersimpan": total_data_baru_tersimpan,
        "duplikat_dilewati": total_data_duplikat_dilewati,
        "mahasiswa_gagal": total_mahasiswa_gagal
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registrasi Individual Foto Wajah ke PostgreSQL")
    parser.add_argument("--folder", type=str, default=None, help="Folder dataset foto mahasiswa")
    parser.add_argument("--dry-run", action="store_true", help="Uji ekstraksi tanpa menyimpan ke database")
    parser.add_argument("--force-insert", action="store_true", help="Paksa simpan tanpa memeriksa duplikat foto")
    args = parser.parse_args()

    registrasi_dataset_wajah(
        folder_dataset=args.folder,
        simpan_ke_db=not args.dry_run,
        izinkan_duplikat_foto=args.force_insert
    )
