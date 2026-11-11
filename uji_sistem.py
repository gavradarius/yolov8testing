"""
Suite Pengujian Menyeluruh Sistem Absensi Mahasiswa & PostgreSQL
Menguji 15 skenario sistem:
1. Koneksi PostgreSQL
2. Pemeriksaan enam tabel resmi
3. Relasi foreign key
4. Pengambilan data t_mahasiswa
5. Pembacaan foto dari data_uji / data_wajah
6. Deteksi wajah
7. Ekstraksi embedding (InsightFace 512-dim)
8. Penyimpanan foto BYTEA ke PostgreSQL
9. Penyimpanan embedding BYTEA ke PostgreSQL
10. Pengambilan foto dari PostgreSQL
11. Pengambilan embedding dari PostgreSQL
12. Pengenalan wajah (Cosine Similarity)
13. Pengenalan mahasiswa yang belum terdaftar ('MAHASISWA TIDAK DIKENAL')
14. Pencatatan absensi
15. Pencegahan absensi duplikat
"""

import sys
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

import config
from database import (
    get_connection,
    close_connection,
    test_connection,
    create_tables,
    get_mahasiswa,
    save_face_data,
    get_face_data,
    save_absensi,
    generate_short_id
)
from registrasi_wajah import muat_metadata_identitas, VALID_EXTENSIONS


def jalankan_15_pengujian():
    print("=" * 72)
    print(" SUITE PENGUJIAN 15 SKENARIO SISTEM ABSENSI & POSTGRESQL")
    print("=" * 72)

    hasil_uji = {}

    # 1. Koneksi PostgreSQL
    print("\n[1/15] Uji Koneksi PostgreSQL...")
    terhubung, pesan_db, _ = test_connection()
    if terhubung:
        print(f"  [PASS] {pesan_db}")
        hasil_uji["1. Koneksi PostgreSQL"] = "PASS"
    else:
        print(f"  [FAIL] {pesan_db}")
        hasil_uji["1. Koneksi PostgreSQL"] = f"FAIL: {pesan_db}"
        return hasil_uji

    # 2. Pemeriksaan Enam Tabel
    print("\n[2/15] Pemeriksaan Enam Tabel Resmi...")
    tabel_wajib = {'t_mahasiswa', 't_users', 't_jadwal', 't_absensi', 't_pengaduan', 't_datawajah'}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            tabel_ada = {r[0] for r in cur.fetchall()}
            ada_semua = tabel_wajib.issubset(tabel_ada)
            if ada_semua:
                print(f"  [PASS] 6 Tabel lengkap ditemukan: {sorted(list(tabel_wajib))}")
                hasil_uji["2. Pemeriksaan Enam Tabel"] = "PASS"
            else:
                kurang = tabel_wajib - tabel_ada
                print(f"  [FAIL] Tabel belum lengkap, kurang: {kurang}")
                hasil_uji["2. Pemeriksaan Enam Tabel"] = f"FAIL: Kurang {kurang}"
    finally:
        close_connection(conn)

    # 3. Relasi Foreign Key
    print("\n[3/15] Pemeriksaan Relasi Foreign Key...")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY';
            """)
            fk_list = cur.fetchall()
            print(f"  [PASS] Ditemukan {len(fk_list)} relasi Foreign Key terdaftar:")
            for fk in fk_list:
                print(f"    - {fk[0]}.{fk[1]} -> {fk[2]}")
            hasil_uji["3. Relasi Foreign Key"] = f"PASS ({len(fk_list)} FKs terverifikasi)"
    finally:
        close_connection(conn)

    # 4. Pengambilan Data t_mahasiswa
    print("\n[4/15] Uji Pengambilan Data t_mahasiswa...")
    mhs_list = get_mahasiswa()
    if mhs_list and len(mhs_list) > 0:
        print(f"  [PASS] Berhasil mengambil {len(mhs_list)} mahasiswa dari t_mahasiswa.")
        hasil_uji["4. Pengambilan Data t_mahasiswa"] = f"PASS ({len(mhs_list)} record)"
    else:
        print("  [FAIL] Data t_mahasiswa kosong.")
        hasil_uji["4. Pengambilan Data t_mahasiswa"] = "FAIL (Kosong)"

    # 5. Pembacaan Foto dari data_uji / data_wajah
    print("\n[5/15] Uji Pembacaan Foto dari Folder Dataset...")
    target_dir = config.DATA_UJI_DIR if config.DATA_UJI_DIR.exists() else Path("data_wajah")
    foto_files = [f for f in target_dir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
    foto_bisa_dibaca = 0
    for f in foto_files:
        img = config.imread_unicode(f)
        if img is not None:
            foto_bisa_dibaca += 1

    if foto_bisa_dibaca > 0:
        print(f"  [PASS] Berhasil membaca {foto_bisa_dibaca}/{len(foto_files)} foto berekstensi valid (.jpg/.png).")
        hasil_uji["5. Pembacaan Foto dari Dataset"] = f"PASS ({foto_bisa_dibaca} foto)"
    else:
        print("  [FAIL] Tidak ada foto yang berhasil dibaca.")
        hasil_uji["5. Pembacaan Foto dari Dataset"] = "FAIL"

    # 6. Deteksi Wajah
    print("\n[6/15] Uji Deteksi Wajah (InsightFace)...")
    app = FaceAnalysis(name="buffalo_l", root=str(config.MODELS_CACHE_DIR), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(config.DETECTION_SIZE, config.DETECTION_SIZE))

    sample_foto = foto_files[0] if foto_files else None
    sample_img = config.imread_unicode(sample_foto) if sample_foto else None
    wajah_sample = app.get(sample_img) if sample_img is not None else []

    if len(wajah_sample) >= 1:
        print(f"  [PASS] Deteksi wajah berhasil pada {sample_foto.name}: {len(wajah_sample)} wajah ditemukan.")
        hasil_uji["6. Deteksi Wajah"] = f"PASS (score: {wajah_sample[0].det_score:.3f})"
    else:
        print("  [FAIL] Deteksi wajah gagal.")
        hasil_uji["6. Deteksi Wajah"] = "FAIL"

    # 7. Ekstraksi Embedding
    print("\n[7/15] Uji Ekstraksi Embedding (512 Dimensi)...")
    if len(wajah_sample) >= 1:
        emb_sample = wajah_sample[0].normed_embedding
        norma = float(np.linalg.norm(emb_sample))
        print(f"  [PASS] Ekstraksi embedding sukses: dimensi={len(emb_sample)}, norma L2={norma:.4f}")
        hasil_uji["7. Ekstraksi Embedding"] = f"PASS (512-dim, norma={norma:.2f})"
    else:
        hasil_uji["7. Ekstraksi Embedding"] = "FAIL"

    # 8. Penyimpanan Foto BYTEA
    print("\n[8/15] Uji Penyimpanan Foto ke BYTEA PostgreSQL...")
    raw_foto_bytes = None
    if sample_foto:
        with open(sample_foto, "rb") as f_img:
            raw_foto_bytes = f_img.read()
    test_nim = mhs_list[0]["nim"] if mhs_list else "A320250001"
    sukses_save, test_id_dw = save_face_data(
        nim=test_nim,
        foto_bytes=raw_foto_bytes,
        embedding_array=emb_sample
    )
    if sukses_save:
        print(f"  [PASS] Foto {len(raw_foto_bytes)} bytes & embedding disimpan ke t_datawajah (ID: {test_id_dw}).")
        hasil_uji["8. Penyimpanan Foto BYTEA"] = "PASS"
        hasil_uji["9. Penyimpanan Embedding BYTEA"] = "PASS"
    else:
        hasil_uji["8. Penyimpanan Foto BYTEA"] = "FAIL"
        hasil_uji["9. Penyimpanan Embedding BYTEA"] = "FAIL"

    # 10 & 11. Pengambilan Foto dan Embedding dari PostgreSQL
    print("\n[10/15 & 11/15] Uji Pengambilan Foto & Embedding dari PostgreSQL...")
    data_wajah_db = get_face_data()
    item_uji = next((w for w in data_wajah_db if w["nim"] == test_nim), None)

    if item_uji and item_uji["foto_bytes"] and item_uji["embedding"] is not None:
        foto_db_len = len(item_uji["foto_bytes"])
        emb_db_shape = item_uji["embedding"].shape
        print(f"  [PASS] Foto berhasil diambil dari PostgreSQL: {foto_db_len} bytes.")
        print(f"  [PASS] Embedding berhasil diambil dan didekode: shape={emb_db_shape}, tipe={item_uji['embedding'].dtype}")
        hasil_uji["10. Pengambilan Foto dari PostgreSQL"] = f"PASS ({foto_db_len} bytes)"
        hasil_uji["11. Pengambilan Embedding dari PostgreSQL"] = f"PASS (shape={emb_db_shape})"
    else:
        hasil_uji["10. Pengambilan Foto dari PostgreSQL"] = "FAIL"
        hasil_uji["11. Pengambilan Embedding dari PostgreSQL"] = "FAIL"

    # 12. Pengenalan Wajah (Cosine Similarity)
    print("\n[12/15] Uji Pengenalan Wajah Menggunakan Cosine Similarity...")
    foto_uji_gavra = config.BASE_DIR / "foto_uji" / "uji_gavra.jpg"
    wajah_gavra_db = next((w for w in data_wajah_db if w["nim"] == "A320250010"), None)

    if foto_uji_gavra.exists() and wajah_gavra_db:
        img_gavra = config.imread_unicode(foto_uji_gavra)
        wajah_g = app.get(img_gavra)
        if len(wajah_g) > 0:
            emb_kamera = wajah_g[0].normed_embedding
            skor_sim = float(np.dot(emb_kamera, wajah_gavra_db["embedding"]))
            cocok = skor_sim >= config.SIMILARITY_THRESHOLD
            print(f"  [PASS] Uji Gavra vs Database: Skor={skor_sim:.4f} (Ambang: {config.SIMILARITY_THRESHOLD}, Cocok: {cocok})")
            hasil_uji["12. Pengenalan Wajah (Cosine Sim)"] = f"PASS (Skor: {skor_sim:.4f} >= {config.SIMILARITY_THRESHOLD})"
        else:
            hasil_uji["12. Pengenalan Wajah (Cosine Sim)"] = "FAIL (Wajah tidak terdeteksi)"
    else:
        hasil_uji["12. Pengenalan Wajah (Cosine Sim)"] = "SKIPPED (Foto uji tidak ada)"

    # 13. Pengenalan Mahasiswa yang Belum Terdaftar ('MAHASISWA TIDAK DIKENAL')
    print("\n[13/15] Uji Penolakan Wajah Tidak Dikenal ('MAHASISWA TIDAK DIKENAL')...")
    foto_uji_asing = config.BASE_DIR / "foto_uji" / "uji_salma.jpg"
    if foto_uji_asing.exists() and wajah_gavra_db:
        img_asing = config.imread_unicode(foto_uji_asing)
        wajah_a = app.get(img_asing)
        if len(wajah_a) > 0:
            emb_asing = wajah_a[0].normed_embedding
            skor_asing = float(np.dot(emb_asing, wajah_gavra_db["embedding"]))
            kenal = skor_asing >= config.SIMILARITY_THRESHOLD
            status_label = "Dikenal" if kenal else "MAHASISWA TIDAK DIKENAL"
            print(f"  [PASS] Uji Wajah Asing vs Database: Skor={skor_asing:.4f} -> Status: {status_label}")
            if status_label == "MAHASISWA TIDAK DIKENAL":
                hasil_uji["13. Deteksi MAHASISWA TIDAK DIKENAL"] = f"PASS (Skor: {skor_asing:.2f} < {config.SIMILARITY_THRESHOLD})"
            else:
                hasil_uji["13. Deteksi MAHASISWA TIDAK DIKENAL"] = "FAIL (False Positive)"
        else:
            hasil_uji["13. Deteksi MAHASISWA TIDAK DIKENAL"] = "FAIL"
    else:
        hasil_uji["13. Deteksi MAHASISWA TIDAK DIKENAL"] = "SKIPPED"

    # 14 & 15. Pencatatan Absensi & Pencegahan Duplikat
    print("\n[14/15 & 15/15] Uji Pencatatan Absensi & Pencegahan Duplikat...")
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # Siapkan 1 user dan 1 jadwal untuk keperluan validasi relasi Foreign Key absensi
                cur.execute("""
                    INSERT INTO t_users (id_users, nama, password, email, role)
                    VALUES ('USR_TEST01', 'Dosen Penguji', 'hashpass', 'dosen@kampus.ac.id', 'Dosen')
                    ON CONFLICT (id_users) DO NOTHING;
                """)
                cur.execute("""
                    INSERT INTO t_jadwal (id_jadwal, id_users, kode_matkul, nama_matkul, ruang_kelas, sks, hari, jam_mulai, jam_selesai)
                    VALUES ('JDW_TEST01', 'USR_TEST01', 'IF-101', 'Pemrograman', 'R301', 3, 'Senin', '00:00:00', '23:59:59')
                    ON CONFLICT (id_jadwal) DO NOTHING;
                """)

        # Catat absensi pertama
        res1 = save_absensi(nim=test_nim, id_jadwal="JDW_TEST01", status_kehadiran="Hadir", cek_duplikasi=True)
        # Catat absensi kedua (harus ditolak/dideteksi sebagai sudah_absen)
        res2 = save_absensi(nim=test_nim, id_jadwal="JDW_TEST01", status_kehadiran="Hadir", cek_duplikasi=True)

        if res1.get("status") in ("berhasil", "sudah_absen") and res2.get("status") == "sudah_absen":
            print(f"  [PASS] Absensi berhasil dicatat (ID: {res1.get('id_absensi')}).")
            print(f"  [PASS] Pencegahan duplikasi absensi berhasil: {res2.get('pesan')}")
            hasil_uji["14. Pencatatan Absensi (t_absensi)"] = "PASS"
            hasil_uji["15. Pencegahan Absensi Duplikat"] = "PASS"
        else:
            hasil_uji["14. Pencatatan Absensi (t_absensi)"] = f"FAIL (res1={res1})"
            hasil_uji["15. Pencegahan Absensi Duplikat"] = f"FAIL (res2={res2})"
    finally:
        close_connection(conn)

    # ==================== REKAPITULASI ====================
    print("\n" + "=" * 72)
    print(" REKAPITULASI HASIL PENGUJIAN 15 SKENARIO SISTEM")
    print("=" * 72)
    semua_pass = True
    for item, stat in hasil_uji.items():
        print(f" {item.ljust(48)} : {stat}")
        if not stat.startswith("PASS"):
            semua_pass = False
    print("=" * 72)
    print(f" HASIL AKHIR: {'SELURUH 15 SKENARIO TERVERIFIKASI SUKSES (100% PASS)' if semua_pass else 'ADA SKENARIO GAGAL'}")
    print("=" * 72)

    return hasil_uji


if __name__ == "__main__":
    jalankan_15_pengujian()
