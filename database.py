"""
Pusat Koneksi dan Operasi Database PostgreSQL Resmi
Sistem Absensi Mahasiswa Berbasis YOLOv11 & InsightFace

Fungsi Standar yang Disediakan:
- get_connection()
- test_connection()
- create_tables()
- get_mahasiswa()
- save_face_data()
- get_face_data()
- save_absensi()
- close_connection()
"""

import uuid
import datetime
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

import config


def get_connection():
    """
    Membuka dan mengembalikan koneksi baru ke database PostgreSQL
    sesuai kredensial di config.py / .env.
    """
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        connect_timeout=5
    )
    try:
        conn.set_client_encoding("UTF8")
    except Exception:
        pass
    return conn


def close_connection(conn):
    """Menutup koneksi database secara aman."""
    if conn and not conn.closed:
        try:
            conn.close()
        except Exception:
            pass


def test_connection():
    """
    Menguji konektivitas ke server PostgreSQL dan memverifikasi
    current_database(), current_user, dan inet_server_port().
    Returns: (bool, str, dict)
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    current_database(),
                    current_user,
                    inet_server_port(),
                    version();
            """)
            db_name, user_name, port, versi = cur.fetchone()
        close_connection(conn)
        info = {
            "database": db_name,
            "user": user_name,
            "port": port,
            "version": versi
        }
        return True, f"Terhubung ke PostgreSQL: DB={db_name}, User={user_name}, Port={port}", info
    except Exception as e:
        return False, str(e), {}


# Alias kompatibilitas
uji_koneksi = test_connection


def generate_short_id(prefix="AB"):
    """
    Membuat ID unik acak dengan panjang maksimal 11 karakter
    untuk memenuhi tipe VARCHAR(11) pada Primary Key tabel resmi.
    Format: 2 huruf kapital prefix + 9 karakter heksadesimal acak = 11 karakter.
    Contoh: 'DW8A1F3C90D' atau 'AB123456789'
    """
    clean_prefix = prefix[:2].upper()
    random_hex = uuid.uuid4().hex[:9].upper()
    return f"{clean_prefix}{random_hex}"


def create_tables(drop_existing=False):
    """
    Membuat 6 tabel resmi PostgreSQL jika belum ada,
    sesuai urutan Foreign Key untuk menjaga integritas relasi.
    ATURAN: Tidak melakukan DROP TABLE tanpa izin tertulis demi menjaga data lama.
    """
    if drop_existing:
        raise PermissionError("Operasi DROP TABLE dilarang untuk menjaga keamanan data yang ada!")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cursor:
                # 1. Tabel Master Mahasiswa
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS t_mahasiswa (
                        nim VARCHAR(11) PRIMARY KEY,
                        nama VARCHAR(255),
                        tingkat VARCHAR(255),
                        program_studi VARCHAR(255)
                    );
                """)

                # 2. Tabel Master Users (Dosen / Admin)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS t_users (
                        id_users VARCHAR(11) PRIMARY KEY,
                        nama VARCHAR(255),
                        password VARCHAR(255),
                        email VARCHAR(255),
                        role VARCHAR(11)
                    );
                """)

                # 3. Tabel Jadwal Kuliah (FK ke t_users)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS t_jadwal (
                        id_jadwal VARCHAR(11) PRIMARY KEY,
                        id_users VARCHAR(11),
                        kode_matkul VARCHAR(255),
                        nama_matkul VARCHAR(22),
                        ruang_kelas VARCHAR(255),
                        sks INT,
                        hari VARCHAR(11),
                        jam_mulai TIME,
                        jam_selesai TIME,
                        FOREIGN KEY (id_users) REFERENCES t_users(id_users)
                    );
                """)

                # 4. Tabel Absensi (FK ke t_mahasiswa & t_jadwal)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS t_absensi (
                        id_absensi VARCHAR(11) PRIMARY KEY,
                        nim VARCHAR(11),
                        id_jadwal VARCHAR(11),
                        tanggal DATE,
                        waktu_absen TIME,
                        status_kehadiran VARCHAR(255),
                        FOREIGN KEY (nim) REFERENCES t_mahasiswa(nim),
                        FOREIGN KEY (id_jadwal) REFERENCES t_jadwal(id_jadwal)
                    );
                """)

                # 5. Tabel Pengaduan (FK ke t_mahasiswa, t_jadwal, & t_absensi)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS t_pengaduan (
                        id_pengaduan VARCHAR(11) PRIMARY KEY,
                        nim VARCHAR(11),
                        id_jadwal VARCHAR(11),
                        id_absensi VARCHAR(11),
                        jenis_kendala VARCHAR(255),
                        status_pengaduan VARCHAR(255),
                        FOREIGN KEY (nim) REFERENCES t_mahasiswa(nim),
                        FOREIGN KEY (id_jadwal) REFERENCES t_jadwal(id_jadwal),
                        FOREIGN KEY (id_absensi) REFERENCES t_absensi(id_absensi)
                    );
                """)

                # 6. Tabel Data Wajah (FK ke t_mahasiswa)
                # Catatan: Satu mahasiswa boleh memiliki banyak baris di t_datawajah.
                # Tidak ada UNIQUE constraint pada kolom nim.
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS t_datawajah (
                        id_datawajah VARCHAR(11) PRIMARY KEY,
                        nim VARCHAR(11),
                        referensi_foto BYTEA,
                        face_embedding BYTEA,
                        FOREIGN KEY (nim) REFERENCES t_mahasiswa(nim)
                    );
                """)

                # Indeks pencarian
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datawajah_nim ON t_datawajah(nim);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_absensi_nim_tgl ON t_absensi(nim, tanggal);")

        return True, "Enam tabel skema resmi PostgreSQL telah diverifikasi dan siap digunakan."
    finally:
        close_connection(conn)


# Alias kompatibilitas
buat_skema_database = create_tables


def get_mahasiswa(nim=None):
    """
    Mengambil data mahasiswa dari tabel t_mahasiswa.
    - Jika nim diberikan: mengembalikan dict mahasiswa atau None jika tidak ditemukan.
    - Jika nim None: mengembalikan list dict seluruh mahasiswa.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if nim:
                cur.execute("""
                    SELECT nim, nama, tingkat, program_studi
                    FROM t_mahasiswa
                    WHERE nim = %s;
                """, (nim,))
                return cur.fetchone()
            else:
                cur.execute("""
                    SELECT nim, nama, tingkat, program_studi
                    FROM t_mahasiswa
                    ORDER BY nim ASC;
                """)
                return cur.fetchall()
    finally:
        close_connection(conn)


def save_face_data(nim, foto_bytes, embedding_array, id_datawajah=None, cek_duplikat_foto=True):
    """
    Menyimpan satu foto dan satu embedding individual ke t_datawajah (SATU FOTO = SATU EMBEDDING = SATU BARIS).
    ATURAN UTAMA:
    1. TIDAK ADA DELETE berdasarkan NIM. Data wajah lain milik mahasiswa yang sama tetap utuh.
    2. Satu mahasiswa boleh memiliki banyak baris di t_datawajah.
    3. Jika cek_duplikat_foto=True, periksa apakah foto identik (berdasarkan hash MD5) sudah pernah disimpan untuk NIM tersebut.
       Jika sudah ada, foto tersebut dilewati tanpa error dan tanpa menghapus data yang ada.
    4. Menyimpan foto asli ke referensi_foto BYTEA dan embedding 512 float32 ke face_embedding BYTEA.
    """
    if not id_datawajah:
        id_datawajah = generate_short_id(prefix="DW")

    # Konversi array numpy float32 ke biner mentah (2048 bytes untuk 512 dimensi float32)
    embedding_bytes = embedding_array.astype(np.float32).tobytes()

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # 1. Pastikan mahasiswa terdaftar di t_mahasiswa
                cur.execute("SELECT 1 FROM t_mahasiswa WHERE nim = %s;", (nim,))
                if not cur.fetchone():
                    raise ValueError(f"NIM {nim} belum terdaftar di t_mahasiswa!")

                # 2. Cek apakah foto yang sama persis sudah ada di t_datawajah untuk NIM ini
                if cek_duplikat_foto and foto_bytes:
                    cur.execute("""
                        SELECT id_datawajah 
                        FROM t_datawajah 
                        WHERE nim = %s AND md5(referensi_foto) = md5(%s)
                        LIMIT 1;
                    """, (nim, foto_bytes))
                    existing = cur.fetchone()
                    if existing:
                        return False, existing[0], "DUPLIKAT"

                # 3. INSERT data wajah baru (TIDAK MENGHAPUS DATA LAIN)
                cur.execute("""
                    INSERT INTO t_datawajah (id_datawajah, nim, referensi_foto, face_embedding)
                    VALUES (%s, %s, %s, %s);
                """, (
                    id_datawajah,
                    nim,
                    psycopg2.Binary(foto_bytes) if foto_bytes else None,
                    psycopg2.Binary(embedding_bytes)
                ))

        return True, id_datawajah, "BARU"
    finally:
        close_connection(conn)


def get_face_data():
    """
    Mengambil seluruh data wajah dari t_datawajah di-JOIN ke t_mahasiswa.
    Setiap baris embedding BYTEA dikonversi menjadi NumPy float32 array berdimensi 512.
    Returns: list of dict -> [{id_datawajah, nim, nama, program_studi, embedding, foto_bytes}]
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    d.id_datawajah,
                    d.nim,
                    m.nama,
                    m.program_studi,
                    d.face_embedding,
                    d.referensi_foto
                FROM t_datawajah d
                JOIN t_mahasiswa m ON d.nim = m.nim
                WHERE d.face_embedding IS NOT NULL
                ORDER BY d.nim ASC, d.id_datawajah ASC;
            """
            cur.execute(query)
            rows = cur.fetchall()

            daftar_referensi = []
            for row in rows:
                id_dw, nim, nama, prodi, emb_bytes, foto_b = row
                # Konversi biner BYTEA ke array numpy float32
                embedding_array = np.frombuffer(emb_bytes, dtype=np.float32)

                daftar_referensi.append({
                    "id_datawajah": id_dw,
                    "nim": nim,
                    "nama": nama,
                    "prodi": prodi,
                    "embedding": embedding_array,
                    "foto_bytes": foto_b
                })

            return daftar_referensi
    finally:
        close_connection(conn)


# Alias kompatibilitas
muat_referensi_wajah = get_face_data


def dapatkan_jadwal_aktif():
    """
    Mendapatkan id_jadwal yang sedang aktif berdasarkan hari dan jam saat ini.
    Jika tidak ada jadwal yang sesuai saat ini, mengembalikan None (tidak membuat jadwal fiktif).
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            hari_map = {
                0: "Senin", 1: "Selasa", 2: "Rabu",
                3: "Kamis", 4: "Jumat", 5: "Sabtu", 6: "Minggu"
            }
            sekarang = datetime.datetime.now()
            nama_hari = hari_map.get(sekarang.weekday(), "Senin")
            waktu_str = sekarang.strftime("%H:%M:%S")

            cur.execute("""
                SELECT id_jadwal, kode_matkul, nama_matkul, ruang_kelas, jam_mulai, jam_selesai
                FROM t_jadwal
                WHERE LOWER(hari) = LOWER(%s)
                  AND %s BETWEEN jam_mulai AND jam_selesai
                LIMIT 1;
            """, (nama_hari, waktu_str))
            return cur.fetchone()
    finally:
        close_connection(conn)


def save_absensi(nim, id_jadwal, status_kehadiran="Hadir", tanggal=None, waktu_absen=None, cek_duplikasi=True):
    """
    Menyimpan catatan absensi ke tabel t_absensi.
    - id_jadwal wajib valid di t_jadwal (tidak boleh fiktif atau None).
    - Menghasilkan id_absensi unik (maksimal 11 karakter).
    - Memiliki validasi duplikasi absensi pada hari & jadwal yang sama.
    """
    if not id_jadwal:
        raise ValueError("id_jadwal tidak boleh kosong atau fiktif!")

    sekarang = datetime.datetime.now()
    tgl = tanggal or sekarang.date()
    wkt = waktu_absen or sekarang.time().replace(microsecond=0)

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                # 1. Pastikan mahasiswa terdaftar
                cur.execute("SELECT 1 FROM t_mahasiswa WHERE nim = %s;", (nim,))
                if not cur.fetchone():
                    raise ValueError(f"Mahasiswa dengan NIM {nim} belum terdaftar di t_mahasiswa!")

                # 2. Pastikan id_jadwal valid di t_jadwal
                cur.execute("SELECT 1 FROM t_jadwal WHERE id_jadwal = %s;", (id_jadwal,))
                if not cur.fetchone():
                    raise ValueError(f"Jadwal {id_jadwal} tidak terdaftar di tabel t_jadwal!")

                # 3. Pencegahan duplikasi absensi pada hari & jadwal yang sama
                if cek_duplikasi:
                    cur.execute("""
                        SELECT id_absensi FROM t_absensi 
                        WHERE nim = %s AND id_jadwal = %s AND tanggal = %s
                        LIMIT 1;
                    """, (nim, id_jadwal, tgl))
                    existing = cur.fetchone()
                    if existing:
                        return {
                            "status": "sudah_absen",
                            "id_absensi": existing[0],
                            "pesan": f"NIM {nim} sudah tercatat hadir pada jadwal ini hari ini."
                        }

                id_absensi = generate_short_id(prefix="AB")
                cur.execute("""
                    INSERT INTO t_absensi (
                        id_absensi, nim, id_jadwal, tanggal, waktu_absen, status_kehadiran
                    ) VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    id_absensi,
                    nim,
                    id_jadwal,
                    tgl,
                    wkt,
                    status_kehadiran
                ))

        return {
            "status": "berhasil",
            "id_absensi": id_absensi,
            "nim": nim,
            "id_jadwal": id_jadwal,
            "tanggal": str(tgl),
            "waktu_absen": str(wkt),
            "status_kehadiran": status_kehadiran,
            "pesan": f"Absensi berhasil dicatat untuk NIM {nim}."
        }
    finally:
        close_connection(conn)


# Alias kompatibilitas
catat_absensi = save_absensi
