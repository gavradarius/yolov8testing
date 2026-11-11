-- ========================================================
-- SKEMA DATABASE POSTGRESQL SISTEM ABSENSI PENGENALAN WAJAH
-- Urutan eksekusi memperhatikan integritas Foreign Key (FK)
-- ========================================================

-- 1. TABEL MASTER MAHASISWA
CREATE TABLE IF NOT EXISTS t_mahasiswa (
    NIM VARCHAR(11) PRIMARY KEY,
    nama VARCHAR(255),
    tingkat VARCHAR(255),
    Program_studi VARCHAR(255)
);

-- 2. TABEL MASTER USERS (DOSEN / ADMIN)
CREATE TABLE IF NOT EXISTS t_users (
    id_users VARCHAR(11) PRIMARY KEY,
    nama VARCHAR(255),
    password VARCHAR(255),
    email VARCHAR(255),
    role VARCHAR(11)
);

-- 3. TABEL JADWAL KULIAH (BERELASI KE t_users)
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

-- 4. TABEL TRANSAKSI ABSENSI (BERELASI KE t_mahasiswa & t_jadwal)
CREATE TABLE IF NOT EXISTS t_absensi (
    id_absensi VARCHAR(11) PRIMARY KEY,
    NIM VARCHAR(11),
    id_jadwal VARCHAR(11),
    tanggal DATE,
    waktu_absen TIME,
    status_kehadiran VARCHAR(255),
    FOREIGN KEY (NIM) REFERENCES t_mahasiswa(NIM),
    FOREIGN KEY (id_jadwal) REFERENCES t_jadwal(id_jadwal)
);

-- 5. TABEL PENGADUAN (BERELASI KE t_mahasiswa, t_jadwal, & t_absensi)
CREATE TABLE IF NOT EXISTS t_pengaduan (
    id_pengaduan VARCHAR(11) PRIMARY KEY,
    NIM VARCHAR(11),
    id_jadwal VARCHAR(11),
    id_absensi VARCHAR(11),
    jenis_kendala VARCHAR(255),
    Status_pengaduan VARCHAR(255),
    FOREIGN KEY (NIM) REFERENCES t_mahasiswa(NIM),
    FOREIGN KEY (id_jadwal) REFERENCES t_jadwal(id_jadwal),
    FOREIGN KEY (id_absensi) REFERENCES t_absensi(id_absensi)
);

-- 6. TABEL DATA WAJAH (BERELASI KE t_mahasiswa)
CREATE TABLE IF NOT EXISTS t_datawajah (
    id_datawajah VARCHAR(11) PRIMARY KEY,
    NIM VARCHAR(11),
    Referensi_foto BYTEA,
    face_embedding BYTEA,
    FOREIGN KEY (NIM) REFERENCES t_mahasiswa(NIM)
);
