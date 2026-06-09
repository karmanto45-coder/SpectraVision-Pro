# SpectraVision Pro v3.0

> Evolusi dari SpectraID Pro — kini dengan **Post-MCR Processing** sebelum identifikasi spektra.

## 🆕 Fitur Baru (v3.0)

### Tab ✨ Proses Spektra MCR (NEW)
Pipeline 3 langkah terpisah yang dapat dikombinasikan bebas:

| Langkah | Opsi |
|---------|------|
| **1. Baseline** | Min subtraction |
| **2. Smoothing** | Savitzky-Golay (window & poly order adjustable) |
| **3. Normalisasi** | Area (trapezoid), Max intensity, Vector L2, Min-Max |

- **Perbandingan overlay**: spektra original vs diproses per komponen
- Spektra yang diproses **otomatis digunakan** di tab Identifikasi
- Download spektra diproses sebagai CSV
- Tombol reset ke spektra MCR original kapan saja

### Tab 🔍 Identifikasi (Updated)
- **Banner status** jelas: menunjukkan apakah menggunakan spektra diproses atau original
- Cosine similarity & HQI dihitung dari spektra pasca-pemrosesan

---

## Struktur File

```
SpectraVisionPro/
├── app.py           ← main application (UPDATED)
├── mcr_engine.py    ← + fungsi postprocess_mcr_spectra() (UPDATED)
├── auth.py          ← tidak berubah
├── cos2d.py         ← tidak berubah
├── database.py      ← tidak berubah (copy dari project lama)
├── requirements.txt
└── README.md
```

---

## Deployment ke Streamlit Cloud (Nama Baru)

### 1. Buat repository GitHub baru
```
Nama repo: spectravision-pro   (atau nama lain pilihan Anda)
```

### 2. Upload semua file ini ke repo baru

### 3. Deploy di streamlit.io
- Buka https://share.streamlit.io
- Klik **"New app"**
- Pilih repo `spectravision-pro`
- Main file: `app.py`
- App URL: `spectravisionpro.streamlit.app` (atau custom)

### 4. Login default
```
Username: admin
Password: admin123
```

---

## Alur Kerja yang Direkomendasikan

```
📂 Input Data
    ↓
🔬 Analisis MCR-ALS
    ↓
✨ Proses Spektra MCR   ← BARU
  • Baseline (opsional)
  • Smoothing SG (opsional)
  • Normalisasi area/max/vector/minmax (opsional)
    ↓
🔍 Identifikasi (cosine + HQI vs library)
    ↓
📊 Laporan (Excel multi-sheet, termasuk sheet spektra diproses)
```

---

## Perbedaan SpectraID Pro vs SpectraVision Pro

| Fitur | SpectraID Pro v2.0 | SpectraVision Pro v3.0 |
|-------|-------------------|------------------------|
| Post-MCR smoothing | ✗ | ✅ SG (window & poly) |
| Post-MCR normalisasi | ✗ | ✅ 4 metode |
| Post-MCR baseline | ✗ | ✅ Min subtraction |
| Overlay original vs diproses | ✗ | ✅ Per komponen |
| Identifikasi dari spektra diproses | ✗ | ✅ Otomatis |
| Status banner di tab identifikasi | ✗ | ✅ |
| Export spektra diproses | ✗ | ✅ CSV + Excel |
| Laporan: sheet raw vs processed | ✗ | ✅ Terpisah |
