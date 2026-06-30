# Klasifikasi Lesi Kulit Melanoma menggunakan Fitur HSV & GLCM

Repositori ini berisi *source code* untuk Proyek Akhir mata kuliah **Computer Vision**. Proyek ini merupakan aplikasi berbasis antarmuka web (Streamlit) yang menggunakan metode *Ensemble* (XGBoost, Extra Trees, SVM) untuk mengklasifikasikan lesi kulit (Melanoma vs Non-Melanoma) berdasarkan ekstraksi 35 fitur warna (HSV) dan tekstur (GLCM). Aplikasi mencakup pra-pemrosesan mandiri seperti *Hair Removal* (Black-Hat) dan segmentasi otomatis (Otsu).

## 📁 Struktur Direktori
```text
comVis/
├── Training_Melanoma.ipynb         # Notebook berisi seluruh alur dari pra-pemrosesan hingga evaluasi akhir
├── app.py                          # Skrip utama antarmuka Streamlit (Termasuk logika ekstraksi 35 fitur)
├── requirements.txt                # Daftar dependensi library
├── XGBoost_Model.pkl               # Model pra-latih XGBoost
├── Extra_Trees_Model.pkl           # Model pra-latih Extra Trees
├── SVM_Model.pkl                   # Model pra-latih Support Vector Machine
├── RobustScaler_35.pkl             # Scaler pra-latih untuk normalisasi data
└── extracted_features_35_Tuned.csv # Dataset tabular berisi 35 fitur (HSV & GLCM)

Lingkungan Eksekusi (Google Colab & Drive)
Eksperimen pemodelan Machine Learning dalam proyek ini (Training_Melanoma.ipynb) dirancang untuk dieksekusi secara optimal menggunakan Google Colab yang terhubung langsung dengan Google Drive. Jika Anda ingin menguji kode pada notebook tersebut:
1. Pastikan seluruh file pendukung (Dataset CSV, Folder Model, dan Folder Hasil Evaluasi) telah diunggah ke Google Drive Anda.
2. Ubah path direktori pada variabel csv_path dan path_simpan di dalam blok kode agar sesuai dengan lokasi penyimpanan Google Drive Anda (misal: /content/drive/MyDrive/...).

Isu Reproduksibilitas
1. Untuk memastikan hasil evaluasi model (Akurasi, Confusion Matrix, Feature Importance, dll.) yang didapat saat menjalankan kode di notebook sama persis 100% dengan metrik yang tertera pada Laporan Proyek, mohon perhatikan panduan berikut:
2. Kendala Randomness OS: Proses ekstraksi fitur langsung dari gambar mentah rentan terhadap keacakan (randomness) tingkat sistem operasi saat membaca urutan file. Jika blok Pre-Processing dan Data Splitting dijalankan ulang, komposisi distribusi data train/val/test akan bergeser dan mengubah hasil akhir training, meskipun parameter random_state=42 telah dikonfigurasi.
3. Gunakan Dataset Tabular (CSV): Untuk menghindari isu tersebut, kami telah menyediakan dataset tabular berisi 35 fitur (HSV & GLCM) yang sudah dibekukan secara permanen dalam bentuk CSV (extracted_features_35_Tuned.csv).
4. Instruksi Eksekusi: Mohon JANGAN menjalankan ulang blok Pre-Processing, Data Splitting, dan Ekstraksi 35 Fitur. Silakan langsung mulai pengeksekusian notebook dari blok "Training and Evaluasi" ke bawah. Sistem akan otomatis memuat file CSV tersebut untuk menjamin reproduksibilitas hasil yang identik.
5. Pengujian Aplikasi (Quick Test): Jika tujuan utamanya adalah untuk melihat output aplikasi dan melakukan klasifikasi gambar lesi kulit baru secara real-time, silakan abaikan notebook dan langsung jalankan antarmuka Streamlit.

## Panduan Menjalankan Secara Lokal (Quick Start)

Jika Anda ingin menjalankan aplikasi pemindaian lesi kulit ini secara lokal di komputer Anda, silakan ikuti langkah-langkah berikut:

1. **Kloning Repositori**
   ```bash
   git clone [https://github.com/winsont123/comVis.git](https://github.com/winsont123/comVis.git)
   cd comVis

2. **Buat Virtual Environment (Sangat Disarankan)**
    ```bash
    python -m venv venv
    
    # Untuk pengguna Windows:
    venv\Scripts\activate
    # Untuk pengguna macOS/Linux:
    source venv/bin/activate

3. **Install Pustaka Pendukung**
    ```bash
    pip install -r requirements.txt

4. **Jalankan Antarmuka Streamlit**
    ```bash
    streamlit run app.py

Aplikasi akan otomatis terbuka di peramban web (browser) Anda pada alamat http://localhost:8501.
