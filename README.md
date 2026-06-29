# Klasifikasi Lesi Kulit Melanoma menggunakan Fitur HSV & GLCM

Repositori ini berisi *source code* untuk Proyek Akhir mata kuliah **Computer Vision**. Proyek ini merupakan aplikasi berbasis antarmuka web (Streamlit) yang menggunakan metode *Ensemble* (XGBoost, Extra Trees, SVM) untuk mengklasifikasikan lesi kulit (Melanoma vs Non-Melanoma) berdasarkan ekstraksi 35 fitur warna (HSV) dan tekstur (GLCM). Aplikasi mencakup pra-pemrosesan mandiri seperti *Hair Removal* (Black-Hat) dan segmentasi otomatis (Otsu).

## Struktur Direktori
```text
comVis/
├── Training_Melanoma.ipynb      # Notebook berisi seluruh alur dari pra-pemrosesan hingga evaluasi model akhir
├── app.py                       # Skrip utama antarmuka Streamlit (Termasuk logika ekstraksi 35 fitur)
├── requirements.txt             # Daftar dependensi library
├── XGBoost_Model.pkl            # Model pra-latih XGBoost
├── Extra_Trees_Model.pkl        # Model pra-latih Extra Trees
├── SVM_Model.pkl                # Model pra-latih Support Vector Machine
└── RobustScaler_35.pkl          # Scaler pra-latih untuk normalisasi data

# Catatan Penting Terkait Reproduksibilitas (Reproducibility Notes)
Untuk memastikan hasil evaluasi (Akurasi, Confusion Matrix, ROC/AUC) yang didapat saat dosen penguji menjalankan kode di notebook sama persis dengan metrik yang tertera pada Laporan Proyek, mohon perhatikan panduan berikut:
1. Hindari Menjalankan Ulang Blok Pre-Processing dan Data Splitting: Pembacaan gambar awal memiliki sifat acak yang bergantung pada Sistem Operasi. Perbedaan urutan baca file ini akan menggeser komposisi train/val/test split, sehingga hasil akhir model akan berbeda dari laporan.
2. Gunakan Dataset Tabular (CSV) yang Telah Disediakan: Untuk melatih ulang model dari awal namun dengan hasil yang 100% identik dengan laporan, silakan langsung eksekusi Training_Melanoma.ipynb dimulai dari blok "Training and Evaluasi". Sistem akan otomatis menggunakan file ekstraksi fitur asli yang sudah disiapkan.
3. Pengujian Aplikasi (Quick Test): Jika tujuan utamanya adalah untuk melihat output aplikasi dan melakukan klasifikasi gambar lesi kulit baru secara real-time, silakan abaikan notebook dan langsung jalankan antarmuka Streamlit (petunjuk di bawah).
