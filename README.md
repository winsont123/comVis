# Klasifikasi Lesi Kulit Melanoma menggunakan Fitur HSV & GLCM

Repositori ini berisi **source code** untuk Proyek Akhir mata kuliah **Computer Vision**. Aplikasi dikembangkan menggunakan **Streamlit** dan mengimplementasikan metode **Ensemble Learning** (XGBoost, Extra Trees, dan Support Vector Machine) untuk mengklasifikasikan lesi kulit menjadi **Melanoma** atau **Non-Melanoma**.

Model memanfaatkan **35 fitur** yang diekstraksi dari karakteristik **warna (HSV)** dan **tekstur (GLCM)**. Selain itu, aplikasi juga melakukan proses pra-pemrosesan secara otomatis, meliputi:

- Hair Removal menggunakan metode **Black-Hat Morphology**
- Segmentasi lesi menggunakan **Otsu Thresholding**
- Ekstraksi fitur HSV dan GLCM
- Prediksi menggunakan model Ensemble

---

# 📁 Struktur Direktori

```text
comVis/
├── Training_Melanoma.ipynb          # Notebook pelatihan dan evaluasi model
├── app.py                           # Aplikasi Streamlit
├── requirements.txt                 # Daftar dependensi
├── XGBoost_Model.pkl                # Model XGBoost
├── Extra_Trees_Model.pkl            # Model Extra Trees
├── SVM_Model.pkl                    # Model Support Vector Machine
├── RobustScaler_35.pkl              # RobustScaler untuk 35 fitur
└── extracted_features_35_Tuned.csv  # Dataset hasil ekstraksi 35 fitur
```

---

# 🚀 Menjalankan Aplikasi Secara Lokal

## 1. Clone Repository

```bash
git clone https://github.com/winsont123/comVis.git
cd comVis
```

## 2. Membuat Virtual Environment (Opsional tetapi Direkomendasikan)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Jalankan Streamlit

```bash
streamlit run app.py
```

Aplikasi akan berjalan pada:

```
http://localhost:8501
```

---

# 📒 Menjalankan Notebook Training

Notebook `Training_Melanoma.ipynb` dikembangkan menggunakan **Google Colab** yang terhubung dengan **Google Drive**.

Sebelum menjalankan notebook:

1. Upload seluruh file pendukung ke Google Drive.
2. Pastikan file berikut tersedia:
   - `extracted_features_35_Tuned.csv`
   - Folder model
   - Folder hasil evaluasi (jika digunakan)
3. Sesuaikan variabel path seperti `csv_path` dan `path_simpan` dengan lokasi penyimpanan pada Google Drive Anda, misalnya:

```python
/content/drive/MyDrive/comVis/
```

---

# 🔄 Reproduksibilitas Hasil

Untuk memperoleh hasil evaluasi yang **identik** dengan laporan proyek, ikuti panduan berikut.

## Mengapa hasil dapat berubah?

Walaupun `random_state=42` telah digunakan, proses ekstraksi fitur langsung dari gambar dapat menghasilkan urutan pembacaan file yang berbeda pada sistem operasi. Perbedaan ini dapat memengaruhi:

- Pembagian data train/validation/test
- Proses training model
- Nilai akurasi
- Confusion Matrix
- Feature Importance

## Cara memperoleh hasil yang sama

Gunakan dataset tabular yang telah disediakan:

```
extracted_features_35_Tuned.csv
```

File tersebut merupakan hasil ekstraksi **35 fitur HSV dan GLCM** yang telah dibekukan sehingga menghasilkan evaluasi yang konsisten.

### Jalankan notebook mulai dari bagian:

```
Training and Evaluation
```

### Jangan menjalankan ulang bagian:

- Pre-processing
- Hair Removal
- Segmentasi
- Feature Extraction
- Data Splitting

Dengan mengikuti langkah tersebut, hasil evaluasi akan sesuai dengan yang tercantum pada laporan proyek.

---

# ⚡ Quick Test

Jika tujuan Anda hanya ingin mencoba aplikasi klasifikasi lesi kulit, **tidak perlu menjalankan notebook**.

Cukup jalankan:

```bash
streamlit run app.py
```

Kemudian unggah gambar lesi kulit melalui antarmuka Streamlit untuk memperoleh hasil klasifikasi secara langsung.

---

# 🛠️ Teknologi yang Digunakan

- Python
- Streamlit
- OpenCV
- scikit-image
- scikit-learn
- XGBoost
- Extra Trees
- Support Vector Machine (SVM)
- NumPy
- Pandas
- Matplotlib

---

# 📊 Fitur yang Digunakan

Model menggunakan total **35 fitur**, terdiri dari:

- **HSV Color Features**
  - Mean
  - Standard Deviation
  - Skewness
  - Kurtosis
  - Entropy

- **GLCM Texture Features**
  - Contrast
  - Correlation
  - Energy
  - Homogeneity
  - ASM
  - Dissimilarity
  - Entropy

---

# 👥 Tim Pengembang

Proyek ini dikembangkan sebagai bagian dari Proyek Akhir mata kuliah **Computer Vision**.
