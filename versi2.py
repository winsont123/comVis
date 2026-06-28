import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# 1. KONFIGURASI HALAMAN & TAMPILAN
# ==========================================
st.set_page_config(
    page_title="Melanoma Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan kotak hasil
st.markdown("""
<style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-top: 10px;
    }
    .melanoma { background-color: #ff4b4b; }
    .non-melanoma { background-color: #00cc96; }
    .feature-text { font-size: 14px; color: #555; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL (Fungsi dengan Cache agar Cepat)
# ==========================================
@st.cache_resource
def load_models():
    # Pastikan file .pkl berada di folder yang sama dengan app.py
    svm = joblib.load('SVM_Best_Model_35.pkl')
    et = joblib.load('ExtraTrees_Best_Model_35.pkl')
    xgb = joblib.load('XGBoost_Best_Model_35.pkl')
    return svm, et, xgb

try:
    svm_model, et_model, xgb_model = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Gagal memuat model. Pastikan file .pkl ada di folder yang sama. Error: {e}")
    models_loaded = False


# ==========================================
# 3. FUNGSI EKSTRAKSI FITUR (Sama Persis dengan Skrip Training)
# ==========================================
def extract_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        channel = hsv[:, :, i].flatten()
        mean = np.mean(channel)
        std = np.std(channel)
        sk = skew(channel)
        if np.isnan(sk): sk = 0
        kur = kurtosis(channel)
        if np.isnan(kur): kur = 0
        ran = np.max(channel) - np.min(channel)
        features.extend([mean, std, sk, kur, ran])
    return features

def extract_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (gray / 16).astype(np.uint8)
    glcm = graycomatrix(
        gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, 'contrast')[0]
    energy = graycoprops(glcm, 'energy')[0]
    correlation = graycoprops(glcm, 'correlation')[0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0]
    
    entropy = []
    for a in range(4):
        glcm_prob = glcm[:, :, 0, a]
        ent = -np.sum(glcm_prob * np.log2(glcm_prob + 1e-10))
        entropy.append(ent)
        
    features = list(contrast) + list(energy) + list(correlation) + list(homogeneity) + entropy
    return features

def process_image_for_prediction(image_pil):
    # Konversi dari PIL ke format OpenCV (BGR)
    img_array = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize wajib ke 64x64 sesuai proses training
    img_resized = cv2.resize(img_bgr, (64, 64))
    
    # Ekstraksi 35 Fitur
    hsv_feat = extract_hsv(img_resized)
    glcm_feat = extract_glcm(img_resized)
    
    # Gabungkan menjadi format 2D array untuk model scikit-learn
    features = hsv_feat + glcm_feat
    return np.array(features).reshape(1, -1)


# ==========================================
# 4. TAMPILAN UTAMA (FRONTEND)
# ==========================================
st.title("🔬 Sistem Cerdas Diagnosis Melanoma")
st.write("""
Aplikasi web ini menggunakan **Machine Learning Ensemble (XGBoost & Extra Trees)** dan ekstraksi 35 metrik fitur visual (GLCM Textures & HSV Colors) untuk membantu membedakan lesi 
kanker kulit Melanoma dari lesi jinak.
""")

st.divider()

# Area Upload
uploaded_file = st.file_uploader("Unggah Citra Dermoskopi (Format: JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and models_loaded:
    # Membaca gambar yang diunggah
    image = Image.open(uploaded_file)
    
    # Membagi layout menjadi 2 kolom (Kiri untuk Gambar, Kanan untuk Hasil)
    col_img, col_res = st.columns([1, 2])
    
    with col_img:
        st.subheader("Citra Lesi")
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)
        
    with col_res:
        st.subheader("Sedang Memproses...")
        with st.spinner('Mengekstrak 35 fitur GLCM & HSV...'):
            try:
                # Proses Ekstraksi
                feature_vector = process_image_for_prediction(image)
                
                # --- PREDIKSI SVM ---
                prob_svm = svm_model.predict_proba(feature_vector)[0]
                # --- PREDIKSI EXTRA TREES ---
                prob_et = et_model.predict_proba(feature_vector)[0]
                # --- PREDIKSI XGBOOST ---
                prob_xgb = xgb_model.predict_proba(feature_vector)[0]
                
                # Asumsi Label Encoding: 0 = Melanoma, 1 = Non-Melanoma
                # (Sesuai urutan abjad M dan N saat kamu training)
                melanoma_idx = 0 
                
                # Nilai Probabilitas (Keyakinan bahwa itu Melanoma)
                conf_svm = prob_svm[melanoma_idx]
                conf_et = prob_et[melanoma_idx]
                conf_xgb = prob_xgb[melanoma_idx]
                
                # Logika Keputusan (Termasuk Custom Threshold 0.40 untuk XGBoost)
                res_svm = "Melanoma" if conf_svm >= 0.50 else "Non-Melanoma"
                res_et = "Melanoma" if conf_et >= 0.50 else "Non-Melanoma"
                res_xgb = "Melanoma" if conf_xgb >= 0.40 else "Non-Melanoma" # Trik 0.40
                
                st.success("✅ Ekstraksi fitur dan inferensi model selesai!")
                
                st.subheader("📊 Hasil Diagnosis Komparatif")
                
                # Menampilkan 3 hasil berjajar
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**XGBoost (Unggulan)**")
                    css_class = "melanoma" if res_xgb == "Melanoma" else "non-melanoma"
                    st.markdown(f"<div class='prediction-box {css_class}'><h4>{res_xgb}</h4><p>{conf_xgb*100:.1f}% Yakin</p></div>", unsafe_allow_html=True)
                    st.caption("Custom Threshold: 40%")
                
                with c2:
                    st.markdown("**Extra Trees**")
                    css_class = "melanoma" if res_et == "Melanoma" else "non-melanoma"
                    st.markdown(f"<div class='prediction-box {css_class}'><h4>{res_et}</h4><p>{conf_et*100:.1f}% Yakin</p></div>", unsafe_allow_html=True)
                    st.caption("Standard Threshold: 50%")
                    
                with c3:
                    st.markdown("**SVM (Baseline)**")
                    css_class = "melanoma" if res_svm == "Melanoma" else "non-melanoma"
                    st.markdown(f"<div class='prediction-box {css_class}'><h4>{res_svm}</h4><p>{conf_svm*100:.1f}% Yakin</p></div>", unsafe_allow_html=True)
                    st.caption("Standard Threshold: 50%")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

st.divider()
st.caption("Catatan: Sistem ini merupakan purwarupa (prototype) Computer-Aided Diagnosis dan bukan pengganti saran medis profesional dari dokter kulit (Dermatologist).")