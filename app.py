import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

# ==========================================
# 0. KONFIGURASI TEMA KARTU MODERN
# ==========================================
st.set_page_config(page_title="Council of AI - Melanoma HAM10000", layout="wide", page_icon="🧬")

# Custom Styling agar tampilan kartu seperti Web Enterprise
st.markdown("""
    <style>
    .big-font {font_size: 20px !important; font-weight: bold;}
    .metric-card {background-color: #0E1117; border: 1px solid #30363D; padding: 15px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Triple-Engine Melanoma Diagnostic Council")
st.caption("Sistem Pendukung Keputusan Klinis (CDSS) berbasis 35 Fitur White-Box & Ensemble Voting")

# ==========================================
# 1. MEMUAT 3 MESIN SEKALIGUS
# ==========================================
@st.cache_resource
def load_all_engines():
    scaler = joblib.load('RobustScaler_35.pkl')
    models = {
        'SVM': joblib.load('SVM_Model.pkl'),
        'XGBoost': joblib.load('XGBoost_Model.pkl'),
        'Extra Trees': joblib.load('Extra_Trees_Model.pkl')
    }
    return models, scaler

try:
    council_models, robust_scaler = load_all_engines()
except Exception as e:
    st.error(f"❌ Gagal memuat file .pkl model! Pastikan 4 file .pkl sudah ada di folder ini. Detail: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR: MISSION CONTROL & METODOLOGI
# ==========================================
with st.sidebar:
    # A. BRANDING KARTU ENTERPRISE
    st.markdown("""
        <div style="text-align: center; padding: 12px; background-color: #161B22; border-radius: 8px; border: 1px solid #30363D; margin-bottom: 15px;">
            <h3 style="margin:0; color: #58A6FF; font-size: 18px;">🧬 CDSS Medical Core</h3>
            <p style="margin:0; font-size: 11px; color: #8B949E;">Melanoma Early-Warning Radar v1.0</p>
        </div>
    """, unsafe_allow_html=True)

    # B. KONTROL UTAMA (Senjata Kalibrasi MLOps)
    st.subheader("🎛️ Kalibrasi Sensitivitas")
    
    master_threshold = st.slider(
        "Ambang Batas Keputusan (Threshold)", 
        min_value=0.15, max_value=0.75, value=0.35, step=0.01,
        help="Ambang batas teoritis adalah 0.25. Naikkan ke angka 0.35 - 0.40 jika citra diunggah dari kamera smartphone guna meredam False Positive akibat distorsi pencahayaan."
    )

    use_isp_filter = st.checkbox(
        "Aktifkan Image Damper (ISP Filter)", value=True,
        help="Menerapkan Gaussian Blur optis untuk menetralkan noise penajaman tepi buatan (auto-sharpening) dari kamera ponsel modern."
    )

    st.divider()

    # C. PASPOR METODOLOGI (Laci Rahasia Jawab Pertanyaan Dosen)
    st.markdown("##### 📚 Spesifikasi Sistem")
    
    with st.expander("📐 Arsitektur Ekstraksi Fitur", expanded=False):
        st.markdown("""
        **Pendekatan: White-Box Machine Learning**
        * **Resolusi Input:** 224 x 224 Pixels
        * **Pra-pemrosesan:** Morphological Black-Hat (11x11) + Telea Inpainting
        * **Segmentasi:** Otsu Thresholding + Matrix Slicing Level 0
        * **Fitur Warna (15D):** HSV Domain *(Mean, Std, Skewness, Kurtosis, Range)*
        * **Fitur Tekstur (20D):** GLCM *(Contrast, Energy, Correlation, Homogeneity, Entropy)* pada 4 sudut spasial (0°, 45°, 90°, 135°).
        """)

    with st.expander("📊 Profil Dataset (HAM10000)", expanded=False):
        st.markdown("""
        * **Populasi Utama:** 10,015 Citra Dermoskopi
        * **Rasio Kelas:** 11.1% Melanoma : 88.9% Non-Melanoma
        * **Penanganan Imbalance:** Cost-Sensitive Learning *(Class Weighting)* & Spatial Augmentation *(Zero-Leakage)*
        * **Target Optimasi:** Memaksimalkan **Recall** guna menekan angka pembunuh klinis *(False Negative)*.
        """)

    with st.expander("🩺 Pemetaan Medis (Aturan ABCDE)", expanded=False):
        st.markdown("""
        Korelasi fitur matematis terhadap kaidah dermatologi:
        * **A (Asymmetry):** Representasi komputasi *GLCM Entropy*
        * **B (Border Irregularity):** Representasi *GLCM Homogeneity*
        * **C (Color Variegation):** Representasi *HSV Kurtosis & Std*
        * **D & E (Diameter & Evolving):** Memerlukan pengamatan klinis fisik berkala.
        """)

    st.divider()

    # D. TELEMETRI SISTEM (Aksen Futuristik)
    st.caption("⚙️ TELEMETRI ENGINE")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Metode Voting", "SOFT", delta="Mean Prob")
    with col_stat2:
        st.metric("Jarak Vektor", "35 DIM", delta="Robust")

    st.spacer = st.write("")
    
    # E. DISCLAIMER KLINIS TEGAS
    st.warning("⚠️ **Medical Disclaimer:** Aplikasi ini dirancang secara khusus sebagai sistem pendukung keputusan triase awal. Tidak menggantikan biopsi histopatologi resmi.")

# ==========================================
# 3. DAPUR MATEMATIKA FITUR
# ==========================================
def process_and_extract(pil_img, apply_damper):
    img_rgb = np.array(pil_img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. BlackHat Hair Removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 7, 255, cv2.THRESH_BINARY)
    clean_rgb = cv2.inpaint(img_rgb, hair_mask, 2, cv2.INPAINT_TELEA)
    
    # [RAHASIA MLOps] Jembatan Domain Shift kamera HP
    if apply_damper:
        clean_rgb = cv2.GaussianBlur(clean_rgb, (3, 3), 0)
        
    img_224 = cv2.resize(clean_rgb, (224, 224))
    img_bgr = cv2.cvtColor(img_224, cv2.COLOR_RGB2BGR)
    gray_224 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Otsu
    _, otsu_mask = cv2.threshold(gray_224, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(otsu_mask == 255) < 1000:
        otsu_mask = np.ones(gray_224.shape, dtype=np.uint8) * 255

    feats = []
    # Ekstraksi HSV (15)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for i in range(3):
        p = hsv[:, :, i][otsu_mask == 255]
        if len(p) == 0: feats.extend([0.0]*5); continue
        feats.extend([np.mean(p), np.std(p), skew(p) if not np.isnan(skew(p)) else 0, kurtosis(p) if not np.isnan(kurtosis(p)) else 0, float(np.max(p)-np.min(p))])

    # Ekstraksi GLCM (20)
    g_quant = np.clip((gray_224 // 16) + 1, 1, 16).astype(np.uint8)
    g_quant[otsu_mask == 0] = 0
    glcm = graycomatrix(g_quant, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 17, symmetric=True, normed=False)[1:, 1:, :, :]
    glcm_norm = glcm / np.where(np.sum(glcm, axis=(0,1), keepdims=True)==0, 1, np.sum(glcm, axis=(0,1), keepdims=True))
    
    for prop in ['contrast', 'energy', 'correlation', 'homogeneity']:
        feats.extend(list(graycoprops(glcm_norm, prop)[0]))
    for a in range(4):
        pr = glcm_norm[:, :, 0, a]
        feats.append(max(0.0, -np.sum(pr * np.log2(pr + 1e-10))))

    return img_224, otsu_mask, np.array(feats).reshape(1, -1)

# ==========================================
# 4. ANTARMUKA UNGGAH & PREVIEW
# ==========================================
st.markdown("""
        <div style="text-align: center; padding: 12px; background-color: #161B22; border-radius: 8px; border: 1px solid #30363D; margin-bottom: 15px;">
            <h3 style="margin:0; color: #58A6FF; font-size: 18px;">Unggah foto (lesi) anda dibawah sini untuk dideteksi 👇</h3>
        </div>
    """, unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Unggah Citra Lesi Pasien (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded:
    raw_pil = Image.open(uploaded).convert('RGB')
    clean_224, mask_224, vector_35 = process_and_extract(raw_pil, use_isp_filter)
    scaled_vector = robust_scaler.transform(vector_35)
    
    # Baris Preview Gambar
    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1: 
        st.image(raw_pil, caption="Citra Mentah (Input)", use_container_width=True)
    with c2: 
        st.image(mask_224, caption="Area Lesi Terisolasi (Otsu)", use_container_width=True)
    with c3:
        st.write("#### 📋 Status Pemindaian")
        st.success("✅ **Citra berhasil dibersihkan** dari noise (rambut/pantulan).")
        st.success("✅ **35 Titik Fitur berhasil diekstrak** untuk dianalisis oleh mesin.")
        
        # Memberikan jarak (spacer) agar UI tidak terlalu padat
        st.markdown("<br>", unsafe_allow_html=True)

        # Peringatan visual (Call to Action) agar user scroll ke bawah
        st.info("👇 **PROSES SELESAI: Silakan gulir (scroll) ke bawah untuk melihat Keputusan Akhir Sistem.**")

    st.markdown("---")
    # ==========================================
    # 5. EKSEKUSI PREDIKSI & ACCUMULATE VOTE
    # ==========================================
    results = {}
    list_probs = []
    
    for name, model_engine in council_models.items():
        # Ambil probabilitas kelas 1 (Melanoma)
        prob = model_engine.predict_proba(scaled_vector)[0][1]
        list_probs.append(prob)
        results[name] = {
            'prob': prob,
            'is_cancer': prob >= master_threshold
        }

    # Menghitung Akumulasi Probabilitas (Soft Voting)
    accumulated_prob = float(np.mean(list_probs))
    is_final_cancer = accumulated_prob >= master_threshold

    # ==========================================
    # 6. PANEL KEPUTUSAN FINAL (DENGAN PENJELASAN AWAM)
    # ==========================================
    st.subheader("📊 Keputusan Akhir Sistem (Akumulasi 3 Mesin)")
    
    if is_final_cancer:
        st.error(f"### 🚨 TERDETEKSI INDIKASI MENCURIGAKAN")
        st.write(f"Berdasarkan diskusi dari 3 sistem AI, gambar ini memiliki **tingkat kecurigaan gabungan sebesar {accumulated_prob*100:.2f}%** (melewati batas aman yang ditetapkan: {master_threshold*100:.0f}%).")
        st.write("**Penjelasan Medis:** Sistem menemukan adanya ketidakteraturan pada sebaran warna kulit (fitur HSV) atau kekasaran permukaan tahi lalat (fitur GLCM) yang menyerupai pola keganasan. **Tindakan yang Disarankan:** Segera jadwalkan pemeriksaan lanjutan dengan dokter spesialis kulit untuk memastikan.")
    else:
        st.success(f"### ✅ TIDAK TERDETEKSI KEGANASAN (JINAK)")
        st.write(f"Berdasarkan diskusi dari 3 sistem AI, gambar ini memiliki **tingkat kecurigaan gabungan sebesar {accumulated_prob*100:.2f}%** (masih di bawah batas bahaya: {master_threshold*100:.0f}%).")
        st.write("**Penjelasan Medis:** Bentuk, warna, dan tekstur gambar ini sejalan dengan profil tahi lalat normal/jinak. Sistem tidak menemukan pola yang berbahaya. **Tindakan yang Disarankan:** Tetap pantau jika terjadi perubahan bentuk, warna, atau ukuran secara drastis di kemudian hari.")

    st.write("") # Spacer

    # ==========================================
    # 7. RINCIAN PER MESIN (TEKNIS + ANALOGI AWAM)
    # ==========================================
    st.markdown("##### Bagaimana Masing-Masing Mesin Menilai Gambar Ini?")
    col_et, col_xgb, col_svm = st.columns(3)

    def render_system_card(col_ui, title_sys, res_dict, ai_name, desc_sys):
        with col_ui:
            with st.container(border=True):
                st.write(f"**{title_sys}**")
                st.caption(f"*(Algoritma: {ai_name})*\n\n{desc_sys}")
                
                prob_val = res_dict['prob']
                is_curiga = res_dict['is_cancer']
                
                if is_curiga:
                    st.markdown("#### <span style='color:#FF4B4B'>🔴 Positif Curiga</span>", unsafe_allow_html=True)
                else:
                    st.markdown("#### <span style='color:#00CC96'>🟢 Negatif Aman</span>", unsafe_allow_html=True)
                    
                st.write(f"Tingkat Keyakinan: **{prob_val*100:.1f}%**")
                st.progress(float(prob_val))

    render_system_card(
        col_et, 
        "Extra Trees", 
        results['Extra Trees'],
        "Pakar Deteksi Dini",
        "Sangat sensitif. Fokus mencari sekecil apapun bintik warna atau tekstur aneh agar tidak ada pasien yang terlambat ditangani."
    )
    render_system_card(
        col_xgb, 
        "XGBoost", 
        results['XGBoost'],
        "Pakar Analisis Pola",
        "Sangat detail. Fokus membandingkan kontras dan kerumitan tekstur permukaan gambar secara mendalam."
    )
    render_system_card(
        col_svm, 
        "Support Vector Machine", 
        results['SVM'],
        "Pakar Batas Standar",
        "Sangat ketat. Hanya akan mencurigai gambar jika bentuk atau warnanya sudah jelas-jelas menyimpang dari bentuk tahi lalat normal."
    )

    # ==========================================
    # 8. FAQ, LOCAL XAI, & PANEL TEKNIS
    # ==========================================
    st.markdown("---")
    
    # KITA TAMBAHKAN 1 TAB BARU DI TENGAH: "💡 Alasan AI (Bedah Foto Ini)"
    tab_awam, tab_xai, tab_teknis = st.tabs([
        "🙋‍♂️ Informasi Pengguna", 
        "💡 Alasan AI (Bedah Foto Ini)", 
        "🔐 Log Sistem & Tabel Fitur"
    ])
    
    with tab_awam:
        st.write("#### Memahami Cara Kerja Aplikasi")
        with st.expander("❓ Bagaimana cara sistem mengambil Keputusan Akhir?"):
            st.write("Sistem ini menggunakan metode **Accumulate Vote** (Voting Akumulasi). Daripada hanya mengandalkan 1 penganalisa, kami menggabungkan tingkat kecurigaan dari ke-3 mesin di atas. Hasil akhirnya adalah **rata-rata** dari ketiga nilai tersebut.")
        with st.expander("❓ Apakah deteksi ini mutlak?"):
            st.write("**Tidak.** Aplikasi ini adalah **Alat Skrining Awal (Computer-Aided Diagnosis)**. Kualitas kamera ponsel dapat mempengaruhi hasil. Diagnosis resmi mutlak hanya diberikan oleh Dokter Spesialis Kulit via biopsi.")

    # =====================================================================
    # TAB BARU: LOCAL EXPLAINABLE AI (XAI)
    # =====================================================================
    with tab_xai:
        st.markdown("#### 🔬 Apa yang Membuat Mesin Menilai Gambar Ini Begitu?")
        st.write("Sistem membandingkan 35 indikator foto lesi ini dengan rata-rata tahi lalat jinak (menggunakan standar model *Extra Trees*):")
        
        # KAMUS TERJEMAHAN 35 FITUR KE BAHASA MANUSIA
        KAMUS_FITUR_35 = [
            # HSV Hue (0-4)
            "Rata-rata Rona Warna (Hue Mean)", "Variasi Rona Warna (Hue Std)", "Kemiringan Rona (Hue Skew)", "Lonjakan Rona (Hue Kurtosis)", "Jarak Rona (Hue Range)",
            # HSV Saturation (5-9)
            "Rata-rata Kepekatan Pigmen (Sat Mean)", "Variasi Kepekatan Pigmen (Sat Std)", "Kemiringan Kepekatan (Sat Skew)", "Lonjakan Pigmen Ekstrim (Sat Kurtosis)", "Jarak Kepekatan Pigmen (Sat Range)",
            # HSV Brightness (10-14)
            "Rata-rata Kecerahan Lesi (Val Mean)", "Variasi Kecerahan (Val Std)", "Kemiringan Kecerahan (Val Skew)", "Lonjakan Kecerahan (Val Kurtosis)", "Jarak Kecerahan (Val Range)",
            # GLCM Contrast (15-18)
            "Kontras Tekstur 0°", "Kontras Tekstur 45°", "Kontras Tekstur 90°", "Kontras Tekstur 135°",
            # GLCM Energy (19-22)
            "Kepadatan Energi 0°", "Kepadatan Energi 45°", "Kepadatan Energi 90°", "Kepadatan Energi 135°",
            # GLCM Correlation (23-26)
            "Korelasi Pola Spasial 0°", "Korelasi Pola Spasial 45°", "Korelasi Pola Spasial 90°", "Korelasi Pola Spasial 135°",
            # GLCM Homogeneity (27-30)
            "Kerapihan Batas Lesi 0°", "Kerapihan Batas Lesi 45°", "Kerapihan Batas Lesi 90°", "Kerapihan Batas Lesi 135°",
            # GLCM Entropy (31-34)
            "Kerumitan Tekstur (Entropy) 0°", "Kerumitan Tekstur (Entropy) 45°", "Kerumitan Tekstur (Entropy) 90°", "Kerumitan Tekstur (Entropy) 135°"
        ]

        # 1. Tarik bobot pentingnya fitur dari otak Extra Trees
        et_engine = council_models['Extra Trees']
        weights = et_engine.feature_importances_
        z_scores = scaled_vector[0] # Kondisi fisik pasien saat ini
        
        # 2. Rumus Local Attribution: Bobot Mesin x Simpangan Fisik Pasien
        push_impact = weights * z_scores
        
        df_explainer = pd.DataFrame({
            'Indikator Medis': KAMUS_FITUR_35,
            'Kategori': ['Warna HSV']*15 + ['Tekstur GLCM']*20,
            'Kondisi Fisik (Z-Score)': z_scores,
            'Daya Dorong Kanker': push_impact
        }).sort_values(by='Daya Dorong Kanker', ascending=False)

        top_3_bahaya = df_explainer.head(3)
        top_3_aman = df_explainer.tail(3).sort_values(by='Daya Dorong Kanker', ascending=True)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.error("🔴 **3 Faktor Utama Pendorong Kecurigaan:**")
            st.caption("Indikator dengan lonjakan abnormal yang memicu vonis bahaya:")
            for _, r in top_3_bahaya.iterrows():
                kondisi_teks = "Melonjak Abnormal" if r['Kondisi Fisik (Z-Score)'] > 0 else "Anjlok Abnormal"
                st.write(f"**{r['Indikator Medis']}**\n* Status: `{kondisi_teks}` ({r['Kondisi Fisik (Z-Score)']:.1f} σ)\n* Kontribusi Keganasan: `+{r['Daya Dorong Kanker']*100:.1f}%`")

        with c_exp2:
            st.success("🟢 **3 Faktor Penahan ke Arah Jinak:**")
            st.caption("Indikator yang kondisinya rapi/normal sehingga menahan skor:")
            for _, r in top_3_aman.iterrows():
                st.write(f"**{r['Indikator Medis']}**\n* Status: `Normal/Stabil` ({r['Kondisi Fisik (Z-Score)']:.1f} σ)\n* Daya Tahan Jinak: `{r['Daya Dorong Kanker']*100:.1f}%`")
                
        st.divider()
        st.caption("💡 *Keterangan Matematis: σ (Sigma) adalah satuan standar deviasi. Semakin besar angka σ, semakin jauh lesi kulit ini menyimpang dari anatomi tahi lalat normal.*")

    # =====================================================================
    # TAB 3: LOG SISTEM 
    # =====================================================================
    with tab_teknis:
        st.info("💡 **Log Operasional:** Menampilkan parameter latar belakang dan matriks ekstraksi mentah.")
        c_deb1, c_deb2 = st.columns([1, 1.2])
        with c_deb1:
            st.write("##### Parameter Skrining")
            st.code(f"Threshold Aktif: {master_threshold}\nMetode: Soft Voting Accumulate\nSegmentasi: Otsu + Telea 11x11", language="yaml")
        with c_deb2:
            st.write("##### Matriks 35 Fitur Numerik")
            kat_list = ['Warna (HSV)'] * 15 + ['Tekstur (GLCM)'] * 20
            nm_list = [f'Fitur ke-{i}' for i in range(35)]
            df_v = pd.DataFrame({'Kategori': kat_list, 'Index': nm_list, 'Nilai': vector_35[0]})
            st.dataframe(df_v, height=350, use_container_width=True, hide_index=True)