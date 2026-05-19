import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Klasifikasi Citra: Asli vs AI",
    layout="wide"
)

# --- DESIGN WEBSITE ---
st.markdown("""
    <style>
        /* DESIGN: Background utama website */
        .stApp {
            background: linear-gradient(135deg, #fff7fb 0%, #ffffff 45%, #ffe4ef 100%);
            color: #111111;
        }
        
        /* DESIGN: Menghilangkan bocor warna hitam di pojok kiri atas */
        header[data-testid="stHeader"] {
            background-color: #ffffff !important;
            border-radius: 0 !important;
        }

        header[data-testid="stHeader"]::before,
        header[data-testid="stHeader"]::after {
            background-color: #ffffff !important;
            border-radius: 0 !important;
        }

        /* DESIGN: Memberi ruang atas agar judul tidak tertutup tombol Deploy */
        .block-container {
            padding-top: 4.5rem !important;
            padding-bottom: 1rem !important;
        }

        /* DESIGN: Judul utama */
        .main-title {
            text-align: center;
            color: #111111;
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            line-height: 1.25;
        }

        /* DESIGN: Nama pada tampilan website */
        .name-title {
            text-align: center;
            color: #d63384;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 1.5rem;
        }

        /* DESIGN: Kotak deskripsi */
        .subtitle-box {
            background-color: #ffffff;
            border: 2px solid #ffc2d9;
            border-radius: 18px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(214, 51, 132, 0.12);
            margin-bottom: 1.5rem;
        }

        .subtitle-box p {
            color: #333333;
            margin: 0;
            font-size: 1rem;
        }

        h1, h2, h3, p, label, span {
            color: #111111;
        }

        /* DESIGN: Kotak upload file */
        [data-testid="stFileUploader"] {
            background-color: #ffffff !important;
            border: 2px dashed #f5a6c8 !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            color: #111111 !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #fff7fb !important;
            border: 1px solid #ffc2d9 !important;
            border-radius: 14px !important;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p {
            color: #111111 !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #d63384 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploader"] button:hover {
            background-color: #b82b70 !important;
            color: #ffffff !important;
        }

        /* DESIGN: Tampilan gambar yang diuji */
        [data-testid="stHorizontalBlock"] img {
            max-height: 38vh !important;
            width: auto !important;
            margin: auto;
            display: block;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
        }

        /* DESIGN: Judul hasil analisis */
        .analysis-title {
            color: #111111;
            font-size: 1.8rem;
            font-weight: 800;
            font-style: italic;
            margin-bottom: 1rem;
        }

        /* DESIGN: Mengatur tampilan pesan success dan error */
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            font-size: 1rem !important;
        }

        /* DESIGN: Mengatur area tombol Deploy kanan atas agar tetap terlihat */
        [data-testid="stToolbar"] {
            background-color: #ffffff !important;
            border-radius: 0 0 0 14px;
            padding: 0.3rem !important;
            box-shadow: 0 3px 12px rgba(0, 0, 0, 0.10);
        }

        [data-testid="stToolbar"] button {
            color: #111111 !important;
        }

        [data-testid="stToolbar"] svg {
            color: #111111 !important;
            fill: #111111 !important;
        }
            
        /* DESIGN: Warna background saat tombol Deploy dan titik tiga diarahkan kursor */
        [data-testid="stToolbar"] button:hover {
            background-color: rgba(255, 56, 152, 0.45) !important;
            border-radius: 10px !important;
        }

        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)


# --- 2. LOAD MODEL MENGGUNAKAN TFSMLayer (KHUSUS KERAS 3) ---
@st.cache_resource
def load_classification_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = BASE_DIR
    
    try:
        return tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model. Detail Error: {e}")
        return None

model = load_classification_model()


# --- 3. FUNGSI PREPROCESSING & PREDIKSI ---
def predict_image(image_input, model_keras):
    img_array = np.asarray(image_input)
    
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    tensor_img = tf.convert_to_tensor(img_array, dtype=tf.float32)
    tensor_img = tf.image.resize_with_pad(tensor_img, 300, 300)
    img_array_padded = tf.expand_dims(tensor_img, axis=0)
    
    prediction_dict = model_keras(img_array_padded)
    
    hasil_probabilitas = 0.0
    for key, value in prediction_dict.items():
        hasil_probabilitas = float(np.array(value)[0][0])
        break
        
    return hasil_probabilitas


# --- 4. TAMPILAN UTAMA ---
st.markdown('<div class="main-title">Sistem Deteksi Citra Asli vs Buatan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="name-title">By Diva Dwicitra Dewi</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="subtitle-box">
        <p>Unggah gambar untuk mengetahui apakah citra terdeteksi sebagai citra asli atau buatan AI.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih file gambar untuk dianalisis...", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    
    with st.spinner("Mengekstrak fitur citra..."):
        raw_output = predict_image(image, model)
        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### *Citra yang Diuji*")
        st.image(image, use_container_width=True)
        
    with col2:
        st.markdown('<div class="analysis-title">Hasil Analisis Klasifikasi</div>', unsafe_allow_html=True)
        
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            txt_path = os.path.join(BASE_DIR, 'final_threshold.txt')
            with open(txt_path, "r") as f:
                threshold = float(f.read().strip())
        except Exception:
            threshold = 0.50
            
        if raw_output >= threshold:
            st.success("### *HASIL: TERDETEKSI SEBAGAI CITRA ASLI (REAL)*")
            confidence_score = raw_output * 100
            probabilitas_asli = raw_output
            probabilitas_ai = 1.0 - raw_output
        else:
            st.error("### *HASIL: TERDETEKSI SEBAGAI CITRA BUATAN AI*")
            confidence_score = (1.0 - raw_output) * 100
            probabilitas_ai = 1.0 - raw_output
            probabilitas_asli = raw_output
            
        st.markdown(f"**Tingkat Keyakinan (*Confidence Score*): {confidence_score:.2f}%**")
        st.caption(f"Data Debug Mentah (Sigmoid): {raw_output:.6f} | Threshold: {threshold:.4f}")

else:
    st.info("Silakan unggah gambar untuk memulai analisis.")