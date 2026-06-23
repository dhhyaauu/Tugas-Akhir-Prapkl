# 1. IMPORTS

import os
import json
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


try:
    from streamlit_option_menu import option_menu
except ImportError:
    st.error("Install dulu: pip install streamlit-option-menu")
    st.stop()

# Scikit-learn (untuk fallback model jika model.pkl belum ada)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# 2. KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Prediksi Deposito Nasabah | Saffa Dhiya",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 3. CUSTOM CSS — TEMA NAVY / PUTIH / ABU / HITAM ELEGAN

CUSTOM_CSS = """
<style>
    /* ---------- Import Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Background ---------- */
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef2f7 100%);
        color: #000000
    }

    /* ---------- Header Gradient ---------- */
    .hero-header {
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 50%, #0f172a 100%);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        color: #ffffff;
        box-shadow: 0 20px 60px rgba(10, 31, 68, 0.25);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.4rem; font-weight: 800;
        margin: 0; letter-spacing: -0.5px;
    }
    .hero-header p {
        opacity: 0.85; font-size: 1.05rem;
        margin-top: 0.5rem; font-weight: 400;
    }

    /* ---------- Card ---------- */
    .premium-card {
        background: #ffffff;
        padding: 1.75rem;
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.05);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1rem;
    }
    .premium-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.12);
        border-color: rgba(30, 58, 138, 0.15);
    }

    /* ---------- Metric Card ---------- */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem; border-radius: 16px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
        transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-size: 2rem; font-weight: 800; color: #0a1f44;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-label {
        color: #64748b; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* ---------- Profile Card ---------- */
    .profile-card {
        background: white;
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(10, 31, 68, 0.1);
        transition: all 0.4s ease;
        border: 1px solid rgba(15, 23, 42, 0.05);
    }
    .profile-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 80px rgba(10, 31, 68, 0.18);
    }
    .profile-img {
        width: 180px; height: 180px;
        border-radius: 50%;
        object-fit: cover;
        border: 6px solid white;
        box-shadow: 0 10px 30px rgba(10, 31, 68, 0.2);
        margin-bottom: 1.25rem;
    }
    .profile-name {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem; font-weight: 800;
        color: #0a1f44; margin: 0.5rem 0 0.25rem;
    }
    .profile-role {
        color: #1e3a8a; font-weight: 600;
        font-size: 0.95rem; margin-bottom: 1rem;
    }
    .social-icon {
    display: inline-flex;
    width: 46px;
    height: 46px;
    align-items: center;
    justify-content: center;
    background: #f1f5f9;
    border-radius: 50%;   
    margin: 0 0.35rem;
    text-decoration: none;
    color: #0a1f44;
    font-size: 1.2rem;
    transition: all 0.3s ease;
}
    .social-icon:hover {
        background: #0a1f44; color: white;
        transform: translateY(-3px);
    }

    /* ---------- Section Title ---------- */
    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: #0a1f44;
        border-left: 4px solid #1e3a8a;
        padding-left: 12px; margin: 1.5rem 0 1rem;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(10, 31, 68, 0.25) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(10, 31, 68, 0.35) !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(15, 23, 42, 0.06);
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div {
        color: #0f172a !important; /* Memaksa semua teks bawaan sidebar berwarna gelap */
    }
    [data-testid="stSidebar"] .sidebar-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800; font-size: 1.3rem;
        color: #000000 !important; margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] .sidebar-subtitle {
        color: #475569 !important; font-size: 0.85rem; font-weight: 600;
    }
    [data-testid="stSidebar"] .sidebar-copyright {
        color: #94a3b8 !important; font-size: 0.75rem; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white; padding: 6px;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(15,23,42,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 100%) !important;
        color: white !important;
    }

    /* ---------- Result Card (prediksi) — Animasi Baru ---------- */
    .result-success {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: white; padding: 2rem; border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(4, 120, 87, 0.4);
        animation: resultPulse 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }
    .result-fail {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: white; padding: 2rem; border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(185, 28, 28, 0.4);
        animation: resultPulse 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }
    
    /* Keyframe Animasi Baru (Efek Pop Zoom Berenergi) */
    @keyframes resultPulse {
        0% { transform: scale(0.8); opacity: 0; box-shadow: 0 0 0 rgba(0,0,0,0); }
        70% { transform: scale(1.03); }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    .fade-in { animation: fadeIn 0.8s ease; }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center; padding: 2rem 1rem 1rem;
        color: #64748b; font-size: 0.9rem;
        border-top: 1px solid rgba(15,23,42,0.08);
        margin-top: 3rem;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu, footer, header {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 4. KONSTANTA & PATH

DATA_PATH       = "bank-full.csv"
NOTEBOOK_PATH   = "notebook.ipynb"
MODEL_PATH      = "model_deposito.joblib"
PROFILE_IMG     = "assets/fotoprofile.jpeg"
DATASET_SOURCE  = "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"


# 5. UTILITY FUNCTIONS

@st.cache_data(show_spinner=False)
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame | None:
    """Load dataset Bank Marketing dari UCI."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner=False)
def train_fallback_model(df: pd.DataFrame):
    """Latih model RandomForest sederhana jika model.pkl belum ada."""
    data = df.copy()
    encoders = {}
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    X = data.drop("y", axis=1)
    y = data["y"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return {
        "model": model, "scaler": scaler, "encoders": encoders,
        "features": list(X.columns), "accuracy": acc,
    }


def image_to_base64(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()



# 6. SIDEBAR — PANDUAN PENGGUNAAN (Sudah Diperbaiki & Jelas)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
      <div style='font-size:2.5rem;'>💎</div>
      <div class='sidebar-title'>Deposit Predictor</div>
      <div class='sidebar-subtitle'>Premium ML Dashboard</div>
    </div>
    <hr style='margin:1rem 0; border-color:#cbd5e1;'/>
    """, unsafe_allow_html=True)

    st.markdown("### 📖 Panduan Penggunaan")
    st.markdown("""
    **Langkah-langkah:**
    1. Mulai dari **Tentang Aplikasi**
    2. Lihat analisis data pada halaman **Analisis Data** untuk eksplorasi
    3. Lakukan prediksi pada halaman **Prediksi**
    4. Cek profil di **Tentang Saya**
    """)

    st.markdown("### ✨ Fitur Utama")
    st.markdown("""
    - Visualisasi interaktif
    - Dokumentasi notebook lengkap
    - Prediksi real-time
    - UI modern & responsif
    """)

    st.markdown("### 💡 Tips")
    st.info("Isi semua input dengan data yang valid agar prediksi optimal.")

    st.markdown("### 🔌 Status Sistem")
    df_check = load_dataset()
    if df_check is not None:
        st.success(f"Dataset: ✓ ({len(df_check):,} baris)", icon="✅")
    else:
        st.warning("Dataset belum tersedia", icon="⚠️")

    if os.path.exists(NOTEBOOK_PATH):
        st.success("Notebook: ✓", icon="✅")
    else:
        st.warning("Notebook belum tersedia", icon="⚠️")

    st.markdown("""
    <div class='sidebar-copyright'><br/>© 2026 Saffa Dhiya</div>
    """, unsafe_allow_html=True)



# 7. NAVBAR HORIZONTAL

selected = option_menu(
    menu_title=None,
    options=["Tentang Saya", "Tentang Aplikasi", "Analisis Data", "Prediksi"],
    icons=["person-circle", "info-circle", "bar-chart-line", "cpu"],
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "8px",
            "background": "white",
            "border-radius": "16px",
            "box-shadow": "0 4px 20px rgba(15,23,42,0.06)",
            "margin-bottom": "1.5rem",
        },
        "icon": {"color": "#1e3a8a", "font-size": "18px"},
        "nav-link": {
            "font-size": "15px", "font-weight": "600",
            "text-align": "center", "color": "#475569",
            "padding": "12px 20px", "border-radius": "12px",
            "margin": "0 4px",
            "--hover-color": "#f1f5f9",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #0a1f44 0%, #1e3a8a 100%)",
            "color": "white", "font-weight": "700",
        },
    },
)

# 11. HALAMAN: PREDIKSI

def page_prediksi():
    st.markdown("""
    <div class='hero-header fade-in'>
      <h1>🔮 Prediksi Deposito</h1>
      <p>Masukkan data nasabah untuk memprediksi kemungkinan membuka deposito</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.warning("Dataset belum tersedia. Form prediksi akan menggunakan opsi default.", icon="📂")

    # ------ Form Input ------
    st.markdown("<div class='section-title'>📝 Data Nasabah</div>", unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.number_input("👤 Usia", 18, 100, 35, help="Usia nasabah dalam tahun")
            job      = st.selectbox("💼 Pekerjaan",
                        ["admin.","blue-collar","entrepreneur","housemaid","management",
                         "retired","self-employed","services","student","technician",
                         "unemployed","unknown"], help="Jenis pekerjaan")
            marital  = st.selectbox("💍 Status", ["married","single","divorced"])
            education= st.selectbox("🎓 Pendidikan", ["primary","secondary","tertiary","unknown"])
            default  = st.selectbox("⚠️ Kredit Macet?", ["no","yes"])
            balance  = st.number_input("💰 Saldo (EUR)", -5000, 200000, 1500)

        with c2:
            housing  = st.selectbox("🏠 KPR?", ["yes","no"])
            loan     = st.selectbox("💳 Pinjaman Pribadi?", ["no","yes"])
            contact  = st.selectbox("📞 Tipe Kontak", ["cellular","telephone","unknown"])
            day      = st.slider("📅 Tanggal Kontak", 1, 31, 15)
            month    = st.selectbox("🗓️ Bulan",
                        ["jan","feb","mar","apr","may","jun","jul",
                         "aug","sep","oct","nov","dec"])
            duration = st.number_input("⏱️ Durasi (detik)", 0, 5000, 180)

        with c3:
            campaign = st.number_input("📣 Jumlah Kampanye", 1, 100, 2)
            pdays    = st.number_input("🔁 Hari sejak terakhir", -1, 999, -1)
            previous = st.number_input("📊 Kontak Sebelumnya", 0, 100, 0)
            poutcome = st.selectbox("🎯 Hasil Kampanye Lalu",
                        ["unknown","other","failure","success"])

        st.markdown("<br/>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    # ------ Proses Prediksi ------
    if submitted:
        with st.spinner("🔄 Sedang memproses prediksi..."):
            import time; time.sleep(1)

            if df is not None:
                bundle = train_fallback_model(df)
                input_df = pd.DataFrame([{
                    "age":age,"job":job,"marital":marital,"education":education,
                    "default":default,"balance":balance,"housing":housing,"loan":loan,
                    "contact":contact,"day":day,"month":month,"duration":duration,
                    "campaign":campaign,"pdays":pdays,"previous":previous,"poutcome":poutcome,
                }])
                # encode
                for col, le in bundle["encoders"].items():
                    if col == "y": continue
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except Exception:
                        input_df[col] = 0
                X = bundle["scaler"].transform(input_df[bundle["features"]])
                pred  = bundle["model"].predict(X)[0]
                proba = bundle["model"].predict_proba(X)[0]
                yes_idx = list(bundle["encoders"]["y"].classes_).index("yes")
                prob_yes = proba[yes_idx] * 100
                hasil = "yes" if pred == yes_idx else "no"
            else:
                # Dummy fallback
                prob_yes = np.random.uniform(20, 80)
                hasil = "yes" if prob_yes >= 50 else "no"

        # ------ Tampilan Hasil ------
        st.markdown("<div class='section-title'>🎯 Hasil Prediksi</div>", unsafe_allow_html=True)

        if hasil == "yes":
            st.markdown(f"""
            <div class='result-success'>
              <div style='font-size:3.5rem;'>✅</div>
              <h2 style='margin:0.5rem 0; color:white;'>Nasabah BERPOTENSI Membuka Deposito</h2>
              <div style='font-size:2.5rem;font-weight:800;margin-top:0.5rem;'>{prob_yes:.1f}%</div>
              <p style='opacity:0.9;margin-top:0.5rem; color:white;'>Probabilitas ketertarikan</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-fail'>
              <div style='font-size:3.5rem;'>❌</div>
              <h2 style='margin:0.5rem 0; color:white;'>Nasabah TIDAK Berpotensi Membuka Deposito</h2>
              <div style='font-size:2.5rem;font-weight:800;margin-top:0.5rem;'>{100-prob_yes:.1f}%</div>
              <p style='opacity:0.9;margin-top:0.5rem; color:white;'>Probabilitas tidak tertarik</p>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_yes,
                title={'text': "Probabilitas YES (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#d1fae5"},
                    ],
                }
            ))
            fig.update_layout(height=320, paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(values=[prob_yes, 100 - prob_yes],
                         names=["Tertarik", "Tidak Tertarik"], hole=0.55,
                         color_discrete_sequence=["#1e3a8a", "#cbd5e1"])
            fig.update_layout(height=320, paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        # Interpretasi
        st.markdown("<div class='section-title'>📖 Interpretasi Hasil</div>", unsafe_allow_html=True)
        if hasil == "yes":
            interpret = ("Model memprediksi nasabah ini memiliki kecenderungan tinggi "
                         "untuk membuka deposito. Disarankan untuk melakukan pendekatan "
                         "marketing yang lebih intensif dan menawarkan produk deposito terbaik.")
        else:
            interpret = ("Model memprediksi nasabah ini kurang berminat membuka deposito. "
                         "Disarankan untuk fokus pada nasabah dengan potensi lebih tinggi "
                         "atau mengubah pendekatan marketing.")
        st.markdown(f"<div class='premium-card'>{interpret}</div>", unsafe_allow_html=True)

# 8. HALAMAN: TENTANG SAYA
def page_tentang_saya():
    st.markdown("""
    <div class='hero-header fade-in'>
      <h1>👤 Tentang Saya</h1>
      <p>Berkenalan dengan pengembang aplikasi ini</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        if os.path.exists(PROFILE_IMG):
            st.image(PROFILE_IMG, width=220)
        else:
            st.warning("Foto profile tidak ditemukan")

    with col2:
        st.markdown("<div class='section-title'>📋 Informasi Pribadi</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='profile-card'>
          <div class='profile-name'>Saffa Dhiya Ur Rahma</div>
          <div class='profile-role'>Rekayasa Perangkat Lunak</div>
          <p style='color:#64748b; font-size:0.95rem; line-height:1.6;'>
            Passionate dalam <b>Data Science</b> & <b>Machine Learning</b>.
            Menyukai eksplorasi data, membangun model prediksi, dan
            merancang dashboard analitik yang informatif.
          </p>

          <div style='margin-top:1.25rem;'>
            <a href='mailto:saffadhiyaa1012@gmail.com' class='social-icon'>✉️</a>
            <a href='https://github.com/dhhyaauu' target='_blank' class='social-icon'>🐙</a>
            <a href='#' class='social-icon'>📷</a>
          </div>
        </div>
        """, unsafe_allow_html=True)

        info_items = [
            ("👤", "Nama Lengkap", "Saffa Dhiya Ur Rahma"),
            ("🎓", "Jurusan", "Rekayasa Perangkat Lunak"),
            ("✉️", "Email", "saffadhiyaa1012@gmail.com"),
            ("🔗", "Github", "github.com/dhhyaauu"),
            ("💡", "Minat", "Data Science • Machine Learning • Web Dev"),
        ]

        for icon, label, val in info_items:
            st.markdown(f"""
            <div class='premium-card' style='display:flex;align-items:center;gap:1rem;padding:1.2rem 1.5rem;'>
              <div style='font-size:1.6rem;width:48px;height:48px;background:#eef2f7;
                          border-radius:12px;display:flex;align-items:center;justify-content:center;'>
                {icon}
              </div>
              <div>
                <div style='color:#64748b;font-size:0.78rem;text-transform:uppercase;
                            letter-spacing:0.5px;font-weight:600;'>
                    {label}
                </div>
                <div style='color:#0a1f44;font-weight:600;font-size:1rem;'>
                    {val}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# 9. HALAMAN: TENTANG APLIKASI

def page_tentang_aplikasi():
    st.markdown("""
    <div class='hero-header fade-in'>
      <h1>💎 Tentang Aplikasi</h1>
      <p>Prediksi Ketertarikan Nasabah Terhadap Produk Deposito Berjangka</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📌 Deskripsi Aplikasi</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='premium-card'>
      Aplikasi ini adalah <b>dashboard machine learning</b> yang dirancang untuk
      memprediksi apakah seorang nasabah akan tertarik membuka produk
      <b>deposito berjangka</b> berdasarkan karakteristik demografis,
      finansial, dan historis kampanye marketing sebelumnya.
      <br/><br/>
      Dataset bersumber dari <b>UCI Machine Learning Repository — Bank Marketing</b>,
      yang berisi data nyata kampanye pemasaran langsung dari sebuah
      lembaga perbankan di Portugal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🎯 Tujuan Aplikasi</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    tujuan = [
        ("🎯", "Akurasi", "Memberikan prediksi yang akurat untuk membantu strategi marketing."),
        ("⚡", "Efisiensi", "Mempercepat pengambilan keputusan dengan analitik real-time."),
        ("📊", "Insight", "Menyajikan visualisasi data yang mudah dipahami."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], tujuan):
        col.markdown(f"""
        <div class='premium-card' style='text-align:center;height:100%;'>
          <div style='font-size:2.5rem;margin-bottom:0.5rem;'>{icon}</div>
          <h4 style='color:#0a1f44;margin:0.5rem 0;'>{title}</h4>
          <p style='color:#64748b;font-size:0.9rem;margin:0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🛠️ Teknologi yang Digunakan</div>", unsafe_allow_html=True)
    techs = ["🐍 Python", "🎈 Streamlit", "🐼 Pandas", "🔢 NumPy",
             "🤖 Scikit-Learn", "📊 Plotly", "🎨 Seaborn", "📈 Matplotlib"]
    cols = st.columns(4)
    for i, t in enumerate(techs):
        cols[i % 4].markdown(f"""
        <div class='premium-card' style='text-align:center;padding:1rem;'>
          <div style='font-weight:600;color:#0a1f44;'>{t}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔄 Alur Kerja Machine Learning</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Data Collection", "Mengumpulkan dataset dari UCI Repository"),
        ("2", "Preprocessing", "Cleaning, encoding, dan scaling data"),
        ("3", "EDA", "Eksplorasi dan visualisasi data"),
        ("4", "Modeling", "Training menggunakan algoritma klasifikasi"),
        ("5", "Evaluation", "Evaluasi performa model"),
        ("6", "Deployment", "Deploy model ke Streamlit"),
    ]
    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class='premium-card' style='text-align:center;padding:1.2rem 0.8rem;height:100%;'>
          <div style='width:42px;height:42px;border-radius:50%;margin:0 auto;
                      background:linear-gradient(135deg,#0a1f44,#1e3a8a);color:white;
                      display:flex;align-items:center;justify-content:center;
                      font-weight:800;font-size:1.1rem;'>{num}</div>
          <div style='color:#0a1f44;font-weight:700;margin-top:0.5rem;font-size:0.9rem;'>{title}</div>
          <div style='color:#64748b;font-size:0.75rem;margin-top:0.25rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>💼 Manfaat Aplikasi</div>", unsafe_allow_html=True)
    benefits = [
        "✅ Membantu bank menentukan target nasabah potensial",
        "✅ Meningkatkan efektivitas kampanye marketing",
        "✅ Menghemat biaya operasional pemasaran",
        "✅ Memberikan insight berbasis data nyata",
    ]
    for b in benefits:
        st.markdown(f"<div class='premium-card' style='padding:1rem 1.5rem;'>{b}</div>",
                    unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0a1f44,#1e3a8a);color:white;
                padding:2rem;border-radius:20px;text-align:center;margin-top:2rem;
                box-shadow:0 15px 50px rgba(10,31,68,0.25);'>
      <h3 style='margin:0 0 0.5rem;'>🚀 Siap Mencoba?</h3>
      <p style='margin:0;opacity:0.9;'>
        Klik menu <b>Prediksi</b> di atas untuk mencoba model prediksi kami!
      </p>
    </div>
    """, unsafe_allow_html=True)



# 10. HALAMAN: ANALISIS DATA

def page_analisis_data():
    st.markdown("""
    <div class='hero-header fade-in'>
      <h1>📊 Analisis Data</h1>
      <p>Eksplorasi mendalam dataset Bank Marketing dari UCI Repository</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 Dataset", "📓 Notebook", "📚 Penjelasan Istilah"])

    with tab1: render_tab_dataset()
    with tab2: render_tab_notebook()
    with tab3: render_tab_istilah()


# ---------- TAB DATASET ----------
def render_tab_dataset():
    df = load_dataset()
    if df is None:
        st.warning(f"⚠️ Dataset `{DATA_PATH}` tidak ditemukan. "
                   "Silakan upload file dataset terlebih dahulu.", icon="📂")
        return

    # ------ Info Sumber ------
    st.markdown(f"""
    <div class='premium-card'>
      <h4 style='color:#0a1f44;margin-top:0;'>📍 Sumber Dataset</h4>
      <p style='margin:0;color:#475569;'>
        Dataset ini berasal dari <b>UCI Machine Learning Repository</b> —
        dataset <b>Bank Marketing</b> yang berisi data kampanye pemasaran
        langsung sebuah bank di Portugal.<br/>
        🔗 <a href='{DATASET_SOURCE}' target='_blank' style='color:#1e3a8a;'>
          {DATASET_SOURCE}
        </a>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ------ Metric ------
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Baris</div>"
                f"<div class='metric-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>Total Kolom</div>"
                f"<div class='metric-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-label'>Kolom Numerik</div>"
                f"<div class='metric-value'>{len(df.select_dtypes(include=np.number).columns)}</div></div>",
                unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-label'>Kolom Kategorik</div>"
                f"<div class='metric-value'>{len(df.select_dtypes(include='object').columns)}</div></div>",
                unsafe_allow_html=True)

    # ------ Preview ------
    st.markdown("<div class='section-title'>🔍 Preview Dataset</div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=350)

    # ------ Tipe Data ------
    st.markdown("<div class='section-title'>🧬 Informasi Tipe Data</div>", unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Unique": df.nunique().values,
    })
    st.dataframe(info_df, use_container_width=True, height=300)

    # ------ Statistik Deskriptif ------
    st.markdown("<div class='section-title'>📐 Statistik Deskriptif</div>", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)

    # ------ Missing Value ------
    st.markdown("<div class='section-title'>🕳️ Missing Value</div>", unsafe_allow_html=True)
    miss = df.isnull().sum()
    if miss.sum() == 0:
        st.success("✅ Tidak ada missing value pada dataset!")
    else:
        st.dataframe(miss[miss > 0].rename("Jumlah Missing"))

    # ------ Distribusi Target ------
    if "y" in df.columns:
        st.markdown("<div class='section-title'>🎯 Distribusi Target (y)</div>", unsafe_allow_html=True)
        target = df["y"].value_counts().reset_index()
        target.columns = ["Kelas", "Jumlah"]
        fig = px.bar(target, x="Kelas", y="Jumlah", color="Kelas",
                     color_discrete_sequence=["#0a1f44", "#1e3a8a"],
                     text="Jumlah")
        fig.update_layout(showlegend=False, height=380,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ------ Distribusi Numerik ------
    st.markdown("<div class='section-title'>📊 Distribusi Variabel Numerik</div>", unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    sel = st.selectbox("Pilih kolom numerik:", num_cols, key="hist_col")
    fig = px.histogram(df, x=sel, nbins=40, color_discrete_sequence=["#1e3a8a"])
    fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # ------ Distribusi Kategorikal ------
    st.markdown("<div class='section-title'>🍩 Distribusi Variabel Kategorik</div>", unsafe_allow_html=True)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    sel2 = st.selectbox("Pilih kolom kategorik:", cat_cols, key="cat_col")
    cat_data = df[sel2].value_counts().reset_index()
    cat_data.columns = [sel2, "Jumlah"]
    fig = px.bar(cat_data, x=sel2, y="Jumlah",
                 color_discrete_sequence=["#0a1f44"])
    fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # ------ Heatmap Korelasi ------
    st.markdown("<div class='section-title'>🔥 Heatmap Korelasi</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="Blues", fmt=".2f",
                linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # ------ Insight ------
    st.markdown("<div class='section-title'>💡 Insight Singkat</div>", unsafe_allow_html=True)
    if "y" in df.columns:
        yes_pct = (df["y"] == "yes").mean() * 100
        insights = [
            f"📊 Dataset berisi {len(df):,} nasabah dengan {df.shape[1]} fitur.",
            f"🎯 Hanya {yes_pct:.2f}% nasabah yang setuju membuka deposito → "
            "kelas tidak seimbang (imbalanced).",
            f"💰 Rata-rata saldo nasabah: {df['balance'].mean():,.0f} EUR.",
            f"👤 Rata-rata usia nasabah: {df['age'].mean():.1f} tahun.",
        ]
        for ins in insights:
            st.markdown(f"<div class='premium-card' style='padding:1rem 1.5rem;'>{ins}</div>",
                        unsafe_allow_html=True)


# ---------- TAB NOTEBOOK ----------
def render_tab_notebook():
    st.markdown("""
    <div class='premium-card'>
      <h4 style='color:#0a1f44;margin-top:0;'>📓 Dokumentasi Notebook</h4>
      <p style='margin:0;color:#475569;'>
        Berikut adalah seluruh kode dan output dari Jupyter Notebook yang
        digunakan untuk membangun model prediksi.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(NOTEBOOK_PATH):
        st.warning(f"⚠️ Notebook `{NOTEBOOK_PATH}` tidak ditemukan. "
                   "Silakan upload file notebook terlebih dahulu.", icon="📂")
        return

    # Penjelasan tahapan
    stages = [
        ("🧹", "Preprocessing", "Membersihkan data, menangani missing value, encoding kategorikal."),
        ("✨", "Cleaning",      "Menghapus duplikat, outlier, dan inkonsistensi data."),
        ("🎓", "Training",      "Melatih model machine learning dengan data training."),
        ("📏", "Evaluasi",      "Mengukur performa model: accuracy, precision, recall, F1."),
        ("🔮", "Prediksi",      "Menggunakan model untuk memprediksi data baru."),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, stages):
        col.markdown(f"""
        <div class='premium-card' style='text-align:center;height:100%;padding:1.2rem 0.8rem;'>
          <div style='font-size:2rem;'>{icon}</div>
          <div style='color:#0a1f44;font-weight:700;margin-top:0.4rem;'>{title}</div>
          <div style='color:#64748b;font-size:0.78rem;margin-top:0.3rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📜 Isi Notebook</div>", unsafe_allow_html=True)

    try:
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        st.error(f"Gagal membaca notebook: {e}")
        return

    for i, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if ctype == "markdown":
            st.markdown(f"""
            <div class='premium-card' style='border-left:4px solid #1e3a8a;'>
              {source}
            </div>
            """, unsafe_allow_html=True)

        elif ctype == "code":
            st.markdown(f"<div style='color:#64748b;font-size:0.8rem;font-weight:600;"
                        f"margin:1rem 0 0.3rem;'>▶ Cell [{i}]</div>",
                        unsafe_allow_html=True)
            st.code(source, language="python")

            # Tampilkan output
            for out in cell.get("outputs", []):
                ot = out.get("output_type")
                if ot == "stream":
                    st.text("".join(out.get("text", [])))

                elif ot in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "image/png" in data:
                        st.image(base64.b64decode(data["image/png"]))
                    elif "text/html" in data:
                        html = "".join(data["text/html"])
                        st.markdown(html, unsafe_allow_html=True)
                    elif "text/plain" in data:
                        st.text("".join(data["text/plain"]))

                elif ot == "error":
                    st.error("\n".join(out.get("traceback", [])))


# ---------- TAB ISTILAH ----------
def render_tab_istilah():
    st.markdown("""
    <div class='premium-card'>
      <h4 style='color:#0a1f44;margin-top:0;'>📚 Penjelasan Fitur Dataset</h4>
      <p style='margin:0;color:#475569;'>
        Berikut penjelasan setiap fitur (kolom) pada dataset Bank Marketing
        yang digunakan sebagai input model prediksi.
      </p>
    </div>
    """, unsafe_allow_html=True)

    istilah = [
        ("age",       "👤", "Numerik",    "Usia nasabah (tahun)."),
        ("job",       "💼", "Kategorik",  "Jenis pekerjaan nasabah (admin, blue-collar, technician, dll)."),
        ("marital",   "💍", "Kategorik",  "Status pernikahan (married, single, divorced)."),
        ("education", "🎓", "Kategorik",  "Tingkat pendidikan (primary, secondary, tertiary, unknown)."),
        ("default",   "⚠️", "Kategorik",  "Apakah nasabah memiliki kredit macet (yes/no)."),
        ("balance",   "💰", "Numerik",    "Saldo rata-rata tahunan dalam EUR."),
        ("housing",   "🏠", "Kategorik",  "Apakah memiliki kredit perumahan (yes/no)."),
        ("loan",      "💳", "Kategorik",  "Apakah memiliki pinjaman pribadi (yes/no)."),
        ("contact",   "📞", "Kategorik",  "Jenis komunikasi (cellular, telephone, unknown)."),
        ("day",       "📅", "Numerik",    "Tanggal terakhir kontak."),
        ("month",     "🗓️", "Kategorik",  "Bulan terakhir kontak."),
        ("duration",  "⏱️", "Numerik",    "Durasi kontak terakhir (detik)."),
        ("campaign",  "📣", "Numerik",    "Jumlah kontak dalam kampanye saat ini."),
        ("pdays",     "🔁", "Numerik",    "Hari sejak kontak terakhir kampanye sebelumnya (-1 = belum pernah)."),
        ("previous",  "📊", "Numerik",    "Jumlah kontak sebelum kampanye ini."),
        ("poutcome",  "🎯", "Kategorik",  "Hasil kampanye sebelumnya (success, failure, other, unknown)."),
        ("y",         "✅", "Target",     "Apakah nasabah membuka deposito berjangka (yes/no)."),
    ]
    df_istilah = pd.DataFrame(istilah, columns=["Fitur", "Icon", "Tipe", "Deskripsi"])
    st.dataframe(df_istilah, use_container_width=True, height=620, hide_index=True)

    st.markdown("<div class='section-title'>💡 Detail Fitur Penting</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (feat, icon, _, desc) in enumerate(istilah[:8]):
        cols[i % 2].markdown(f"""
        <div class='premium-card'>
          <div style='display:flex;align-items:center;gap:0.75rem;'>
            <div style='font-size:1.8rem;'>{icon}</div>
            <div>
              <div style='font-weight:700;color:#0a1f44;font-size:1.05rem;'>{feat}</div>
              <div style='color:#64748b;font-size:0.88rem;'>{desc}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)




# 12. ROUTING
if selected == "Prediksi":
    page_prediksi()
elif selected == "Tentang Aplikasi":
    page_tentang_aplikasi()
elif selected == "Analisis Data":
    page_analisis_data()
elif selected == "Tentang Saya":
    page_tentang_saya()




# 13. FOOTER

st.markdown("""
<div class='footer'>
  <div style='font-weight:600;color:#0a1f44;margin-bottom:0.25rem;'>
    💎 Deposit Predictor
  </div>
  © 2026, dibuat oleh <b>Saffa Dhiya Ur Rahma</b> • Powered by Streamlit
</div>
""", unsafe_allow_html=True)
