# 1. IMPPORTS

import os
import json
import base64
import warnings
import re
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


# 3. SIDEBAR INTERAKTIF UNTUK PILIHAN TEMA

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
      <div style='font-size:2.5rem;'>💎</div>
      <div style='font-family: "Plus Jakarta Sans", sans-serif; font-weight: 800; font-size: 1.3rem; color: #000000 !important;'>Deposit Predictor</div>
      <div style='color:#475569 !important; font-size:0.85rem; font-weight:600;'>Premium ML Dashboard</div>
    </div>
    <hr style='margin:0.5rem 0; border-color:#cbd5e1;'/>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌓 Pengaturan Tampilan")
    theme_choice = st.radio("Pilih Tema Aplikasi:", ["Terang (Light Mode)", "Gelap (Dark Mode)"], index=0)
    st.markdown("<hr style='margin:1rem 0; border-color:#cbd5e1;'/>", unsafe_allow_html=True)

# Menentukan variabel warna berdasarkan tema pilihan user
if theme_choice == "Terang (Light Mode)":
    bg_app = "linear-gradient(135deg, #f7f9fc 0%, #eef2f7 100%)"
    text_main = "#0f172a"
    card_bg = "#ffffff"
    card_border = "rgba(15, 23, 42, 0.05)"
    card_shadow = "rgba(15, 23, 42, 0.06)"
    card_shadow_hover = "rgba(15, 23, 42, 0.12)"
    label_color = "#0a1f44"   # Warna teks label input (sangat kontras)
    widget_bg = "#ffffff"     # Background dalam box input
    widget_text = "#0f172a"   # Teks di dalam box input
else:
    bg_app = "linear-gradient(135deg, #0f172a 0%, #020617 100%)"
    text_main = "#f8fafc"
    card_bg = "#1e293b"
    card_border = "rgba(255, 255, 255, 0.1)"
    card_shadow = "rgba(0, 0, 0, 0.3)"
    card_shadow_hover = "rgba(0, 0, 0, 0.5)"
    label_color = "#38bdf8"   # Warna teks label input biru muda terang agar menyala di tema gelap
    widget_bg = "#0f172a"     # Background dalam box input
    widget_text = "#ffffff"   # Teks di dalam box input


# 4. CUSTOM CSS DENGAN DUKUNGAN TEMA DINAMIS & FIX KONTRAS LABEL

CUSTOM_CSS = f"""
<style>
    /* ---------- Import Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* ---------- Background ---------- */
    .stApp {{
        background: {bg_app};
        color: {text_main};
    }}

    /* ---------- Perbaikan Total Baca Teks Input & Label ---------- */
    /* Menargetkan semua label bawaan streamlit agar tebal, jelas, dan kontras */
    div[data-testid="stWidgetLabel"] p, 
    label, 
    .stSlider p,
    div[data-testid="stForm"] label {{
        color: {label_color} !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
        margin-bottom: 4px !important;
    }}
    
    /* Memperbaiki teks input, selectbox, dan slider agar box-nya terlihat kontras */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] div {{
        color: {widget_text} !important;
        background-color: {widget_bg} !important;
    }}

    /* ---------- Header Gradient ---------- */
    .hero-header {{
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 50%, #0f172a 100%);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        color: #ffffff;
        box-shadow: 0 20px 60px rgba(10, 31, 68, 0.25);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }}
    .hero-header::before {{
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }}
    .hero-header h1 {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.4rem; font-weight: 800;
        margin: 0; letter-spacing: -0.5px;
        color: #ffffff !important;
    }}
    .hero-header p {{
        opacity: 0.85; font-size: 1.05rem;
        margin-top: 0.5rem; font-weight: 400;
        color: #ffffff !important;
    }}

    /* ---------- Card Premium ---------- */
    .premium-card {{
        background: {card_bg};
        padding: 1.75rem;
        border-radius: 18px;
        box-shadow: 0 4px 24px {card_shadow};
        border: 1px solid {card_border};
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1rem;
        color: {text_main};
    }}
    .premium-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px {card_shadow_hover};
        border-color: rgba(30, 58, 138, 0.3);
    }}

    /* ---------- Metric Card ---------- */
    .metric-card {{
        background: {card_bg};
        padding: 1.5rem; border-radius: 16px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 12px {card_shadow};
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{ transform: translateY(-2px); }}
    .metric-value {{
        font-size: 2rem; font-weight: 800; color: #1e3a8a;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    if theme_choice == "Gelap (Dark Mode)":
    .metric-value {{ color: #38bdf8; }}
    .metric-label {{
        color: #64748b; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.5px;
        font-weight: 600;
    }}

    /* ---------- Profile Card Bulat Estetik ---------- */
    .profile-card {{
        background: {card_bg};
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 60px {card_shadow};
        transition: all 0.4s ease;
        border: 1px solid {card_border};
    }}
    .profile-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 30px 80px {card_shadow_hover};
    }}
    .profile-img {{
        width: 180px; height: 180px;
        border-radius: 50% !important;
        object-fit: cover;
        border: 6px solid #1e3a8a;
        box-shadow: 0 10px 30px rgba(10, 31, 68, 0.2);
        margin: 0 auto 1.25rem auto;
        display: block;
    }}
    .profile-name {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem; font-weight: 800;
        color: {text_main}; margin: 0.5rem 0 0.25rem;
    }}
    .profile-role {{
        color: #1e3a8a; font-weight: 600;
        font-size: 0.95rem; margin-bottom: 1rem;
    }}
    if theme_choice == "Gelap (Dark Mode)":
    .profile-role {{ color: #38bdf8; }}
    
    .social-icon {{
        display: inline-flex;
        width: 46px; height: 46px;
        align-items: center; justify-content: center;
        background: #f1f5f9; border-radius: 50%;   
        margin: 0 0.35rem; text-decoration: none;
        color: #0a1f44; font-size: 1.2rem;
        transition: all 0.3s ease;
    }}
    .social-icon:hover {{
        background: #0a1f44; color: white;
        transform: translateY(-3px);
    }}

    /* ---------- Section Title ---------- */
    .section-title {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: {label_color};
        border-left: 4px solid #1e3a8a;
        padding-left: 12px; margin: 1.5rem 0 1rem;
    }}

    /* ---------- Buttons ---------- */
    .stButton>button {{
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(10, 31, 68, 0.25) !important;
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(10, 31, 68, 0.35) !important;
    }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(15, 23, 42, 0.06);
    }}
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {{
        color: #0f172a !important; /* Sidebar tetap terbaca gelap agar konsisten */
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {card_bg}; padding: 6px;
        border-radius: 14px;
        box-shadow: 0 2px 12px {card_shadow};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        color: #64748b;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 100%) !important;
        color: white !important;
    }}

    /* ---------- Result Card ---------- */
    .result-success {{
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: white !important; padding: 2rem; border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(4, 120, 87, 0.4);
        animation: resultPulse 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }}
    .result-fail {{
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: white !important; padding: 2rem; border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(185, 28, 28, 0.4);
        animation: resultPulse 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
    }}
    .result-success h2, .result-success p, .result-fail h2, .result-fail p {{
        color: white !important;
    }}
    
    @keyframes resultPulse {{
        0% {{ transform: scale(0.8); opacity: 0; }}
        70% {{ transform: scale(1.03); }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    .fade-in {{ animation: fadeIn 0.8s ease; }}

    /* ---------- Notebook Output Styling Fix ---------- */
    .notebook-text-output {{
        background-color: #0f172a !important;
        color: #f8fafc !important;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        word-break: break-all;
        margin: 0.5rem 0;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
    }}

    /* ---------- Footer ---------- */
    .footer {{
        text-align: center; padding: 2rem 1rem 1rem;
        color: #64748b; font-size: 0.9rem;
        border-top: 1px solid {card_border};
        margin-top: 3rem;
    }}

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu, footer, header {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# 5. KONSTANTA & PATH

DATA_PATH       = "bank-full.csv"
NOTEBOOK_PATH   = "notebook.ipynb"
MODEL_PATH      = "model_deposito.joblib"
PROFILE_IMG     = "assets/saffa.png"  # Diubah dari fotoprofile.jpeg ke saffa.png sesuai request
DATASET_SOURCE  = "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"


# 6. UTILITY FUNCTIONS

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


def clean_ansi_codes(text: str) -> str:
    """Membersihkan kode warna terminal (ANSI) agar teks terbaca murni."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


# 7. SISA SIDEBAR INFO

with st.sidebar:
    st.markdown("### 📖 Panduan Penggunaan")
    st.markdown("""
    **Langkah-langkah:**
    1. Mulai dari halaman **Prediksi** untuk melakukan simulasi nasabah
    2. Pelajari detail program di **Tentang Aplikasi**
    3. Lihat eksplorasi data mendalam pada **Analisis Data**
    4. Cek profil pengembang di **Tentang Saya**
    """)

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
    <div style='text-align:center; margin-top:2rem; padding-top:1rem; border-top:1px solid #e2e8f0; color:#94a3b8; font-size:0.75rem;'><br/>© 2026 Saffa Dhiya</div>
    """, unsafe_allow_html=True)


# 8. NAVBAR HORIZONTAL

selected = option_menu(
    menu_title=None,
    options=["Prediksi", "Tentang Aplikasi", "Analisis Data", "Tentang Saya"],
    icons=["cpu", "info-circle", "bar-chart-line", "person-circle"],
    default_index=0,
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


# 9. HALAMAN: PREDIKSI

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
                prob_yes = np.random.uniform(20, 80)
                hasil = "yes" if prob_yes >= 50 else "no"

        st.markdown("<div class='section-title'>🎯 Hasil Prediksi</div>", unsafe_allow_html=True)

        if hasil == "yes":
            st.markdown(f"""
            <div class='result-success'>
              <div style='font-size:3.5rem;'>✅</div>
              <h2 style='margin:0.5rem 0;'>Nasabah BERPOTENSI Membuka Deposito</h2>
              <div style='font-size:2.5rem;font-weight:800;margin-top:0.5rem;'>{prob_yes:.1f}%</div>
              <p style='opacity:0.9;margin-top:0.5rem;'>Probabilitas ketertarikan</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-fail'>
              <div style='font-size:3.5rem;'>❌</div>
              <h2 style='margin:0.5rem 0;'>Nasabah TIDAK Berpotensi Membuka Deposito</h2>
              <div style='font-size:2.5rem;font-weight:800;margin-top:0.5rem;'>{100-prob_yes:.1f}%</div>
              <p style='opacity:0.9;margin-top:0.5rem;'>Probabilitas tidak tertarik</p>
            </div>
            """, unsafe_allow_html=True)

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


# 10. HALAMAN: TENTANG SAYA (DENGAN REVISI FOTO BULAT SEMPURNA)

def page_tentang_saya():
    st.markdown("""
    <div class='hero-header fade-in'>
      <h1>👤 Tentang Saya</h1>
      <p>Berkenalan dengan pengembang aplikasi ini</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        img_b64 = image_to_base64(PROFILE_IMG)
        if img_b64:
            # Menggunakan tag img HTML murni dengan class profile-img agar bulat sempurna & estetik
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_b64}" class="profile-img" alt="Saffa Profile">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Foto profile '{PROFILE_IMG}' tidak ditemukan di folder assets.")

    with col2:
        st.markdown("<div class='section-title'>📋 Informasi Pribadi</div>", unsafe_allow_html=True)

        st.markdown(f"""
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
                <div style='color:{text_main};font-weight:600;font-size:1rem;'>
                    {val}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# 11. HALAMAN: TENTANG APLIKASI

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


# 12. HALAMAN: ANALISIS DATA

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


def render_tab_dataset():
    df = load_dataset()
    if df is None:
        st.warning(f"⚠️ Dataset `{DATA_PATH}` tidak ditemukan. Silakan upload file dataset terlebih dahulu.", icon="📂")
        return

    st.markdown(f"""
    <div class='premium-card'>
      <h4 style='color:#0a1f44;margin-top:0;'>📍 Sumber Dataset</h4>
      <p style='margin:0;color:#475569;'>
        Dataset ini berasal dari <b>UCI Machine Learning Repository</b> —
        dataset <b>Bank Marketing</b> yang berisi data kampanye pemasaran langsung sebuah bank di Portugal.<br/>
        🔗 <a href='{DATASET_SOURCE}' target='_blank' style='color:#1e3a8a;'>{DATASET_SOURCE}</a>
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Baris</div><div class='metric-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-label'>Total Kolom</div><div class='metric-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-label'>Kolom Numerik</div><div class='metric-value'>{len(df.select_dtypes(include=np.number).columns)}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-label'>Kolom Kategorik</div><div class='metric-value'>{len(df.select_dtypes(include='object').columns)}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔍 Preview Dataset</div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=350)


def render_tab_notebook():
    st.markdown("""
    <div class='premium-card'>
      <h4 style='color:#0a1f44;margin-top:0;'>📓 Dokumentasi Notebook</h4>
      <p style='margin:0;color:#475569;'>Berikut adalah seluruh kode dan output dari Jupyter Notebook yang digunakan untuk membangun model prediksi.</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(NOTEBOOK_PATH):
        st.warning(f"⚠️ Notebook `{NOTEBOOK_PATH}` tidak ditemukan.", icon="📂")
        return

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
            st.markdown(f"<div class='premium-card' style='border-left:4px solid #1e3a8a;'>{source}</div>", unsafe_allow_html=True)
        elif ctype == "code":
            st.markdown(f"<div style='color:#64748b;font-size:0.8rem;font-weight:600;margin:1rem 0 0.3rem;'>▶ Cell [{i}]</div>", unsafe_allow_html=True)
            st.code(source, language="python")

            for out in cell.get("outputs", []):
                ot = out.get("output_type")
                if ot == "stream":
                    raw_text = "".join(out.get("text", []))
                    st.markdown(f"<div class='notebook-text-output'>{clean_ansi_codes(raw_text)}</div>", unsafe_allow_html=True)
                elif ot in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "image/png" in data:
                        st.image(base64.b64decode(data["image/png"]))
                    elif "text/html" in data:
                        st.markdown("".join(data["text/html"]), unsafe_allow_html=True)
                    elif "text/plain" in data:
                        raw_text = "".join(data["text/plain"])
                        st.markdown(f"<div class='notebook-text-output'>{clean_ansi_codes(raw_text)}</div>", unsafe_allow_html=True)
                elif ot == "error":
                    st.error(clean_ansi_codes("\n".join(out.get("traceback", []))))


def render_tab_istilah():
    istilah = [
        ("age",       "👤", "Numerik",    "Usia nasabah (tahun)."),
        ("job",       "💼", "Kategorik",  "Jenis pekerjaan nasabah."),
        ("marital",   "💍", "Kategorik",  "Status pernikahan."),
        ("education", "🎓", "Kategorik",  "Tingkat pendidikan."),
        ("default",   "⚠️", "Kategorik",  "Apakah nasabah memiliki kredit macet."),
        ("balance",   "💰", "Numerik",    "Saldo rata-rata tahunan dalam EUR."),
        ("housing",   "🏠", "Kategorik",  "Apakah memiliki kredit perumahan."),
        ("loan",      "💳", "Kategorik",  "Apakah memiliki pinjaman pribadi."),
    ]
    df_istilah = pd.DataFrame(istilah, columns=["Fitur", "Icon", "Tipe", "Deskripsi"])
    st.dataframe(df_istilah, use_container_width=True, hide_index=True)


# 13. ROUTING
if selected == "Prediksi":
    page_prediksi()
elif selected == "Tentang Aplikasi":
    page_tentang_aplikasi()
elif selected == "Analisis Data":
    page_analisis_data()
elif selected == "Tentang Saya":
    page_tentang_saya()


# 14. FOOTER
st.markdown(f"""
<div class='footer'>
  <div style='font-weight:600;color:#0a1f44;margin-bottom:0.25rem;'>💎 Deposit Predictor</div>
  © 2026, dibuat oleh <b>Saffa Dhiya Ur Rahma</b> • Powered by Streamlit
</div>
""", unsafe_allow_html=True)
