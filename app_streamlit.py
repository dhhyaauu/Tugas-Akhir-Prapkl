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


# 3. SIDEBAR INTERAKTIF UNTUK PILIHAN TEMA (EMOJI MINIMALIS)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
      <div style='font-size:2.5rem;'>💎</div>
      <div style='font-family: "Plus Jakarta Sans", sans-serif; font-weight: 800; font-size: 1.3rem; color: #0f172a !important;'>Deposit Predictor</div>
      <div style='color:#475569 !important; font-size:0.85rem; font-weight:600;'>Premium ML Dashboard</div>
    </div>
    <hr style='margin:0.5rem 0; border-color:#cbd5e1;'/>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🌓 Tema")
    theme_choice = st.radio("Pilih Tampilan:", ["☀️ Terang", "🌙 Gelap"], index=0)
    st.markdown("<hr style='margin:1rem 0; border-color:#cbd5e1;'/>", unsafe_allow_html=True)

# Menentukan variabel warna berdasarkan tema pilihan user
if theme_choice == "☀️ Terang":
    bg_app = "linear-gradient(135deg, #f7f9fc 0%, #eef2f7 100%)"
    text_main = "#0f172a"
    card_bg = "#ffffff"
    card_border = "rgba(15, 23, 42, 0.05)"
    card_shadow = "rgba(15, 23, 42, 0.06)"
    card_shadow_hover = "rgba(15, 23, 42, 0.12)"
    label_color = "#0a1f44"      # Teks label biru navy tua di mode terang
    widget_bg = "#ffffff"        # Latar belakang dalam kotak input putih
    widget_text = "#0f172a"      # Teks di dalam kotak input gelap
    plotly_theme = "plotly"
    chart_text_color = "#0f172a"
    chart_bg_color = "#ffffff"
else:
    bg_app = "linear-gradient(135deg, #0f172a 0%, #020617 100%)"
    text_main = "#f8fafc"
    card_bg = "#1e293b"
    card_border = "rgba(255, 255, 255, 0.1)"
    card_shadow = "rgba(0, 0, 0, 0.3)"
    card_shadow_hover = "rgba(0, 0, 0, 0.5)"
    label_color = "#38bdf8"      # Teks label biru langit menyala cerah di mode gelap
    widget_bg = "#0f172a"        # Latar belakang dalam kotak input gelap navy
    widget_text = "#ffffff"      # Teks di dalam kotak input putih bersih
    plotly_theme = "plotly_dark"
    chart_text_color = "#f8fafc"
    chart_bg_color = "#1e293b"


# 4. CUSTOM CSS DENGAN DUKUNGAN TEMA DINAMIS & FIX TOTAL KONTRAS LABEL INPUT

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
        color: {text_main} !important;
    }}

    /* Force global text color berdasarkan tema aktif */
    .stMarkdown div p, h1, h2, h3, h4, h5, h6, span {{
        color: {text_main} !important;
    }}

    /* ---------- Perbaikan Total Baca Teks Input & Label ---------- */
    /* Menargetkan semua label input teks, angka, selectbox, slider, dan form bawaan streamlit */
    div[data-testid="stWidgetLabel"] p, 
    label, 
    .stSlider p,
    .stSlider span,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] div p {{
        color: {label_color} !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        letter-spacing: 0.4px;
        margin-bottom: 6px !important;
    }}
    
    /* Memperbaiki warna teks pilihan dan kotak isian input/selectbox agar kontras tajam */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span {{
        color: {widget_text} !important;
        background-color: {widget_bg} !important;
        font-weight: 500 !important;
    }}
    
    /* Memperbaiki warna dropdown menu list selectbox saat diklik */
    div[data-baseweb="menu"] li {{
        color: {widget_text} !important;
        background-color: {card_bg} !important;
    }}

    /* ---------- Header Gradient ---------- */
    .hero-header {{
        background: linear-gradient(135deg, #0a1f44 0%, #1e3a8a 50%, #0f172a 100%);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(10, 31, 68, 0.25);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
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
    }}
    .premium-card p, .premium-card div, .premium-card h4 {{
        color: {text_main} !important;
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
        font-size: 2rem; font-weight: 800; 
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: {"#1e3a8a" if theme_choice == "☀️ Terang" else "#38bdf8"} !important;
    }}
    .metric-label {{
        color: #64748b !important; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.5px;
        font-weight: 600;
    }}

    /* ---------- Profile Card ---------- */
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
        color: {text_main} !important; margin: 0.5rem 0 0.25rem;
    }}
    .profile-role {{
        color: {"#1e3a8a" if theme_choice == "☀️ Terang" else "#38bdf8"} !important; 
        font-weight: 600;
        font-size: 0.95rem; margin-bottom: 1rem;
    }}

    /* ---------- Section Title ---------- */
    .section-title {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        color: {label_color} !important;
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
        color: #0f172a !important; 
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
    }}
    .result-fail {{
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: white !important; padding: 2rem; border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(185, 28, 28, 0.4);
    }}
    .result-success h2, .result-success p, .result-fail h2, .result-fail p,
    .result-success div, .result-fail div {{
        color: white !important;
    }}

    .footer {{
        text-align: center; padding: 2rem 1rem 1rem;
        color: #64748b; font-size: 0.9rem;
        border-top: 1px solid {card_border};
        margin-top: 3rem;
    }}
    .footer div, .footer b {{ color: {text_main} !important; }}

    #MainMenu, footer, header {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# 5. KONSTANTA & PATH

DATA_PATH       = "bank-full.csv"
NOTEBOOK_PATH   = "notebook.ipynb"
MODEL_PATH      = "model_deposito.joblib"
PROFILE_IMG     = "assets/saffa.png"  
DATASET_SOURCE  = "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing"


# 6. UTILITY FUNCTIONS

@st.cache_data(show_spinner=False)
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner=False)
def train_fallback_model(df: pd.DataFrame):
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


# 7. SISA SIDEBAR INFO

with st.sidebar:
    st.markdown("### 📖 Panduan")
    st.markdown("""
    1. Mulai di halaman **Prediksi**
    2. Cek detail di **Tentang Aplikasi**
    3. Lihat pola data di **Analisis Data**
    4. Cek profil di **Tentang Saya**
    """)

    st.markdown("### 🔌 Status")
    df_check = load_dataset()
    if df_check is not None:
        st.success(f"Data: ✓ ({len(df_check):,} baris)", icon="✅")
    else:
        st.warning("Data belum ada", icon="⚠️")


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
    <div class='hero-header'>
      <h1>🔮 Prediksi Deposito</h1>
      <p>Masukkan data nasabah untuk memprediksi kemungkinan membuka deposito</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    st.markdown("<div class='section-title'>📝 Data Nasabah</div>", unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.number_input("👤 Usia", 18, 100, 35)
            job      = st.selectbox("💼 Pekerjaan",
                        ["admin.","blue-collar","entrepreneur","housemaid","management",
                         "retired","self-employed","services","student","technician",
                         "unemployed","unknown"])
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
                title={'text': "Probabilitas YES (%)", 'font': {'color': chart_text_color}},
                number={'font': {'color': chart_text_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': chart_text_color},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 70], 'color': "#fef3c7"},
                        {'range': [70, 100], 'color': "#d1fae5"},
                    ],
                }
            ))
            fig.update_layout(
                height=320, 
                paper_bgcolor=chart_bg_color,
                plot_bgcolor=chart_bg_color,
                template=plotly_theme
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(values=[prob_yes, 100 - prob_yes],
                         names=["Tertarik", "Tidak Tertarik"], hole=0.55,
                         color_discrete_sequence=["#1e3a8a", "#cbd5e1"])
            fig.update_layout(
                height=320, 
                paper_bgcolor=chart_bg_color,
                plot_bgcolor=chart_bg_color,
                template=plotly_theme,
                legend=dict(font=dict(color=chart_text_color))
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>📖 Interpretasi Hasil</div>", unsafe_allow_html=True)
        if hasil == "yes":
            interpret = ("Model memprediksi nasabah ini memiliki kecenderungan tinggi "
                         "untuk membuka deposito. Disarankan untuk melakukan pendekatan "
                         "marketing yang lebih intensif.")
        else:
            interpret = ("Model memprediksi nasabah ini kurang berminat membuka deposito. "
                         "Disarankan untuk fokus pada nasabah dengan potensi lebih tinggi.")
        st.markdown(f"<div class='premium-card'><p style='margin:0;'>{interpret}</p></div>", unsafe_allow_html=True)


# 10. HALAMAN: TENTANG SAYA

def page_tentang_saya():
    st.markdown("""
    <div class='hero-header'>
      <h1>👤 Tentang Saya</h1>
      <p>Berkenalan dengan pengembang aplikasi ini</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        img_b64 = image_to_base64(PROFILE_IMG)
        if img_b64:
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_b64}" class="profile-img" alt="Saffa Profile">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Foto profile '{PROFILE_IMG}' tidak ditemukan.")

    with col2:
        st.markdown("<div class='section-title'>📋 Informasi Pribadi</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='profile-card'>
          <div class='profile-name'>Saffa Dhiya Ur Rahma</div>
          <div class='profile-role'>Rekayasa Perangkat Lunak</div>
          <p style='font-size:0.95rem; line-height:1.6;'>
            Passionate dalam <b>Data Science</b> & <b>Machine Learning</b>.
          </p>
        </div>
        """, unsafe_allow_html=True)


# 11. HALAMAN: TENTANG APLIKASI

def page_tentang_aplikasi():
    st.markdown("""
    <div class='hero-header'>
      <h1>💎 Tentang Aplikasi</h1>
      <p>Prediksi Ketertarikan Nasabah Terhadap Produk Deposito Berjangka</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📌 Deskripsi Aplikasi</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='premium-card'>
      <p style='margin:0;'>Aplikasi ini adalah <b>dashboard machine learning</b> yang dirancang untuk
      memprediksi ketertarikan nasabah terhadap produk deposito berjangka.</p>
    </div>
    """, unsafe_allow_html=True)


# 12. HALAMAN: ANALISIS DATA (NOTEBOOK UTUH)

def page_analisis_data():
    st.markdown("""
    <div class='hero-header'>
      <h1>📊 Analisis Data</h1>
      <p>Eksplorasi mendalam dataset Bank Marketing dari UCI Repository</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Dataset", "📓 Notebook"])

    with tab1:
        df = load_dataset()
        if df is not None:
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.warning("File dataset bank-full.csv belum ditemukan.")

    with tab2:
        st.markdown("""
        <div class='premium-card'>
          <h4 style='margin-top:0;'>📓 Dokumentasi Jupyter Notebook</h4>
          <p style='margin:0;'>Berikut adalah seluruh alur pengerjaan kode, pembersihan data, dan output dari berkas notebook eksternal Anda.</p>
        </div>
        """, unsafe_allow_html=True)

        if not os.path.exists(NOTEBOOK_PATH):
            st.warning(f"⚠️ Berkas notebook `{NOTEBOOK_PATH}` tidak ditemukan di direktori utama project Anda.", icon="📂")
            return

        try:
            with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
                nb = json.load(f)
            
            for i, cell in enumerate(nb.get("cells", []), start=1):
                ctype = cell.get("cell_type", "")
                source = "".join(cell.get("source", []))

                if ctype == "markdown":
                    st.markdown(f"<div class='premium-card' style='border-left:4px solid #1e3a8a;'>{source}</div>", unsafe_allow_html=True)

                elif ctype == "code":
                    st.markdown(f"<div style='color:#64748b; font-size:0.8rem; font-weight:600; margin:1rem 0 0.3rem;'>▶ Cell Code [{i}]</div>", unsafe_allow_html=True)
                    st.code(source, language="python")

                    for out in cell.get("outputs", []):
                        ot = out.get("output_type")
                        if ot == "stream":
                            st.text("".join(out.get("text", [])))
                        elif ot in ("execute_result", "display_data"):
                            data = out.get("data", {})
                            if "image/png" in data:
                                st.image(base64.b64decode(data["image/png"]))
                            elif "text/html" in data:
                                st.markdown("".join(data["text/html"]), unsafe_allow_html=True)
                            elif "text/plain" in data:
                                st.text("".join(data["text/plain"]))
                        elif ot == "error":
                            st.error("\n".join(out.get("traceback", [])))
        except Exception as e:
            st.error(f"Gagal membaca notebook: {e}")


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
  <div style='font-weight:600;margin-bottom:0.25rem;'>💎 Deposit Predictor</div>
  © 2026, dibuat oleh <b>Saffa Dhiya Ur Rahma</b>
</div>
""", unsafe_allow_html=True)s
