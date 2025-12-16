import streamlit as st
from pathlib import Path

from src.db import init_db
from src.auth import login_ui, signup_ui, logout, require_login
from src.utils import load_css
from src.ml import predict_page, model_info_page, charts_page
from src.utils import treatment_page

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

BASE_DIR = Path(__file__).parent
load_css(BASE_DIR / "assets" / "styles.css")

init_db(BASE_DIR / "db" / "app.db")

if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# --- AUTH GATE ---
if st.session_state.user is None:
    st.title("🩺 Diabetes Prediction App")
    tabs = st.tabs(["🔐 Login", "🆕 Sign Up"])

    with tabs[0]:
        login_ui()
    with tabs[1]:
        signup_ui()

    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
    f"""
    <div class="welcome-box">
        👋 <b>Welcome, {st.session_state.user['username']}!</b>
    </div>
    """,
    unsafe_allow_html=True
)


    st.markdown("### 🧭 Navigation")
    page = st.radio(
        label="",
        options=[
            "🏠 Home",
            "🧪 Predict Diabetes",
            "💊 Treatment",
            "ℹ️ Model Info",
            "📊 Charts & Visualization",
            "🔓 Logout",
        ],
        index=0
    )

st.session_state.page = page

# --- ROUTING ---
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center'>🏠 Home</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px'>This is the Diabetes Prediction App homepage.</p>", unsafe_allow_html=True)

elif page == "🧪 Predict Diabetes":
    require_login()
    predict_page(base_dir=BASE_DIR)

elif page == "💊 Treatment":
    require_login()
    treatment_page()

elif page == "ℹ️ Model Info":
    require_login()
    model_info_page(base_dir=BASE_DIR)

elif page == "📊 Charts & Visualization":
    require_login()
    charts_page(base_dir=BASE_DIR)

elif page == "🔓 Logout":
    logout()
    st.rerun()
