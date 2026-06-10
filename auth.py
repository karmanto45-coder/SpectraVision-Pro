import hashlib
import streamlit as st

# load_users / save_users sekarang didelegasikan ke database.py
# agar data user tersimpan di Supabase (persisten)
from database import load_users, save_users, delete_user


def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def verify_login(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return users[username]
    return None


def get_role():
    return st.session_state.get("role", None)


def is_admin():
    return st.session_state.get("role") == "admin"


def is_logged_in():
    return st.session_state.get("logged_in", False)


def logout():
    for key in ["logged_in", "role", "username", "display_name"]:
        if key in st.session_state:
            del st.session_state[key]


def render_login():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}

    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    [data-testid="stToolbar"] {visibility:hidden;}
    a[href*="github"] {display:none !important;}
    .stDeployButton {display:none !important;}

    .stApp { background: #080d14; }
    [data-testid="stSidebar"] { display: none; }

    .login-outer {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-height: 88vh; padding: 2rem;
    }
    .login-card {
        background: linear-gradient(145deg, #0f1520, #141e2e);
        border: 1px solid #1e3a5f;
        border-radius: 20px;
        padding: 2.8rem 2.5rem 2rem;
        width: 100%; max-width: 420px;
        box-shadow: 0 0 60px rgba(56,139,253,0.08), 0 20px 40px rgba(0,0,0,0.4);
    }
    .logo-icon { font-size: 2.8rem; text-align: center; margin-bottom: 0.5rem; }
    .login-logo {
        font-family: 'DM Mono', monospace !important;
        font-size: 2.2rem !important; font-weight: 600 !important;
        text-align: center !important; color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: none !important; margin: 0.3rem 0 !important;
        text-shadow: 0 0 30px rgba(125,211,252,0.5), 0 0 60px rgba(59,130,246,0.3);
    }
    .login-sub {
        color: #64748b !important; font-size: 0.72rem !important;
        text-align: center !important; letter-spacing: 0.08em !important;
        margin: 0.4rem 0 1rem !important;
    }
    .login-version {
        display: block !important; text-align: center !important;
        font-size: 0.68rem !important; background: #0d1829 !important;
        border: 1px solid #1e3a5f !important; color: #7dd3fc !important;
        padding: 3px 14px !important; border-radius: 20px !important;
        margin: 0 auto 1.5rem !important;
        font-family: 'DM Mono', monospace !important;
        width: fit-content !important;
    }
    </style>

    <div class="login-outer">
      <div class="login-card">
        <div class="logo-icon">🔭</div>
        <p class="login-logo">SpectraVision Pro</p>
        <p class="login-sub">MULTIVARIATE CURVE RESOLUTION &nbsp;·&nbsp; SPECTRAL IDENTIFICATION</p>
        <span class="login-version">v3.0 &nbsp;·&nbsp; Professional Edition</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            lang = st.selectbox("", ["🇮🇩 Bahasa Indonesia", "🇬🇧 English"],
                label_visibility="collapsed")
            is_en = "English" in lang

            username = st.text_input(
                "Username" if is_en else "Nama pengguna",
                placeholder="Enter username" if is_en else "Masukkan username"
            )
            password = st.text_input(
                "Password" if is_en else "Kata sandi",
                type="password",
                placeholder="Enter password" if is_en else "Masukkan kata sandi"
            )

            if st.button("Login", use_container_width=True):
                if not username or not password:
                    st.error("Please fill in all fields." if is_en else "Harap isi semua kolom.")
                else:
                    user = verify_login(username, password)
                    if user:
                        st.session_state["logged_in"]     = True
                        st.session_state["username"]      = username
                        st.session_state["role"]          = user["role"]
                        st.session_state["display_name"]  = user["name"]
                        st.session_state["lang"]          = "en" if is_en else "id"
                        st.rerun()
                    else:
                        st.error("Invalid username or password." if is_en else
                                 "Username atau kata sandi salah.")


