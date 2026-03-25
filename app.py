import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="OmniCore ROI Auditor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

authenticated = st.session_state.get("authenticated")
demo_only     = st.session_state.get("demo_only")

# ── Show landing if neither authenticated nor in demo ─────────────────────────
if not authenticated and not demo_only:
    from ui.landing import show_landing
    show_landing()
    st.stop()

# ── Sidebar controls ───────────────────────────────────────────────────────────
auth_user = st.session_state.get("auth_user", {})
is_admin  = auth_user.get("role") == "superadmin"

with st.sidebar:
    if demo_only and not authenticated:
        # Compact "Sign in" bar
        if st.button("🔐 Войти в полную версию", key="nav_login_demo",
                     use_container_width=True):
            st.session_state.pop("demo_only", None)
            st.rerun()
    else:
        # Compact user row
        col_u, col_out = st.columns([3, 1])
        col_u.markdown(
            f'<div style="padding:6px 0;font-size:13px;color:#6E6E73;">'
            f'@{auth_user.get("username","")}</div>',
            unsafe_allow_html=True,
        )
        if col_out.button("↩", key="nav_logout", help="Выйти",
                          use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if is_admin:
            if "show_admin" not in st.session_state:
                st.session_state["show_admin"] = False
            lbl = "← Аудитор" if st.session_state["show_admin"] else "⚙️ Пользователи"
            if st.button(lbl, key="nav_admin", use_container_width=True):
                st.session_state["show_admin"] = not st.session_state["show_admin"]
                st.rerun()

    st.markdown("<hr style='margin:4px 0 2px;border-color:rgba(0,0,0,0.07);'>",
                unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────────
if is_admin and st.session_state.get("show_admin"):
    from ui.admin import show_admin
    show_admin()
else:
    from ui.dashboard import run_dashboard
    run_dashboard()
