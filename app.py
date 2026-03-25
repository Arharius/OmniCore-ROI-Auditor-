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

# ── 1. Check authentication ────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    from ui.landing import show_landing
    show_landing()
    st.stop()

# ── 2. Authenticated — inject logout + admin toggle into sidebar ───────────────
auth_user = st.session_state.get("auth_user", {})
is_admin  = auth_user.get("role") == "superadmin"

# sidebar logout / admin nav (rendered before dashboard imports its own sidebar)
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;'
        f'padding:10px 0 4px;">'
        f'<div style="background:#0071E3;color:#fff;font-size:10px;'
        f'font-weight:700;padding:2px 8px;border-radius:980px;letter-spacing:0.05em;">'
        f'{"ADMIN" if is_admin else "DEMO"}</div>'
        f'<span style="font-size:13px;color:#1D1D1F;font-weight:600;">'
        f'@{auth_user.get("username","")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _nav_cols = st.columns(2) if is_admin else [None]
    if is_admin:
        if "show_admin" not in st.session_state:
            st.session_state["show_admin"] = False
        lbl = "← Аудитор" if st.session_state["show_admin"] else "⚙️ Пользователи"
        if _nav_cols[0].button(lbl, key="nav_admin", use_container_width=True):
            st.session_state["show_admin"] = not st.session_state.get("show_admin", False)
            st.rerun()
        if _nav_cols[1].button("Выйти", key="nav_logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    else:
        if st.button("Выйти", key="nav_logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    st.markdown("<hr style='margin:8px 0 4px;border-color:rgba(0,0,0,0.08);'>",
                unsafe_allow_html=True)

# ── 3. Route: admin panel or main dashboard ────────────────────────────────────
if is_admin and st.session_state.get("show_admin"):
    from ui.admin import show_admin
    show_admin()
else:
    from ui.dashboard import run_dashboard
    run_dashboard()
