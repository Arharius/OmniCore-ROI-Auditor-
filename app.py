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

if not authenticated and not demo_only:
    from ui.landing import show_landing
    show_landing()
    st.stop()

auth_user = st.session_state.get("auth_user", {})
is_admin  = auth_user.get("role") == "superadmin"

with st.sidebar:
    if demo_only and not authenticated:
        st.markdown("""
<div style="
    background:#1D1D1F;
    border-radius:10px;
    padding:10px 14px;
    margin-bottom:2px;
">
  <div style="font-size:11px;font-weight:600;letter-spacing:0.08em;
              text-transform:uppercase;color:rgba(255,255,255,0.45);
              margin-bottom:4px;">Demo mode</div>
  <div style="font-size:12px;color:rgba(255,255,255,0.7);line-height:1.4;">
    Sign in to upload CSV data<br>and generate PDF reports
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("Sign in", key="nav_login_demo", use_container_width=True):
            st.session_state.pop("demo_only", None)
            st.rerun()
    else:
        username_display = auth_user.get("username", "")
        col_u, col_out = st.columns([4, 1])
        col_u.markdown(
            f'<div style="padding:8px 0 4px;font-size:13px;font-weight:500;'
            f'color:#1D1D1F;letter-spacing:-0.01em;">@{username_display}</div>',
            unsafe_allow_html=True,
        )
        if col_out.button("↩", key="nav_logout", help="Sign out",
                          use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        if is_admin:
            st.session_state.setdefault("show_admin", False)
            lbl = "← Auditor" if st.session_state["show_admin"] else "Users"
            if st.button(lbl, key="nav_admin", use_container_width=True):
                st.session_state["show_admin"] = not st.session_state["show_admin"]
                st.rerun()

    st.markdown(
        "<hr style='margin:6px 0 2px;border:none;border-top:1px solid rgba(0,0,0,0.07);'>",
        unsafe_allow_html=True,
    )

if is_admin and st.session_state.get("show_admin"):
    from ui.admin import show_admin
    show_admin()
else:
    from ui.dashboard import run_dashboard
    run_dashboard()
