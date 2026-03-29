import hmac
import hashlib
import time
import os
from datetime import datetime, timedelta, timezone

import streamlit as st

COOKIE_NAME = "omnicore_auth"
_SECRET     = os.environ.get("COOKIE_SECRET", "omnicore-secret-2025-xK9")
_EXP_DAYS   = 7


def _sign(payload: str) -> str:
    return hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(username: str, role: str) -> str:
    ts      = int(time.time())
    payload = f"{username}:{role}:{ts}"
    return f"{payload}:{_sign(payload)}"


def verify_token(token: str):
    """Returns dict with username/role or None if invalid / expired."""
    try:
        *payload_parts, sig = token.split(":")
        payload = ":".join(payload_parts)
        if not hmac.compare_digest(_sign(payload), sig):
            return None
        username, role, ts = payload.split(":", 2)
        if time.time() - int(ts) > _EXP_DAYS * 86400:
            return None
        return {"username": username, "role": role}
    except Exception:
        return None


def get_cookie_manager():
    import extra_streamlit_components as stx
    return stx.CookieManager(key="omnicore_cm")


def restore_session(cookie_manager) -> bool:
    """
    Read the auth cookie and restore session_state if the token is valid.
    Returns True if session was restored.
    """
    if st.session_state.get("authenticated"):
        return True
    try:
        all_cookies = cookie_manager.get_all()
        token = (all_cookies or {}).get(COOKIE_NAME, "")
    except Exception:
        return False
    if not token:
        return False
    user = verify_token(str(token))
    if not user:
        return False
    st.session_state["authenticated"] = True
    st.session_state["auth_user"]     = user
    return True


def set_auth_cookie(cookie_manager, username: str, role: str) -> None:
    token   = make_token(username, role)
    expires = datetime.now(timezone.utc) + timedelta(days=_EXP_DAYS)
    try:
        cookie_manager.set(COOKIE_NAME, token, expires_at=expires, key="set_omnicore_auth")
    except Exception:
        pass


def clear_auth_cookie(cookie_manager) -> None:
    try:
        cookie_manager.delete(COOKIE_NAME, key="del_omnicore_auth")
    except Exception:
        pass
