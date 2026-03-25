import hashlib
import json
import os
from datetime import datetime

_USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

_SUPERADMIN = {
    "username": "weerowoolf",
    "password_hash": "abfbf0b9fee161b9a2b9ccf1dea82c8aa774c35746c095976d3efc6b16331e65",
    "role": "superadmin",
    "name": "Andrew",
    "created_at": "2026-01-01",
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load() -> dict:
    if not os.path.exists(_USERS_FILE):
        data = {_SUPERADMIN["username"]: {k: v for k, v in _SUPERADMIN.items() if k != "username"}}
        _save(data)
        return data
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {_SUPERADMIN["username"]: {k: v for k, v in _SUPERADMIN.items() if k != "username"}}


def _save(data: dict) -> None:
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def authenticate(username: str, password: str) -> dict | None:
    users = _load()
    u = username.strip().lower()
    record = users.get(u)
    if record and record.get("password_hash") == _hash(password):
        return {"username": u, "role": record.get("role", "demo"), "name": record.get("name", u)}
    return None


def list_users() -> list[dict]:
    users = _load()
    return [
        {"username": u, "role": d.get("role", "demo"),
         "name": d.get("name", u), "created_at": d.get("created_at", "")}
        for u, d in users.items()
    ]


def add_user(username: str, password: str, name: str = "", role: str = "demo") -> bool:
    if not username or not password:
        return False
    users = _load()
    users[username.strip().lower()] = {
        "password_hash": _hash(password),
        "role": role,
        "name": name or username,
        "created_at": datetime.today().strftime("%Y-%m-%d"),
    }
    _save(users)
    return True


def remove_user(username: str) -> bool:
    users = _load()
    u = username.strip().lower()
    if u == "weerowoolf":
        return False
    if u in users:
        del users[u]
        _save(users)
        return True
    return False


def change_password(username: str, new_password: str) -> bool:
    users = _load()
    u = username.strip().lower()
    if u not in users:
        return False
    users[u]["password_hash"] = _hash(new_password)
    _save(users)
    return True
