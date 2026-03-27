"""
PostgreSQL persistence layer for OmniCore ROI Auditor.

Uses pg8000 (pure-Python, no libpq required) so the app builds
on any platform including Render free tier.

Falls back to data/clients.json when DATABASE_URL is not set.
"""

import os
import json
import urllib.parse
from datetime import datetime
from contextlib import contextmanager

try:
    import pg8000.dbapi as pg8000
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_FALLBACK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "clients.json"
)
_MAX_JSON_RECORDS = 15


# ── URL parser ────────────────────────────────────────────────────────────────

def _parse_url(url: str) -> dict:
    """Parse a postgres(ql):// URL into pg8000.connect() kwargs."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    p = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(p.query)
    ssl_mode = qs.get("sslmode", [""])[0]
    host = p.hostname or "localhost"
    kwargs = {
        "host":     host,
        "port":     p.port    or 5432,
        "database": p.path.lstrip("/"),
        "user":     p.username or "",
        "password": p.password or "",
    }
    # Determine SSL need:
    # - explicit sslmode=disable/disab → no SSL
    # - Render internal hostname (no dot, e.g. dpg-xxx-a) → no SSL needed
    # - everything else (external or unknown) → enforce SSL
    _internal = "." not in host
    if ssl_mode in ("disable", "disab") or _internal:
        pass  # no SSL
    else:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
    return kwargs


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    conn = pg8000.connect(**_parse_url(DATABASE_URL))
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_available() -> bool:
    """Return True if PostgreSQL is configured and the library is available."""
    return bool(DATABASE_URL) and _PG_AVAILABLE


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db() -> bool:
    """Create tables if they don't exist. Safe to call on every startup."""
    if not db_available():
        return False
    try:
        with _get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_history (
                    id                      SERIAL PRIMARY KEY,
                    company_name            VARCHAR(255) NOT NULL,
                    saved_at                TIMESTAMP DEFAULT NOW(),
                    params                  TEXT,
                    friction_tax_usd        DOUBLE PRECISION,
                    adjusted_confidence_pct DOUBLE PRECISION,
                    bottleneck_stage        VARCHAR(255),
                    roi_pct                 DOUBLE PRECISION,
                    rework_rate_pct         DOUBLE PRECISION,
                    total_transitions       INTEGER,
                    total_rework            INTEGER
                )
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_company
                    ON audit_history(company_name)
            """)
            cur.close()
        return True
    except Exception as e:
        print(f"[DB] init_db error: {e}")
        return False


# ── Load history ──────────────────────────────────────────────────────────────

def load_history() -> list:
    """Return list of audit records, newest first (max 50)."""
    if db_available():
        try:
            with _get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT
                        company_name,
                        to_char(saved_at, 'YYYY-MM-DD HH24:MI') AS saved_at,
                        params,
                        friction_tax_usd,
                        adjusted_confidence_pct,
                        bottleneck_stage,
                        roi_pct,
                        rework_rate_pct,
                        total_transitions,
                        total_rework
                    FROM audit_history
                    ORDER BY saved_at DESC
                    LIMIT 50
                """)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                cur.close()
                # params stored as JSON text → deserialise
                for row in rows:
                    if isinstance(row.get("params"), str):
                        try:
                            row["params"] = json.loads(row["params"])
                        except Exception:
                            row["params"] = {}
                return rows
        except Exception as e:
            print(f"[DB] load_history error: {e}")

    return _load_json()


# ── Save audit ────────────────────────────────────────────────────────────────

def save_audit(
    company_name,
    params,
    friction_tax_usd=None,
    adjusted_confidence_pct=None,
    bottleneck_stage=None,
    roi_pct=None,
    rework_rate_pct=None,
    total_transitions=None,
    total_rework=None,
):
    """Upsert an audit record by company name."""
    if db_available():
        try:
            params_json = json.dumps(params, default=str)
            with _get_conn() as conn:
                cur = conn.cursor()
                # Delete existing record for this company, then insert fresh
                cur.execute(
                    "DELETE FROM audit_history WHERE company_name = %s",
                    (company_name,)
                )
                cur.execute(
                    """
                    INSERT INTO audit_history (
                        company_name, saved_at, params,
                        friction_tax_usd, adjusted_confidence_pct,
                        bottleneck_stage, roi_pct,
                        rework_rate_pct, total_transitions, total_rework
                    ) VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        company_name,
                        params_json,
                        friction_tax_usd,
                        adjusted_confidence_pct,
                        bottleneck_stage,
                        roi_pct,
                        rework_rate_pct,
                        total_transitions,
                        total_rework,
                    )
                )
                cur.close()
            return True
        except Exception as e:
            print(f"[DB] save_audit error: {e}")

    return _save_json(company_name, params)


# ── Delete audit ──────────────────────────────────────────────────────────────

def delete_audit(company_name):
    """Delete an audit record by company name."""
    if db_available():
        try:
            with _get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM audit_history WHERE company_name = %s",
                    (company_name,)
                )
                cur.close()
            return True
        except Exception as e:
            print(f"[DB] delete_audit error: {e}")

    return _delete_json(company_name)


# ── JSON fallback helpers ─────────────────────────────────────────────────────

def _load_json() -> list:
    try:
        if os.path.exists(_FALLBACK_FILE):
            with open(_FALLBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_json(company_name, params) -> bool:
    try:
        history = _load_json()
        history = [h for h in history if h.get("company_name") != company_name]
        history.insert(0, {
            "company_name": company_name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "params": params,
        })
        history = history[:_MAX_JSON_RECORDS]
        os.makedirs(os.path.dirname(_FALLBACK_FILE), exist_ok=True)
        with open(_FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _delete_json(company_name) -> bool:
    try:
        history = _load_json()
        history = [h for h in history if h.get("company_name") != company_name]
        with open(_FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False
