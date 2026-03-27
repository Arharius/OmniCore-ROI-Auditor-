"""
PostgreSQL persistence layer for OmniCore ROI Auditor.

Falls back to data/clients.json when DATABASE_URL is not set,
so the app works both locally and on Render without any code changes.
"""

import os
import json
from datetime import datetime
from contextlib import contextmanager

try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_FALLBACK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "clients.json"
)
_MAX_JSON_RECORDS = 15


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    url = DATABASE_URL
    # Render uses postgres:// scheme; psycopg2 needs postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # If the URL already contains sslmode (e.g. Replit local DB), honour it.
    # Otherwise (Render external URL) enforce SSL.
    kwargs = {} if "sslmode=" in url else {"sslmode": "require"}
    conn = psycopg2.connect(url, **kwargs)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_available() -> bool:
    """Return True if PostgreSQL is configured and reachable."""
    return bool(DATABASE_URL) and _PG_AVAILABLE


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db() -> bool:
    """Create tables if they don't exist. Safe to call on every startup."""
    if not db_available():
        return False
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS audit_history (
                        id                      SERIAL PRIMARY KEY,
                        company_name            VARCHAR(255) NOT NULL,
                        saved_at                TIMESTAMP DEFAULT NOW(),
                        params                  JSONB,
                        friction_tax_usd        FLOAT,
                        adjusted_confidence_pct FLOAT,
                        bottleneck_stage        VARCHAR(255),
                        roi_pct                 FLOAT,
                        rework_rate_pct         FLOAT,
                        total_transitions       INT,
                        total_rework            INT
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_company
                        ON audit_history(company_name);
                """)
        return True
    except Exception as e:
        print(f"[DB] init_db error: {e}")
        return False


# ── Load history ──────────────────────────────────────────────────────────────

def load_history() -> list:
    """Return list of audit records, newest first."""
    if db_available():
        try:
            with _get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[DB] load_history error: {e}")

    return _load_json()


# ── Save audit ────────────────────────────────────────────────────────────────

def save_audit(
    company_name: str,
    params: dict,
    friction_tax_usd: float | None = None,
    adjusted_confidence_pct: float | None = None,
    bottleneck_stage: str | None = None,
    roi_pct: float | None = None,
    rework_rate_pct: float | None = None,
    total_transitions: int | None = None,
    total_rework: int | None = None,
) -> bool:
    """Upsert an audit record by company name."""
    if db_available():
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO audit_history (
                            company_name, saved_at, params,
                            friction_tax_usd, adjusted_confidence_pct,
                            bottleneck_stage, roi_pct,
                            rework_rate_pct, total_transitions, total_rework
                        )
                        VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (company_name) DO UPDATE SET
                            saved_at                = NOW(),
                            params                  = EXCLUDED.params,
                            friction_tax_usd        = EXCLUDED.friction_tax_usd,
                            adjusted_confidence_pct = EXCLUDED.adjusted_confidence_pct,
                            bottleneck_stage        = EXCLUDED.bottleneck_stage,
                            roi_pct                 = EXCLUDED.roi_pct,
                            rework_rate_pct         = EXCLUDED.rework_rate_pct,
                            total_transitions       = EXCLUDED.total_transitions,
                            total_rework            = EXCLUDED.total_rework
                    """, (
                        company_name,
                        json.dumps(params, default=str),
                        friction_tax_usd,
                        adjusted_confidence_pct,
                        bottleneck_stage,
                        roi_pct,
                        rework_rate_pct,
                        total_transitions,
                        total_rework,
                    ))
            return True
        except Exception as e:
            print(f"[DB] save_audit error: {e}")

    return _save_json(company_name, params)


# ── Delete audit ──────────────────────────────────────────────────────────────

def delete_audit(company_name: str) -> bool:
    """Delete an audit record by company name."""
    if db_available():
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM audit_history WHERE company_name = %s",
                        (company_name,)
                    )
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


def _save_json(company_name: str, params: dict) -> bool:
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


def _delete_json(company_name: str) -> bool:
    try:
        history = _load_json()
        history = [h for h in history if h.get("company_name") != company_name]
        with open(_FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False
