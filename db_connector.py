"""
OmniCore Sovereign OS — PostgreSQL connector (psycopg2-binary).

Stores simplified audit snapshots in the `omnicore_audits` table on Neon.tech.
Falls back silently when DATABASE_URL is not set.
"""

import os
import psycopg2
from psycopg2 import sql

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS omnicore_audits (
    id               SERIAL PRIMARY KEY,
    client_name      VARCHAR(255),
    bottleneck_stage VARCHAR(255),
    friction_tax_usd NUMERIC,
    posterior_prob   NUMERIC,
    net_roi          NUMERIC,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_conn():
    """Open a psycopg2 connection to Neon with SSL enforced."""
    url = DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = url + sep + "sslmode=require"
    return psycopg2.connect(url)


def init_db() -> bool:
    """
    Create the omnicore_audits table if it does not exist.
    Safe to call on every startup — idempotent.
    Returns True on success, False if DB is unavailable.
    """
    if not DATABASE_URL:
        return False
    conn = None
    cur = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(_CREATE_TABLE_SQL)
        conn.commit()
        return True
    except Exception as e:
        print(f"[db_connector] init_db error: {e}")
        return False
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def save_audit_result(
    client_name: str,
    bottleneck: str,
    tax: float,
    prob: float,
    roi: float,
) -> bool:
    """
    Insert one audit record into omnicore_audits.

    Args:
        client_name: Company / client name entered by the user.
        bottleneck:  Bottleneck stage label from Markov analysis.
        tax:         Friction tax in USD.
        prob:        Bayesian posterior probability (0–100 scale).
        roi:         Net ROI value in the project currency.

    Returns True on success, False otherwise.
    """
    if not DATABASE_URL:
        return False
    conn = None
    cur = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO omnicore_audits
                (client_name, bottleneck_stage, friction_tax_usd, posterior_prob, net_roi)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (client_name, bottleneck, float(tax), float(prob), float(roi)),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[db_connector] save_audit_result error: {e}")
        return False
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
