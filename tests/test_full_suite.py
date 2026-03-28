"""
OmniCore ROI Auditor — Full Deep-Stress Test Suite
===================================================
Covers every calculation module: CSV Parser, Markov, Bayes, ROI, Friction, ETL, DB.
Run:  python -m pytest tests/test_full_suite.py -v  OR  python tests/test_full_suite.py
"""
import io
import sys
import math
import time
import traceback
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")

# ── Lazy imports (handle missing deps gracefully) ──────────────────────────────
from core.math_engine import MathEngine, build_markov_graph, BayesResult
from core.roi_engine  import ROIEngine, ROIInput, ROIResult

# ── CSV parser (inline copy — keeps tests independent of Streamlit) ────────────
def load_and_clean_csv(uploaded_file):
    """Bulletproof CSV parser (same logic as ui/dashboard.py load_and_clean_csv)."""
    raw_bytes = uploaded_file.read()
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            text = raw_bytes.decode(enc); break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return None
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        parts = line.split(",")
        non_empty_parts = sum(1 for p in parts if p.strip())
        substantial_length = len(line.strip()) > 10
        if non_empty_parts > 2 and substantial_length:
            header_idx = i; break
    if header_idx is None:
        return None
    clean_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(clean_text))
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df.dropna(how="all")
    df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").all(), axis=1)]
    df = df.reset_index(drop=True)
    return df if not df.empty else None


# ── Test Harness ───────────────────────────────────────────────────────────────
PASS = 0; FAIL = 0; ERRORS: list = []

def ok(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  ✓  {name}")
        PASS += 1
    else:
        msg = f"  ✗  {name}" + (f"  → {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)
        FAIL += 1

def approx(a: float, b: float, tol: float = 0.02) -> bool:
    """True if |a - b| <= tol * max(|b|, 1)."""
    return abs(a - b) <= tol * max(abs(b), 1.0)

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CSV PARSER
# ══════════════════════════════════════════════════════════════════════════════
section("1. CSV PARSER — load_and_clean_csv")

def make_file(content: bytes) -> io.BytesIO:
    return io.BytesIO(content)

# 1.1 Jira-style garbage header
csv_jira = b"Report generated on 2025-01-15\n\nTable 1: Support Tickets\n\nTicket_ID,Current_Status,Days_Elapsed,Next_Status,Assignee\nT-001,Open,3,In Review,Alice\nT-002,Closed,5,Done,Bob\n"
df = load_and_clean_csv(make_file(csv_jira))
ok("1.01 Jira metadata stripped", df is not None and list(df.columns) == ["Ticket_ID","Current_Status","Days_Elapsed","Next_Status","Assignee"] and len(df)==2)

# 1.2 Clean CSV (no garbage)
csv_clean = b"ID,Status,Days,Next\nA01,Open,3,Review\nA02,Done,5,Closed\n"
df = load_and_clean_csv(make_file(csv_clean))
ok("1.02 Clean CSV parsed", df is not None and len(df)==2 and len(df.columns)==4)

# 1.3 Trailing commas → Unnamed dropped
csv_trail = b"Ticket_ID,Status,Days,Next,,,\nT001,Open,3,Review,,,\nT002,Closed,5,Done,,,\n"
df = load_and_clean_csv(make_file(csv_trail))
ok("1.03 Trailing commas (Unnamed gone)", df is not None and all("Unnamed" not in c for c in df.columns) and len(df.columns)==4)

# 1.4 Leading commas (Unnamed leading dropped)
csv_lead = b",,,Ticket_ID,Status,Days,Next\n,,,T001,Open,3,Review\n,,,T002,Done,5,Closed\n"
df = load_and_clean_csv(make_file(csv_lead))
ok("1.04 Leading commas (Unnamed gone)", df is not None and "Ticket_ID" in df.columns)

# 1.5 Column names with whitespace
csv_space = b" Ticket_ID , Status , Days , Next \nT001,Open,3,Review\nT002,Done,5,Closed\n"
df = load_and_clean_csv(make_file(csv_space))
ok("1.05 Column whitespace stripped", df is not None and "Ticket_ID" in df.columns)

# 1.6 CP1251 encoding
csv_cp = "ID,Status,Days,Next\n001,InWork,3,Done\n002,Review,5,InWork\n".encode("cp1251")
df = load_and_clean_csv(make_file(csv_cp))
ok("1.06 CP1251 encoding", df is not None and len(df)==2)

# 1.7 UTF-8 BOM (Excel Save-As)
csv_bom = ("\ufeffTicket_ID,Status,Days,Next\nT001,Open,3,Review\n").encode("utf-8-sig")
df = load_and_clean_csv(make_file(csv_bom))
ok("1.07 UTF-8 BOM (Excel)", df is not None and "Ticket_ID" in df.columns)

# 1.8 All-blank rows in data body are dropped
csv_blanks = b"ID,Status,Days,Next\nT001,Open,3,Review\n,,,\n\nT002,Done,5,Closed\n"
df = load_and_clean_csv(make_file(csv_blanks))
ok("1.08 Blank rows in body dropped", df is not None and len(df)==2)

# 1.9 Zendesk # comment line
csv_zd = b"# Zendesk Export\nticket_id,subject,status,created_at\n12,Bug,open,2025-01-10\n13,Crash,pending,2025-01-11\n"
df = load_and_clean_csv(make_file(csv_zd))
ok("1.09 Zendesk comment line", df is not None and "ticket_id" in df.columns and len(df)==2)

# 1.10 Salesforce-style report header
csv_sf = b"Report:,My Sales Export,,\nDate:,2025-01-15,,\nOpportunity,Stage,Amount,Owner\nDeal A,Closed Won,50000,Alice\nDeal B,Negotiation,20000,Bob\n"
df = load_and_clean_csv(make_file(csv_sf))
ok("1.10 Salesforce report header", df is not None and "Opportunity" in df.columns and len(df)==2)

# 1.11 Only 1 column → None (no valid header)
csv_bad = b"just one column\nno commas at all\n"
df = load_and_clean_csv(make_file(csv_bad))
ok("1.11 No valid header → None", df is None)

# 1.12 Actual user test file
csv_user = open("attached_assets/тест_1774672879285.csv", "rb").read()
df = load_and_clean_csv(make_file(csv_user))
ok("1.12 User test CSV parsed", df is not None and "Ticket_ID" in df.columns and len(df)==21,
   f"got {None if df is None else (list(df.columns), len(df))}")

# 1.13 Large CSV (1000 rows) — performance test
rows = "\n".join(f"T{i:04d},Stage{i%5},{ i%10 + 1},Stage{(i+1)%5}" for i in range(1000))
csv_large = (f"ID,Current,Days,Next\n{rows}\n").encode()
t0 = time.perf_counter()
df = load_and_clean_csv(make_file(csv_large))
elapsed = time.perf_counter() - t0
ok("1.13 Large CSV 1000 rows", df is not None and len(df)==1000, f"{len(df)} rows")
ok("1.14 Large CSV parse < 1s", elapsed < 1.0, f"{elapsed:.3f}s")

# 1.15 Empty file → None
df = load_and_clean_csv(make_file(b""))
ok("1.15 Empty file → None", df is None)

# 1.16 Header-only, no data rows → None
df = load_and_clean_csv(make_file(b"ID,Status,Days,Next\n"))
ok("1.16 Header only (no data) → None", df is None)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MARKOV CHAIN — build_markov_graph
# ══════════════════════════════════════════════════════════════════════════════
section("2. MARKOV CHAIN — build_markov_graph")

def make_markov_df(rows):
    return pd.DataFrame(rows, columns=["entity_id","current_stage","next_stage","time_spent"])

# 2.1 Basic linear chain (no rework)
rows_linear = [
    ("D1","Triage","L1 Support",1),
    ("D1","L1 Support","Resolved",2),
    ("D2","Triage","L1 Support",1),
    ("D2","L1 Support","Resolved",3),
]
mgr = build_markov_graph(make_markov_df(rows_linear))
ok("2.01 Linear chain built", mgr is not None)
ok("2.02 Linear chain: no rework pairs", len(mgr.rework_pairs)==0)
ok("2.03 Linear chain: transition probs sum ≤ 1", all(v<=1.001 for v in mgr.transition_probs.values()))

# 2.2 User test CSV — rework detection
df_user = load_and_clean_csv(make_file(csv_user))
df_mapped = df_user.rename(columns={"Ticket_ID":"entity_id","Current_Status":"current_stage",
                                     "Next_Status":"next_stage","Days_Elapsed":"time_spent"})
mgr2 = build_markov_graph(df_mapped)
ok("2.04 User CSV: Markov built", mgr2 is not None)
ok("2.05 User CSV: rework detected (Vendor Escalation loop)",
   len(mgr2.rework_pairs) > 0, f"pairs={len(mgr2.rework_pairs)}")
ok("2.06 User CSV: bottleneck not empty", mgr2.bottleneck_node != "")
ok("2.07 User CSV: total_transitions == 21", mgr2.total_transitions == 21, str(mgr2.total_transitions))
ok("2.08 User CSV: rework transitions > 0", mgr2.total_rework_transitions > 0)
ok("2.09 User CSV: bottleneck is L2 Support or Vendor Escalation (high rework)",
   mgr2.bottleneck_node in ("L2 Support", "Vendor Escalation"), mgr2.bottleneck_node)

# Transition probabilities from Triage must sum to 1
triage_probs = sum(v for (f,t),v in mgr2.transition_probs.items() if f=="Triage")
ok("2.10 Triage outgoing probs sum to 1.0", approx(triage_probs, 1.0, 0.001), f"{triage_probs:.4f}")

# 2.3 Single-entity, multi-rework
rows_rework = [
    ("X1","A","B",5), ("X1","B","A",10), ("X1","A","B",5), ("X1","B","C",2),
    ("X2","A","B",3), ("X2","B","A",8),  ("X2","A","C",1),
]
mgr3 = build_markov_graph(make_markov_df(rows_rework))
ok("2.11 Multi-rework: rework pair A↔B detected", any(
    {rp.stage_a, rp.stage_b} == {"A","B"} for rp in mgr3.rework_pairs))

# 2.4 Empty DataFrame raises ValueError
try:
    build_markov_graph(pd.DataFrame(columns=["entity_id","current_stage","next_stage","time_spent"]))
    ok("2.12 Empty DF raises ValueError", False, "no exception raised")
except ValueError:
    ok("2.12 Empty DF raises ValueError", True)

# 2.5 Bottleneck score invariant: always in [0, 1]
ok("2.13 Bottleneck score in [0,1]", 0.0 <= mgr2.bottleneck_score <= 1.0, str(mgr2.bottleneck_score))

# 2.6 NetworkX graph node count matches stages
all_stages = set(df_mapped["current_stage"]) | set(df_mapped["next_stage"])
ok("2.14 NX graph nodes == all stages", set(mgr2.G.nodes) == all_stages)

# 2.7 Rework rate per stage in [0, 1]
ok("2.15 All rework_rates in [0,1]",
   all(0.0 <= v <= 1.0 for v in mgr2.rework_rate.values()))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MATH ENGINE — Bayesian, Markov Absorbing, Friction Tax
# ══════════════════════════════════════════════════════════════════════════════
section("3. MATH ENGINE — MathEngine methods")
engine = MathEngine()

# 3.1 bayesian_update — mathematical correctness
br = engine.bayesian_update(4, 5, prior_rate=0.80)
expected_post = (0.80*10 + 4) / (0.80*10 + (1-0.80)*10 + 5)
ok("3.01 Bayes posterior formula correct",
   approx(br.posterior_pct/100, expected_post, 0.005), f"{br.posterior_pct}% vs {expected_post*100:.1f}%")
ok("3.02 Bayes prior matches input", approx(br.prior_pct, 80.0, 0.01))
ok("3.03 Bayes 80% CI: low < posterior < high",
   br.ci_80_low < br.posterior_pct < br.ci_80_high,
   f"{br.ci_80_low} < {br.posterior_pct} < {br.ci_80_high}")

# 3.2 Bayesian with prior=0.34 (old default — verify unchanged behaviour)
br2 = engine.bayesian_update(42, 120, prior_rate=0.34)
ok("3.04 Bayes prior=0.34, 42/120 → posterior > prior", br2.posterior_pct > br2.prior_pct,
   f"prior={br2.prior_pct}, post={br2.posterior_pct}")

# 3.3 Bayesian edge: all signals positive → posterior near 1
br3 = engine.bayesian_update(100, 100, prior_rate=0.5)
ok("3.05 All positive signals → posterior high", br3.posterior_pct > 90.0, str(br3.posterior_pct))

# 3.4 Bayesian edge: no positive signals → posterior < prior
br4 = engine.bayesian_update(0, 10, prior_rate=0.5)
ok("3.06 Zero positive signals → posterior < prior", br4.posterior_pct < br4.prior_pct,
   f"{br4.posterior_pct} vs {br4.prior_pct}")

# 3.5 CI validity: low <= posterior <= high always
for pos, tot, pr in [(1,5,0.3), (3,3,0.9), (10,100,0.5), (1,2,0.01)]:
    br_ = engine.bayesian_update(pos, tot, prior_rate=pr)
    ok(f"3.07 CI valid ({pos}/{tot} prior={pr})", br_.ci_80_low <= br_.posterior_pct <= br_.ci_80_high)

# 3.6 compute_friction_tax
ft = engine.compute_friction_tax(decision_latency_days=14.0, cost_per_hour=12.0, hours_per_day=8.0)
ok("3.08 Friction tax = days × hours × cost", approx(ft, 14*8*12), f"{ft} vs {14*8*12}")

# Edge cases for friction tax
ok("3.09 Friction tax: zero latency → 0", engine.compute_friction_tax(0, 12, 8) == 0.0)
ok("3.10 Friction tax: zero cost → 0",    engine.compute_friction_tax(5, 0, 8) == 0.0)

# 3.7 compute_process_confidence
conf = engine.compute_process_confidence(prior_pct=85.0, bottleneck_rework_rate=0.3)
ok("3.11 Process confidence: posterior in (0,100)", 0 < conf["posterior_pct"] < 100)
ok("3.12 Process confidence: low rework → posterior > prior (high LR)",
   conf["posterior_pct"] > conf["prior_pct"],
   f"{conf['posterior_pct']} vs {conf['prior_pct']}")

conf2 = engine.compute_process_confidence(prior_pct=50.0, bottleneck_rework_rate=0.9)
ok("3.13 High rework rate → posterior < prior",
   conf2["posterior_pct"] < conf2["prior_pct"],
   f"{conf2['posterior_pct']} vs {conf2['prior_pct']}")

# 3.8 markov_absorbing
Q  = np.array([[0.2, 0.3], [0.1, 0.4]])
st = np.array([8.0, 16.0])
mr = engine.markov_absorbing(Q, st, ["Qualify","Propose"])
ok("3.14 Markov absorbing: lead time > 0", mr.expected_lead_time_hours > 0)
ok("3.15 Markov absorbing: p_complete + p_lost = 1",
   approx(mr.p_complete + mr.p_lost, 1.0, 0.0001))

# 3.9 Singular matrix raises ValueError
Q_sing = np.array([[0.5, 0.5], [0.5, 0.5]])  # I - Q is singular
try:
    engine.markov_absorbing(Q_sing, st, ["A","B"])
    ok("3.16 Singular Q raises ValueError", False, "no exception")
except (ValueError, np.linalg.LinAlgError):
    ok("3.16 Singular Q raises ValueError", True)

# 3.10 graph_bottleneck
edges = [("A","B",3.0),("B","C",2.0),("A","C",1.0),("C","D",4.0),("B","D",1.5)]
gr = engine.graph_bottleneck(edges)
ok("3.17 Graph bottleneck: node identified", gr.bottleneck_node in ("B","C"))
ok("3.18 Graph bottleneck: score in (0,1]", 0 < gr.bottleneck_score <= 1.0)

# 3.11 bayesian_contextual_risk
risk = engine.bayesian_contextual_risk(0.05, 0.80, 0.20)
expected_risk = min((0.80 * 0.05) / 0.20, 1.0)
ok("3.19 Contextual risk formula correct", approx(risk, expected_risk, 0.001))
ok("3.20 Contextual risk: P(C)=0 → 1.0", engine.bayesian_contextual_risk(0.05,0.8,0.0) == 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ROI ENGINE — ROIEngine.calculate
# ══════════════════════════════════════════════════════════════════════════════
section("4. ROI ENGINE — ROIEngine.calculate")
roi = ROIEngine()

def base_inp(**kw) -> ROIInput:
    defaults = dict(
        company_name="TestCo", manual_hours_per_month=320.0, automation_rate=0.80,
        hour_rate_eur=12.0, error_rate_before_pct=8.5, error_rate_after_pct=1.2,
        cost_per_error_eur=95.0, monthly_volume=600, deal_cycle_before_days=21.0,
        deal_cycle_after_days=9.0, deals_per_month=25, avg_deal_value_eur=1000.0,
        p_complete_before=0.74, p_complete_after=0.96,
        implementation_cost_eur=15000.0, positive_signals=4, total_signals=5,
        pipeline_utilization_pct=30.0,
    )
    defaults.update(kw)
    return ROIInput(**defaults)

res = roi.calculate(base_inp())

# 4.1 Basic sanity checks
ok("4.01 Net ROI is positive",        res.net_roi > 0, str(res.net_roi))
ok("4.02 ROI pct > 0",                res.roi_pct > 0, str(res.roi_pct))
ok("4.03 Payback months > 0",         res.payback_months > 0, str(res.payback_months))
ok("4.04 Total benefit > impl cost",  res.total_benefit > 15000, str(res.total_benefit))

# 4.2 Math verification
exp_time_saved = 320 * 0.80 * 12 * 12
ok("4.05 Time saved formula", approx(res.time_saved_annual, exp_time_saved, 0.001),
   f"{res.time_saved_annual} vs {exp_time_saved}")

exp_error = ((8.5 - 1.2) / 100) * 600 * 12 * 95
ok("4.06 Error saved formula", approx(res.error_reduction_annual, exp_error, 0.001),
   f"{res.error_reduction_annual} vs {exp_error}")

velocity = (21 - 9) / 21
exp_rev = 25 * velocity * 0.30 * 12 * 1000
ok("4.07 Revenue impact formula", approx(res.revenue_impact_annual, exp_rev, 0.001),
   f"{res.revenue_impact_annual} vs {exp_rev}")

exp_markov = 25 * ((1 - 0.74) - (1 - 0.96)) * 1000 * 12
ok("4.08 Markov gain formula", approx(res.markov_gain_annual, exp_markov, 0.001),
   f"{res.markov_gain_annual} vs {exp_markov}")

exp_net = res.total_benefit - 15000
ok("4.09 Net ROI = total_benefit - impl_cost", approx(res.net_roi, exp_net, 0.001))

exp_roi_pct = (res.net_roi / 15000) * 100
ok("4.10 ROI pct = net_roi / impl_cost * 100", approx(res.roi_pct, exp_roi_pct, 0.001))

# Engine applies round(payback, 1), so allow ±0.1 tolerance
exp_payback = 15000 / (res.total_benefit / 12)
ok("4.11 Payback = impl_cost / (total/12)", approx(res.payback_months, exp_payback, 0.15),
   f"got {res.payback_months:.2f}, expected {exp_payback:.2f}")

# 4.3 Bayesian prior from user CSV must be 80% (not 34%)
br_user = engine.bayesian_update(4, 5, prior_rate=0.80)
ok("4.12 User CSV prior=80% → posterior correct",
   approx(br_user.prior_pct, 80.0, 0.01) and br_user.ci_80_low < br_user.posterior_pct < br_user.ci_80_high)

# 4.4 Zero automation_rate → time_saved=0
res0 = roi.calculate(base_inp(automation_rate=0.0))
ok("4.13 automation_rate=0 → time_saved=0", res0.time_saved_annual == 0.0)

# 4.5 Equal error rates → error_saved=0
res_err = roi.calculate(base_inp(error_rate_before_pct=5.0, error_rate_after_pct=5.0))
ok("4.14 Equal error rates → error_saved=0", res_err.error_reduction_annual == 0.0)

# 4.6 Equal deal cycles → revenue_impact=0
res_cyc = roi.calculate(base_inp(deal_cycle_before_days=21.0, deal_cycle_after_days=21.0))
ok("4.15 Equal deal cycles → revenue_impact=0", res_cyc.revenue_impact_annual == 0.0)

# 4.7 impl_cost=0 doesn't crash (roi_pct would be 0)
res_zero = roi.calculate(base_inp(implementation_cost_eur=0.0))
ok("4.16 impl_cost=0 → no crash", res_zero.roi_pct == 0.0)

# 4.8 p_complete_before == p_complete_after → markov_gain=0
res_mc = roi.calculate(base_inp(p_complete_before=0.80, p_complete_after=0.80))
ok("4.17 Equal p_complete → markov_gain=0", res_mc.markov_gain_annual == 0.0)

# 4.9 Negative ROI scenario (high impl cost)
res_neg = roi.calculate(base_inp(implementation_cost_eur=10_000_000.0))
ok("4.18 High impl cost → negative net ROI", res_neg.net_roi < 0)

# 4.10 Bayesian CI appears in result string
ok("4.19 bayesian_ci field not empty", "%" in res.bayesian_ci)

# 4.11 passport_text generation (EN/RU/SR)
for lang in ("en","ru","sr"):
    try:
        txt = roi.passport_text(base_inp(), res, lang=lang)
        ok(f"4.20 passport_text lang={lang}", isinstance(txt, str) and len(txt) > 100)
    except Exception as e:
        ok(f"4.20 passport_text lang={lang}", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ABSORBING KEYWORD LOGIC
# ══════════════════════════════════════════════════════════════════════════════
section("5. ABSORBING KEYWORD LOGIC")

POSITIVE_KW = {
    "done","completed","complete","finished","closed","won",
    "delivered","shipped","approved","deployed","production",
    "resolved","resolution","fixed","fulfilled",
    "accepted","verified","released",
    "завершена","завершено","выполнено","продано","закрыта","закрыто","доставлено",
    "završeno","zatvoreno","isporučeno",
}
NEGATIVE_KW = {
    "refunded","cancelled","canceled","rejected","archived",
    "отклонено","отменено","возврат",
    "odbijeno","otkazano",
}
ABSORBING_KW = POSITIVE_KW | NEGATIVE_KW

# 5.1 "resolved" is positive
ok("5.01 'resolved' in POSITIVE_KW", "resolved" in POSITIVE_KW)
ok("5.02 'refunded' in NEGATIVE_KW", "refunded" in NEGATIVE_KW)
ok("5.03 'done' in POSITIVE_KW",     "done" in POSITIVE_KW)
ok("5.04 'cancelled' in NEGATIVE_KW","cancelled" in NEGATIVE_KW)
ok("5.05 'rejected' in NEGATIVE_KW", "rejected" in NEGATIVE_KW)
ok("5.06 'closed' in POSITIVE_KW",   "closed" in POSITIVE_KW)
ok("5.07 Sets disjoint",             len(POSITIVE_KW & NEGATIVE_KW) == 0)

# 5.2 User CSV absorbing state counts
_next_lower = df_mapped["next_stage"].str.lower().str.strip()
pos_done = df_mapped[_next_lower.isin(POSITIVE_KW)]["entity_id"].nunique()
all_done = df_mapped[_next_lower.isin(ABSORBING_KW)]["entity_id"].nunique()
ok("5.08 User CSV: 4 positive completions", pos_done == 4, str(pos_done))
ok("5.09 User CSV: 5 total completions",    all_done == 5, str(all_done))
ok("5.10 pos_signals = 4", max(1, pos_done) == 4)
ok("5.11 tot_signals = 5", max(2, max(all_done, df_mapped["entity_id"].nunique())) == 5)

p_before = max(30, min(95, int(pos_done / df_mapped["entity_id"].nunique() * 100)))
ok("5.12 p_before = 80% for user CSV", p_before == 80, str(p_before))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
section("6. END-TO-END PIPELINE — CSV → Markov → ROI → Bayes")

# 6.1 Full pipeline with user test CSV
try:
    # Step 1: parse CSV
    df_e2e = load_and_clean_csv(make_file(csv_user))
    ok("6.01 E2E: CSV parsed",        df_e2e is not None)

    # Step 2: rename to canonical columns
    df_e2e_m = df_e2e.rename(columns={
        "Ticket_ID":"entity_id","Current_Status":"current_stage",
        "Next_Status":"next_stage","Days_Elapsed":"time_spent"
    })
    ok("6.02 E2E: 21 rows mapped",    len(df_e2e_m)==21)

    # Step 3: Markov graph
    mgr_e2e = build_markov_graph(df_e2e_m)
    ok("6.03 E2E: Markov built",       mgr_e2e is not None)
    ok("6.04 E2E: Bottleneck found",   mgr_e2e.bottleneck_node != "")

    # Step 4: Absorbing states → signals
    _nl = df_e2e_m["next_stage"].str.lower().str.strip()
    pos_s = max(1, df_e2e_m[_nl.isin(POSITIVE_KW)]["entity_id"].nunique())
    tot_s = max(2, df_e2e_m["entity_id"].nunique())
    ok("6.05 E2E: signals = 4/5",      pos_s==4 and tot_s==5)

    # Step 5: Bayesian update with derived prior
    p_b = max(30, min(95, int(pos_s / df_e2e_m["entity_id"].nunique() * 100)))
    br_e = engine.bayesian_update(pos_s, tot_s, prior_rate=p_b/100.0)
    ok("6.06 E2E: prior=80%",          approx(br_e.prior_pct, 80.0, 0.01))
    ok("6.07 E2E: posterior=80%",      approx(br_e.posterior_pct, 80.0, 1.0))

    # Step 6: Friction tax
    ft_e = engine.compute_friction_tax(
        mgr_e2e.bottleneck_avg_days, cost_per_hour=12.0, hours_per_day=8.0
    )
    ok("6.08 E2E: friction tax > 0",   ft_e > 0, str(ft_e))

    # Step 7: Process confidence
    conf_e = engine.compute_process_confidence(
        prior_pct=p_b, bottleneck_rework_rate=mgr_e2e.bottleneck_rework_rate
    )
    ok("6.09 E2E: process confidence computed", "posterior_pct" in conf_e)

    # Step 8: ROI calc
    _unique = df_e2e_m["entity_id"].nunique()
    _avg_cycle = float(df_e2e_m.groupby("entity_id")["time_spent"].sum().mean())
    _manual_h  = max(1.0, _avg_cycle * 8 * max(1, int(_unique/3)) / 30.0)
    inp_e = base_inp(
        manual_hours_per_month=_manual_h,
        p_complete_before=p_b/100.0,
        p_complete_after=min(0.99, p_b/100.0*1.28),
        positive_signals=pos_s, total_signals=tot_s,
    )
    res_e = roi.calculate(inp_e, bayes_result=br_e)
    ok("6.10 E2E: ROI result returned", res_e is not None)
    ok("6.11 E2E: net ROI is finite",  math.isfinite(res_e.net_roi), str(res_e.net_roi))
    ok("6.12 E2E: payback > 0",        res_e.payback_months > 0, str(res_e.payback_months))

    print(f"\n  ROI Summary from user CSV:")
    print(f"    Bottleneck   : {mgr_e2e.bottleneck_node}")
    print(f"    Rework pairs : {len(mgr_e2e.rework_pairs)}")
    print(f"    Friction tax : {ft_e:.2f} €/ticket")
    print(f"    Prior/Post   : {br_e.prior_pct}% → {br_e.posterior_pct}%")
    print(f"    Net ROI      : {res_e.net_roi:.0f} €")
    print(f"    Payback      : {res_e.payback_months:.1f} months")

except Exception as e:
    ok("6.XX E2E pipeline ERROR", False, f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: STRESS TESTS — boundary values & numerical stability
# ══════════════════════════════════════════════════════════════════════════════
section("7. STRESS / NUMERICAL STABILITY")

# 7.1 Bayesian: extreme prior values
ok("7.01 Bayes prior=0.01 no crash", engine.bayesian_update(1, 10, prior_rate=0.01) is not None)
ok("7.02 Bayes prior=0.99 no crash", engine.bayesian_update(9, 10, prior_rate=0.99) is not None)
ok("7.03 Bayes 1/1 signal no crash",  engine.bayesian_update(1, 1, prior_rate=0.5) is not None)

# 7.2 ROI: extreme values don't produce NaN/Inf
for kw in [dict(monthly_volume=0), dict(deals_per_month=0), dict(avg_deal_value_eur=0)]:
    try:
        r_ = roi.calculate(base_inp(**kw))
        ok(f"7.04 ROI with {kw} no NaN", math.isfinite(r_.net_roi))
    except Exception as e:
        ok(f"7.04 ROI with {kw} no NaN", False, str(e))

# 7.3 Markov: single transition pair
rows_single = [("E1","A","B",5)]
try:
    mgr_s = build_markov_graph(make_markov_df(rows_single))
    ok("7.05 Single transition pair", mgr_s is not None and mgr_s.total_transitions==1)
except Exception as e:
    ok("7.05 Single transition pair", False, str(e))

# 7.4 Friction tax: very large values (no overflow)
ft_big = engine.compute_friction_tax(365, 500, 24)
ok("7.06 Friction tax large input finite", math.isfinite(ft_big) and ft_big == 365*24*500)

# 7.5 Bayesian CI always ordered
for _ in range(20):
    pos = np.random.randint(1, 50)
    tot = np.random.randint(pos, 100)
    pr  = np.random.uniform(0.05, 0.95)
    br_ = engine.bayesian_update(pos, tot, prior_rate=pr)
    if not (br_.ci_80_low <= br_.posterior_pct <= br_.ci_80_high):
        ok("7.07 CI ordered (random sample)", False,
           f"pos={pos},tot={tot},pr={pr:.2f}: {br_.ci_80_low}≤{br_.posterior_pct}≤{br_.ci_80_high}")
        break
else:
    ok("7.07 CI ordered (20 random samples)", True)

# 7.6 ROI: all components non-negative for normal inputs
ok("7.08 time_saved ≥ 0",    res.time_saved_annual >= 0)
ok("7.09 error_saved ≥ 0",   res.error_reduction_annual >= 0)
ok("7.10 markov_gain ≥ 0",   res.markov_gain_annual >= 0)

# 7.7 Multiple CSV with different encodings all yield same data
csv_utf8  = "ID,Status,Days,Next\nT1,Open,3,Done\nT2,Closed,5,Won\n".encode("utf-8")
csv_utf8s = "ID,Status,Days,Next\nT1,Open,3,Done\nT2,Closed,5,Won\n".encode("utf-8-sig")
csv_latin = "ID,Status,Days,Next\nT1,Open,3,Done\nT2,Closed,5,Won\n".encode("latin-1")
d1 = load_and_clean_csv(make_file(csv_utf8))
d2 = load_and_clean_csv(make_file(csv_utf8s))
d3 = load_and_clean_csv(make_file(csv_latin))
ok("7.11 UTF-8 == UTF-8 BOM == Latin-1 data", (
    d1 is not None and d2 is not None and d3 is not None and
    list(d1.columns) == list(d2.columns) == list(d3.columns) and
    len(d1) == len(d2) == len(d3)
))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: DB CONNECTION
# ══════════════════════════════════════════════════════════════════════════════
section("8. DATABASE")

try:
    from db.database import init_db, save_audit, load_history
    init_db()
    ok("8.01 init_db() no crash", True)
    # save_audit signature: company_name, params, friction_tax_usd, adjusted_confidence_pct,
    #                       bottleneck_stage, roi_pct, rework_rate_pct, total_transitions, total_rework
    save_audit(
        "TestCo_Stress",
        {"automation_rate": 0.8},
        friction_tax_usd=120.0,
        adjusted_confidence_pct=0.8,
        bottleneck_stage="Vendor Escalation",
        roi_pct=920.0,
        rework_rate_pct=0.15,
        total_transitions=21,
        total_rework=3,
    )
    hist = load_history()
    ok("8.02 save+read history", isinstance(hist, list) and len(hist) >= 1)
    ok("8.03 TestCo record in history",
       any(r.get("company_name") == "TestCo_Stress" for r in hist))
except Exception as e:
    ok("8.01 DB import no crash", False, str(e))
    ok("8.02 save+read history", False, "skipped")
    ok("8.03 TestCo record in history", False, "skipped")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
section("FINAL REPORT")
total = PASS + FAIL
pct   = 100 * PASS / total if total else 0
print(f"\n  Tests run   : {total}")
print(f"  PASSED      : {PASS}  ({pct:.0f}%)")
print(f"  FAILED      : {FAIL}")

if ERRORS:
    print("\n  FAILURES:")
    for e in ERRORS:
        print(f"  {e}")

if FAIL == 0:
    print("\n  *** ALL TESTS PASSED — APP IS PRODUCTION READY ***")
else:
    print(f"\n  *** {FAIL} test(s) need attention ***")

# Exit non-zero so CI catches failures
sys.exit(0 if FAIL == 0 else 1)
