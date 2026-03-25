import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import csv
from datetime import datetime, timedelta

import numpy as np

from etl.extractor import MatrixExtractor
from core.math_engine import MathEngine
from core.roi_engine import ROIEngine, ROIInput


FLOWS = [
    (0.35, ["New", "In Review", "Approved"]),
    (0.25, ["New", "In Review", "Revision", "In Review", "Approved"]),
    (0.15, ["New", "In Review", "Rejected"]),
    (0.10, ["New", "In Review", "Revision", "Rejected"]),
    (0.15, ["New", "In Review", "Revision", "In Review", "Revision", "Approved"]),
]

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_client_data.csv")


def generate_mock_csv(path=OUTPUT_PATH, seed=42):
    rng = random.Random(seed)

    weights = [f[0] for f in FLOWS]
    patterns = [f[1] for f in FLOWS]
    cumulative = []
    total = 0.0
    for w in weights:
        total += w
        cumulative.append(total)

    base_date = datetime(2025, 1, 1, 8, 0, 0)
    rows = []

    for deal_id in range(1, 101):
        r = rng.random()
        flow = next(p for c, p in zip(cumulative, patterns) if r <= c)

        offset_days = rng.uniform(0, 330)
        current_ts = base_date + timedelta(days=offset_days)

        deal_value = round(rng.uniform(300, 1200), 2)

        for status in flow:
            has_error = "true" if status == "Revision" else "false"
            rows.append({
                "Deal_ID": deal_id,
                "Status": status,
                "Timestamp": current_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Has_Error": has_error,
                "Deal_Value": deal_value,
            })
            step_hours = rng.uniform(2, 72)
            current_ts += timedelta(hours=step_hours)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Deal_ID", "Status", "Timestamp", "Has_Error", "Deal_Value"])
        writer.writeheader()
        writer.writerows(rows)

    print("Generated: {} ({} rows)".format(path, len(rows)))
    return path


def run_integration_test(path=OUTPUT_PATH):
    try:
        extractor = MatrixExtractor()
        log = extractor.from_csv(path)

        math = MathEngine()
        graph_res = math.graph_bottleneck([
            ("New",       "In Review", 1.0),
            ("In Review", "Approved",  0.8),
            ("In Review", "Revision",  0.6),
            ("Revision",  "In Review", 0.7),
            ("Revision",  "Rejected",  0.4),
        ])

        bayes_res = math.bayesian_update(4, 5)

        engine = ROIEngine()
        inp = ROIInput(
            company_name="Marteco Digital Services",
            manual_hours_per_month=320,
            automation_rate=0.86,
            hour_rate_eur=12,
            error_rate_before_pct=8.5,
            error_rate_after_pct=1.2,
            cost_per_error_eur=95,
            monthly_volume=600,
            deal_cycle_before_days=21,
            deal_cycle_after_days=9,
            deals_per_month=25,
            avg_deal_value_eur=650,
            p_complete_before=0.74,
            p_complete_after=0.96,
            implementation_cost_eur=14000,
            positive_signals=4,
            total_signals=5,
        )
        roi_res = engine.calculate(inp, bayes_result=bayes_res)

        print("\n✅ INTEGRATION TEST PASSED")
        print("-" * 40)
        print("total_deals    :", log.total_deals)
        print("error_rate     : {:.2%}".format(log.error_rate))
        print("bottleneck_node:", graph_res.bottleneck_node)
        print("posterior_pct  : {}%".format(bayes_res.posterior_pct))
        print("net_roi        : {:,.2f} EUR".format(roi_res.net_roi))
        print("-" * 40)

        assert log.total_deals == 100,          "Expected 100 deals, got {}".format(log.total_deals)
        assert 0.15 <= log.error_rate <= 0.30,  "error_rate out of range: {:.2%}".format(log.error_rate)
        assert graph_res.bottleneck_node == "In Review", \
            "Expected bottleneck 'In Review', got '{}'".format(graph_res.bottleneck_node)
        assert abs(bayes_res.posterior_pct - 49.3) < 1.0, \
            "posterior_pct unexpected: {}".format(bayes_res.posterior_pct)
        assert roi_res.net_roi > 100000, \
            "net_roi below threshold: {:,.2f}".format(roi_res.net_roi)

        print("✅ All assertions passed.")

    except Exception as e:
        print("\n❌ INTEGRATION TEST FAILED:", e)
        raise


if __name__ == "__main__":
    generate_mock_csv()
    run_integration_test()

    print("\n--- Running: python main.py --demo ---\n")
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"), "--demo"],
        capture_output=False,
    )
    sys.exit(result.returncode)
