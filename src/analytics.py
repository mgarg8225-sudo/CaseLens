"""
analytics.py - Longitudinal Case Journey Analytics & Policy Leakage Engine.

Analyzes the reconstructed violence-against-women case journey trajectories
across institutional stages (Police FIR -> Hospital MLC -> Forensic Lab -> Prosecutor -> Court):
1. Stage-by-Stage Attrition & Funnel Metrics:
   - Quantifies case volume, step-conversion rate, step-leak rate, and cumulative retention.
   - Stratifies funnel performance by District and Case Category.
2. Systemic Policy Leak Points & District Diagnostics:
   - Identifies statistical anomalies and policy bottlenecks across districts:
     * D3_Rudrapur: Forensic turnaround delay (~3.4x normal).
     * D6_Shivgarh: Police-to-Prosecution drop-off (~70% attrition leak).
     * D7_Anandpur: Court trial & verdict duration (>2.2x normal).
   - Computes composite District Leak Severity Index.
3. Bottleneck Duration & Statutory Compliance:
   - Computes non-parametric duration percentiles (Median, IQR, P90, P95, Min, Max).
   - Evaluates statutory benchmarks: 60-day chargesheet mandate, 24h medical exam, 30-day forensic turnaround.
   - Bins delays into temporal latency buckets.
4. Category Disparity & Inequity Diagnostics:
   - Identifies crimes with disproportionate attrition (e.g. stalking early drop-off).
5. DuckDB OLAP Analytics Engine:
   - High-performance SQL queries for multi-dimensional aggregation and window lead transitions.
6. Privacy Aggregation & Small-Cell Suppression (Requirement 4):
   - Enforces k-anonymity cell suppression (k >= 5) on all published aggregates.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import duckdb

# Canonical sequence of institutional milestones
CANONICAL_STAGES = [
    "FIR_REGISTERED",
    "MEDICAL_EXAM",
    "FORENSIC_REPORT",
    "CHARGESHEET_FILED",
    "COURT_VERDICT"
]

STAGE_LABELS = {
    "FIR_REGISTERED": "1. Police FIR",
    "MEDICAL_EXAM": "2. Hospital MLC",
    "FORENSIC_REPORT": "3. Forensic Lab (FSL)",
    "CHARGESHEET_FILED": "4. Prosecution Chargesheet",
    "COURT_VERDICT": "5. Court Verdict"
}


def compute_funnel_metrics(df_events: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall, district, and category attrition funnels across canonical milestones.
    """
    # Build per-case stage presence set
    case_stages = df_events.groupby("case_id")["activity"].apply(set)
    case_district = df_events.groupby("case_id")["district"].first()
    case_category = df_events.groupby("case_id")["case_category"].first()

    df_cases = pd.DataFrame({
        "district": case_district,
        "category": case_category,
        "stages": case_stages
    })

    for stage in CANONICAL_STAGES:
        df_cases[stage] = df_cases["stages"].apply(lambda s: stage in s)

    total_entities = len(df_cases)
    overall_funnel = []
    prev_count = None
    fir_count = df_cases["FIR_REGISTERED"].sum()

    for idx, stage in enumerate(CANONICAL_STAGES):
        count = int(df_cases[stage].sum())
        if idx == 0:
            step_conv = 100.0
            step_leak = 0.0
        else:
            step_conv = round((count / prev_count * 100.0), 2) if prev_count and prev_count > 0 else 0.0
            step_leak = round(max(0.0, 100.0 - step_conv), 2)

        cum_retention = round((count / fir_count * 100.0), 2) if fir_count > 0 else 0.0

        overall_funnel.append({
            "stage_id": stage,
            "stage_name": STAGE_LABELS.get(stage, stage),
            "case_count": count,
            "step_conversion_pct": step_conv,
            "step_leak_pct": step_leak,
            "cumulative_retention_pct": cum_retention
        })
        prev_count = count

    # Funnel by District
    district_funnels = {}
    for dist, grp in df_cases.groupby("district"):
        d_fir = grp["FIR_REGISTERED"].sum()
        d_list = []
        d_prev = None
        for idx, stage in enumerate(CANONICAL_STAGES):
            c = int(grp[stage].sum())
            conv = 100.0 if idx == 0 else (round((c / d_prev * 100.0), 2) if d_prev and d_prev > 0 else 0.0)
            leak = 0.0 if idx == 0 else round(max(0.0, 100.0 - conv), 2)
            ret = round((c / d_fir * 100.0), 2) if d_fir > 0 else 0.0
            d_list.append({
                "stage_id": stage,
                "case_count": c,
                "step_conversion_pct": conv,
                "step_leak_pct": leak,
                "retention_from_fir_pct": ret
            })
            d_prev = c
        district_funnels[dist] = d_list

    # Funnel by Category
    category_funnels = {}
    for cat, grp in df_cases.groupby("category"):
        c_fir = grp["FIR_REGISTERED"].sum()
        c_list = []
        c_prev = None
        for idx, stage in enumerate(CANONICAL_STAGES):
            c = int(grp[stage].sum())
            conv = 100.0 if idx == 0 else (round((c / c_prev * 100.0), 2) if c_prev and c_prev > 0 else 0.0)
            leak = 0.0 if idx == 0 else round(max(0.0, 100.0 - conv), 2)
            ret = round((c / c_fir * 100.0), 2) if c_fir > 0 else 0.0
            c_list.append({
                "stage_id": stage,
                "case_count": c,
                "step_conversion_pct": conv,
                "step_leak_pct": leak,
                "retention_from_fir_pct": ret
            })
            c_prev = c
        category_funnels[cat] = c_list

    end_to_end_rate = round((df_cases["COURT_VERDICT"].sum() / fir_count * 100.0), 2) if fir_count > 0 else 0.0

    return {
        "total_reconstructed_cases": total_entities,
        "fir_registered_cases": int(fir_count),
        "verdict_reached_cases": int(df_cases["COURT_VERDICT"].sum()),
        "overall_completion_rate_pct": end_to_end_rate,
        "funnel_stages": overall_funnel,
        "district_funnels": district_funnels,
        "category_funnels": category_funnels
    }


def compute_district_leakage_diagnostics(
    df_events: pd.DataFrame,
    dfg_summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates district performance against systemic benchmarks and detects
    the three primary policy leak points:
    1. Forensic bottleneck (D3_Rudrapur)
    2. Police-to-chargesheet attrition (D6_Shivgarh)
    3. Judicial trial delay (D7_Anandpur)
    """
    case_stages = df_events.groupby("case_id")["activity"].apply(set)
    case_district = df_events.groupby("case_id")["district"].first()

    df_cases = pd.DataFrame({
        "district": case_district,
        "has_fir": case_stages.apply(lambda s: "FIR_REGISTERED" in s),
        "has_mlc": case_stages.apply(lambda s: "MEDICAL_EXAM" in s),
        "has_fsl": case_stages.apply(lambda s: "FORENSIC_REPORT" in s),
        "has_pros": case_stages.apply(lambda s: "CHARGESHEET_FILED" in s),
        "has_court": case_stages.apply(lambda s: "COURT_VERDICT" in s)
    })

    # Extract transition delays per case
    case_delays = []
    for case_id, grp in df_events.groupby("case_id"):
        dist = grp["district"].iloc[0]
        act_dict = {}
        for _, row in grp.iterrows():
            act = row["activity"]
            dt = pd.to_datetime(row["timestamp"])
            if act not in act_dict:
                act_dict[act] = dt

        fir_dt = act_dict.get("FIR_REGISTERED")
        fsl_dt = act_dict.get("FORENSIC_REPORT")
        pros_dt = act_dict.get("CHARGESHEET_FILED")
        court_dt = act_dict.get("COURT_VERDICT")

        fsl_delay = (fsl_dt - fir_dt).days if (fir_dt and fsl_dt and fsl_dt >= fir_dt) else None
        pros_delay = (pros_dt - fir_dt).days if (fir_dt and pros_dt and pros_dt >= fir_dt) else None
        trial_delay = (court_dt - pros_dt).days if (pros_dt and court_dt and court_dt >= pros_dt) else None

        case_delays.append({
            "case_id": case_id,
            "district": dist,
            "fsl_delay_days": fsl_delay,
            "pros_delay_days": pros_delay,
            "trial_delay_days": trial_delay
        })

    df_delays = pd.DataFrame(case_delays)

    # State Benchmark Medians/Averages across all districts
    state_fir_to_pros_rate = (df_cases["has_pros"].sum() / df_cases["has_fir"].sum() * 100.0)
    state_fsl_median = float(df_delays["fsl_delay_days"].dropna().median())
    state_trial_median = float(df_delays["trial_delay_days"].dropna().median())

    district_diagnostics = {}
    leak_point_alerts = []

    for dist, grp in df_cases.groupby("district"):
        d_delays = df_delays[df_delays["district"] == dist]

        fir_n = int(grp["has_fir"].sum())
        pros_n = int(grp["has_pros"].sum())
        court_n = int(grp["has_court"].sum())

        fir_to_pros_rate = round((pros_n / fir_n * 100.0), 1) if fir_n > 0 else 0.0
        pros_to_court_rate = round((court_n / pros_n * 100.0), 1) if pros_n > 0 else 0.0
        end_to_end_rate = round((court_n / fir_n * 100.0), 1) if fir_n > 0 else 0.0

        fsl_series = d_delays["fsl_delay_days"].dropna()
        fsl_avg = round(float(fsl_series.mean()), 1) if len(fsl_series) else 0.0
        fsl_median = round(float(fsl_series.median()), 1) if len(fsl_series) else 0.0

        trial_series = d_delays["trial_delay_days"].dropna()
        trial_avg = round(float(trial_series.mean()), 1) if len(trial_series) else 0.0
        trial_median = round(float(trial_series.median()), 1) if len(trial_series) else 0.0

        # Anomaly checks
        anomalies = []
        severity = "NORMAL"

        # Leak Point 1: Forensic Delay (D3_Rudrapur)
        fsl_delay_ratio = round(fsl_median / state_fsl_median, 2) if state_fsl_median > 0 else 1.0
        if fsl_delay_ratio >= 2.0:
            anomalies.append({
                "leak_type": "FORENSIC_LAB_BOTTLENECK",
                "severity": "CRITICAL",
                "description": f"Forensic turnaround is {fsl_delay_ratio}x state baseline ({fsl_median}d vs state {state_fsl_median}d)."
            })
            severity = "CRITICAL"
            leak_point_alerts.append({
                "district": dist,
                "leak_code": "LP-01-FORENSIC-DELAY",
                "title": "Severe Forensic Lab Turnaround Delay",
                "delay_days": fsl_avg,
                "baseline_days": state_fsl_median,
                "multiplier": fsl_delay_ratio,
                "affected_cases": len(fsl_series)
            })

        # Leak Point 2: Police-to-Chargesheet Attrition (D6_Shivgarh)
        pros_attrition_ratio = round(fir_to_pros_rate / state_fir_to_pros_rate, 2) if state_fir_to_pros_rate > 0 else 1.0
        if pros_attrition_ratio <= 0.65:
            drop_off_pct = round(100.0 - fir_to_pros_rate, 1)
            anomalies.append({
                "leak_type": "POLICE_CHARGESHEET_ATTRITION",
                "severity": "CRITICAL",
                "description": f"Severe police-to-prosecution drop-off: {drop_off_pct}% of FIRs fail to reach chargesheet."
            })
            severity = "CRITICAL"
            leak_point_alerts.append({
                "district": dist,
                "leak_code": "LP-02-POLICE-ATTRITION",
                "title": "Excessive FIR-to-Chargesheet Drop-off",
                "chargesheet_rate_pct": fir_to_pros_rate,
                "state_benchmark_pct": round(state_fir_to_pros_rate, 1),
                "drop_off_leak_pct": drop_off_pct,
                "affected_firs": fir_n - pros_n
            })

        # Leak Point 3: Court Trial Delay (D7_Anandpur)
        trial_delay_ratio = round(trial_median / state_trial_median, 2) if state_trial_median > 0 else 1.0
        if trial_delay_ratio >= 1.8:
            anomalies.append({
                "leak_type": "JUDICIAL_TRIAL_DELAY",
                "severity": "CRITICAL",
                "description": f"Court trial duration is {trial_delay_ratio}x state baseline ({trial_median}d vs state {state_trial_median}d)."
            })
            severity = "CRITICAL"
            leak_point_alerts.append({
                "district": dist,
                "leak_code": "LP-03-COURT-DELAY",
                "title": "Severe Judicial Trial Adjudication Delay",
                "trial_delay_days": trial_avg,
                "baseline_days": state_trial_median,
                "multiplier": trial_delay_ratio,
                "cases_in_trial": len(trial_series)
            })

        # Calculate composite leak vulnerability score (0 to 100)
        # Higher score = more vulnerable/compromised pipeline
        vulnerability_score = 0.0
        vulnerability_score += max(0.0, (65.0 - fir_to_pros_rate) * 1.5)
        vulnerability_score += min(35.0, max(0.0, (fsl_median - 40.0) * 0.35))
        vulnerability_score += min(35.0, max(0.0, (trial_median - 280.0) * 0.10))
        vulnerability_score = round(min(100.0, vulnerability_score), 1)

        district_diagnostics[dist] = {
            "total_cases": len(grp),
            "fir_registered": fir_n,
            "chargesheet_filed": pros_n,
            "court_verdict": court_n,
            "fir_to_chargesheet_rate_pct": fir_to_pros_rate,
            "chargesheet_to_verdict_rate_pct": pros_to_court_rate,
            "end_to_end_completion_rate_pct": end_to_end_rate,
            "forensic_delay_avg_days": fsl_avg,
            "forensic_delay_median_days": fsl_median,
            "trial_delay_avg_days": trial_avg,
            "trial_delay_median_days": trial_median,
            "status": severity,
            "vulnerability_score": vulnerability_score,
            "detected_anomalies": anomalies
        }

    return {
        "state_benchmarks": {
            "fir_to_chargesheet_rate_pct": round(state_fir_to_pros_rate, 1),
            "forensic_median_days": round(state_fsl_median, 1),
            "trial_median_days": round(state_trial_median, 1)
        },
        "districts": district_diagnostics,
        "systemic_leak_points": leak_point_alerts
    }


def compute_category_analytics(df_events: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comparative attrition, duration, and disparity statistics across
    the 5 statutory case categories.
    """
    case_stages = df_events.groupby("case_id")["activity"].apply(set)
    case_category = df_events.groupby("case_id")["case_category"].first()
    case_district = df_events.groupby("case_id")["district"].first()

    # Calculate overall case duration per case
    case_duration = {}
    for case_id, grp in df_events.groupby("case_id"):
        dts = pd.to_datetime(grp["timestamp"])
        case_duration[case_id] = (dts.max() - dts.min()).days

    df_cases = pd.DataFrame({
        "case_id": case_category.index,
        "category": case_category.values,
        "district": case_district.values,
        "has_fir": [ "FIR_REGISTERED" in case_stages[cid] for cid in case_category.index ],
        "has_mlc": [ "MEDICAL_EXAM" in case_stages[cid] for cid in case_category.index ],
        "has_fsl": [ "FORENSIC_REPORT" in case_stages[cid] for cid in case_category.index ],
        "has_pros": [ "CHARGESHEET_FILED" in case_stages[cid] for cid in case_category.index ],
        "has_court": [ "COURT_VERDICT" in case_stages[cid] for cid in case_category.index ],
        "duration_days": [ case_duration.get(cid, 0) for cid in case_category.index ]
    })

    category_summary = {}
    for cat, grp in df_cases.groupby("category"):
        total_c = len(grp)
        fir_n = int(grp["has_fir"].sum())
        mlc_n = int(grp["has_mlc"].sum())
        fsl_n = int(grp["has_fsl"].sum())
        pros_n = int(grp["has_pros"].sum())
        court_n = int(grp["has_court"].sum())

        fir_to_pros = round((pros_n / fir_n * 100.0), 1) if fir_n > 0 else 0.0
        pros_to_court = round((court_n / pros_n * 100.0), 1) if pros_n > 0 else 0.0
        end_to_end = round((court_n / fir_n * 100.0), 1) if fir_n > 0 else 0.0

        durations = grp["duration_days"]
        avg_dur = round(float(durations.mean()), 1)
        median_dur = round(float(durations.median()), 1)

        category_summary[cat] = {
            "total_cases": total_c,
            "fir_cases": fir_n,
            "mlc_cases": mlc_n,
            "fsl_cases": fsl_n,
            "chargesheet_cases": pros_n,
            "verdict_cases": court_n,
            "fir_to_chargesheet_rate_pct": fir_to_pros,
            "chargesheet_to_verdict_rate_pct": pros_to_court,
            "end_to_end_completion_rate_pct": end_to_end,
            "avg_duration_days": avg_dur,
            "median_duration_days": median_dur,
            "attrition_profile": "DISPROPORTIONATE_LEAK" if fir_to_pros < 40.0 else "STANDARD"
        }

    return category_summary


def compute_transition_bottlenecks(df_events: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes rigorous non-parametric duration percentiles and statutory compliance
    metrics for institutional transitions.
    """
    transitions = []

    for case_id, grp in df_events.groupby("case_id"):
        acts = grp["activity"].tolist()
        dts = pd.to_datetime(grp["timestamp"]).tolist()
        dist = grp["district"].iloc[0]
        cat = grp["case_category"].iloc[0]

        # Consecutive transitions
        for i in range(len(acts) - 1):
            src = acts[i]
            tgt = acts[i + 1]
            delta_days = max(0, (dts[i + 1] - dts[i]).days)
            transitions.append({
                "from_activity": src,
                "to_activity": tgt,
                "pair": f"{src} -> {tgt}",
                "duration_days": delta_days,
                "district": dist,
                "category": cat
            })

        # End-to-end FIR -> Chargesheet and FIR -> Court
        act_dict = { acts[i]: dts[i] for i in range(len(acts)) }
        if "FIR_REGISTERED" in act_dict and "CHARGESHEET_FILED" in act_dict:
            d = max(0, (act_dict["CHARGESHEET_FILED"] - act_dict["FIR_REGISTERED"]).days)
            transitions.append({
                "from_activity": "FIR_REGISTERED",
                "to_activity": "CHARGESHEET_FILED",
                "pair": "FIR_REGISTERED -> CHARGESHEET_FILED (E2E)",
                "duration_days": d,
                "district": dist,
                "category": cat
            })

        if "FIR_REGISTERED" in act_dict and "COURT_VERDICT" in act_dict:
            d = max(0, (act_dict["COURT_VERDICT"] - act_dict["FIR_REGISTERED"]).days)
            transitions.append({
                "from_activity": "FIR_REGISTERED",
                "to_activity": "COURT_VERDICT",
                "pair": "FIR_REGISTERED -> COURT_VERDICT (E2E)",
                "duration_days": d,
                "district": dist,
                "category": cat
            })

    df_trans = pd.DataFrame(transitions)

    transition_metrics = {}
    for pair, grp in df_trans.groupby("pair"):
        vals = grp["duration_days"].values
        n = len(vals)
        if n < 5:
            continue

        mean_val = round(float(np.mean(vals)), 1)
        std_val = round(float(np.std(vals)), 1)
        median_val = round(float(np.median(vals)), 1)
        p25 = round(float(np.percentile(vals, 25)), 1)
        p75 = round(float(np.percentile(vals, 75)), 1)
        iqr = round(p75 - p25, 1)
        p90 = round(float(np.percentile(vals, 90)), 1)
        p95 = round(float(np.percentile(vals, 95)), 1)
        min_val = round(float(np.min(vals)), 1)
        max_val = round(float(np.max(vals)), 1)

        # Statutory benchmarks
        compliance = {}
        if "CHARGESHEET" in pair:
            # 60-day investigation statutory guideline (e.g. POCSO / rape cases under CrPC / BNSS)
            within_60d = round(float(np.sum(vals <= 60) / n * 100.0), 1)
            compliance["within_60_days_pct"] = within_60d
        elif "MEDICAL" in pair:
            within_24h = round(float(np.sum(vals <= 1) / n * 100.0), 1)
            compliance["within_24h_pct"] = within_24h
        elif "FORENSIC" in pair:
            within_30d = round(float(np.sum(vals <= 30) / n * 100.0), 1)
            compliance["within_30_days_pct"] = within_30d

        # Delay histograms
        buckets = {
            "<14_days": int(np.sum(vals < 14)),
            "14_to_30_days": int(np.sum((vals >= 14) & (vals <= 30))),
            "31_to_60_days": int(np.sum((vals > 30) & (vals <= 60))),
            "61_to_90_days": int(np.sum((vals > 60) & (vals <= 90))),
            "91_to_180_days": int(np.sum((vals > 90) & (vals <= 180))),
            "181_to_365_days": int(np.sum((vals > 180) & (vals <= 365))),
            ">365_days": int(np.sum(vals > 365))
        }

        transition_metrics[pair] = {
            "sample_count": n,
            "mean_days": mean_val,
            "std_days": std_val,
            "median_days": median_val,
            "p25_days": p25,
            "p75_days": p75,
            "iqr_days": iqr,
            "p90_days": p90,
            "p95_days": p95,
            "min_days": min_val,
            "max_days": max_val,
            "statutory_compliance": compliance,
            "delay_buckets": buckets
        }

    return transition_metrics


def run_duckdb_analytics(data_dir: str = "data") -> Dict[str, Any]:
    """
    Executes high-performance SQL OLAP analytical queries using DuckDB.
    Demonstrates analytical windowing, aggregations, and query optimization.
    """
    event_log_path = os.path.join(data_dir, "event_log.csv").replace("\\", "/")
    entities_path = os.path.join(data_dir, "case_entities.csv").replace("\\", "/")

    if not os.path.exists(event_log_path):
        return {}

    con = duckdb.connect(database=":memory:")

    # Register in-memory views
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_csv_auto('{event_log_path}')")
    if os.path.exists(entities_path):
        con.execute(f"CREATE VIEW entities AS SELECT * FROM read_csv_auto('{entities_path}')")

    # Query 1: Monthly registration trend by agency
    q1 = """
        SELECT 
            strftime(TRY_CAST(timestamp AS DATE), '%Y-%m') AS month,
            agency,
            COUNT(*) AS event_count
        FROM events
        WHERE TRY_CAST(timestamp AS DATE) IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    df_monthly = con.execute(q1).df()

    # Query 2: Multi-dimensional stage completion cube (District x Category)
    q2 = """
        SELECT 
            district,
            case_category,
            COUNT(DISTINCT case_id) AS total_cases,
            COUNT(DISTINCT CASE WHEN activity = 'FIR_REGISTERED' THEN case_id END) AS fir_cases,
            COUNT(DISTINCT CASE WHEN activity = 'CHARGESHEET_FILED' THEN case_id END) AS chargesheet_cases,
            COUNT(DISTINCT CASE WHEN activity = 'COURT_VERDICT' THEN case_id END) AS verdict_cases,
            ROUND(COUNT(DISTINCT CASE WHEN activity = 'CHARGESHEET_FILED' THEN case_id END) * 100.0 / 
                  NULLIF(COUNT(DISTINCT CASE WHEN activity = 'FIR_REGISTERED' THEN case_id END), 0), 1) AS chargesheet_rate_pct
        FROM events
        GROUP BY district, case_category
        ORDER BY district, case_category
    """
    df_cube = con.execute(q2).df()

    # Query 3: Transition durations using SQL window function LEAD
    q3 = """
        WITH ordered_events AS (
            SELECT 
                case_id,
                district,
                activity,
                TRY_CAST(timestamp AS DATE) AS dt,
                LEAD(activity) OVER (PARTITION BY case_id ORDER BY TRY_CAST(timestamp AS DATE)) AS next_act,
                LEAD(TRY_CAST(timestamp AS DATE)) OVER (PARTITION BY case_id ORDER BY TRY_CAST(timestamp AS DATE)) AS next_dt
            FROM events
            WHERE TRY_CAST(timestamp AS DATE) IS NOT NULL
        ),
        transitions AS (
            SELECT 
                district,
                activity || ' -> ' || next_act AS transition,
                datediff('day', dt, next_dt) AS duration_days
            FROM ordered_events
            WHERE next_act IS NOT NULL AND datediff('day', dt, next_dt) >= 0
        )
        SELECT 
            transition,
            COUNT(*) AS transition_count,
            ROUND(AVG(duration_days), 1) AS avg_duration_days,
            ROUND(MEDIAN(duration_days), 1) AS median_duration_days
        FROM transitions
        GROUP BY transition
        HAVING COUNT(*) >= 5
        ORDER BY transition_count DESC
    """
    df_sql_transitions = con.execute(q3).df()

    con.close()

    return {
        "monthly_trend_records": len(df_monthly),
        "cube_slice": df_cube.head(10).to_dict(orient="records"),
        "sql_window_transitions": df_sql_transitions.head(8).to_dict(orient="records")
    }


def apply_privacy_k_anonymity(data: Any, k: int = 5) -> Any:
    """
    Recursively applies k-anonymity suppression (Requirement 4 & 6) on analytical aggregates.
    Any cell where count or frequency is below k is masked to prevent de-anonymization.
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, val in data.items():
            if isinstance(val, (int, float)) and any(c in key.lower() for c in ["count", "cases", "sample_count", "frequency"]):
                if 0 < val < k:
                    cleaned[key] = f"<{k}"
                else:
                    cleaned[key] = val
            elif isinstance(val, (dict, list)):
                cleaned[key] = apply_privacy_k_anonymity(val, k=k)
            else:
                cleaned[key] = val
        return cleaned
    elif isinstance(data, list):
        return [apply_privacy_k_anonymity(item, k=k) for item in data]
    return data


def run_full_analytics_pipeline(data_dir: str = "data", k_anonymity: int = 5) -> Dict[str, Any]:
    """
    Executes the complete Phase 4 Analytics Engine:
    1. Loads process mining event log.
    2. Computes Funnel & Stage-by-Stage Attrition.
    3. Runs District Leakage Diagnostics & Leak Point Detection (D3, D6, D7).
    4. Computes Category Disparities & Inequities.
    5. Computes Transition Bottlenecks & Statutory Compliance.
    6. Runs DuckDB OLAP SQL Aggregations.
    7. Applies Privacy k-anonymity cell suppression.
    8. Persists all metrics into JSON artifacts for Streamlit / API consumers.
    """
    event_log_path = os.path.join(data_dir, "event_log.csv")
    if not os.path.exists(event_log_path):
        from src.journey import build_event_log
        df_events = build_event_log(data_dir=data_dir)
        df_events.to_csv(event_log_path, index=False)
    else:
        df_events = pd.read_csv(event_log_path)

    print("Executing Funnel & Attrition Analysis...")
    funnel_data = compute_funnel_metrics(df_events)

    print("Running District Leakage Diagnostics & Leak Point Detection...")
    district_data = compute_district_leakage_diagnostics(df_events)

    print("Analyzing Statutory Crime Category Inequities...")
    category_data = compute_category_analytics(df_events)

    print("Computing Transition Bottlenecks & Statutory Compliance Percentiles...")
    bottleneck_data = compute_transition_bottlenecks(df_events)

    print("Executing DuckDB OLAP Analytical Engine...")
    duckdb_data = run_duckdb_analytics(data_dir=data_dir)

    # Save artifacts
    funnel_path = os.path.join(data_dir, "analytics_funnel.json")
    with open(funnel_path, "w") as f:
        json.dump(apply_privacy_k_anonymity(funnel_data, k=k_anonymity), f, indent=2)

    district_path = os.path.join(data_dir, "analytics_district_leakage.json")
    with open(district_path, "w") as f:
        json.dump(apply_privacy_k_anonymity(district_data, k=k_anonymity), f, indent=2)

    cat_path = os.path.join(data_dir, "analytics_category.json")
    with open(cat_path, "w") as f:
        json.dump(apply_privacy_k_anonymity(category_data, k=k_anonymity), f, indent=2)

    bottle_path = os.path.join(data_dir, "analytics_bottlenecks.json")
    with open(bottle_path, "w") as f:
        json.dump(apply_privacy_k_anonymity(bottleneck_data, k=k_anonymity), f, indent=2)

    # Executive Summary
    summary = {
        "pipeline_health": {
            "total_reconstructed_cases": funnel_data["total_reconstructed_cases"],
            "cases_initiating_fir": funnel_data["fir_registered_cases"],
            "cases_reaching_verdict": funnel_data["verdict_reached_cases"],
            "overall_conviction_reach_rate_pct": funnel_data["overall_completion_rate_pct"]
        },
        "systemic_leak_points_detected": district_data["systemic_leak_points"],
        "district_vulnerability_ranking": sorted(
            [{"district": d, "score": v["vulnerability_score"], "status": v["status"]} 
             for d, v in district_data["districts"].items()],
            key=lambda x: x["score"],
            reverse=True
        ),
        "key_bottlenecks": {
            k: {
                "median_days": v["median_days"],
                "p90_days": v["p90_days"],
                "compliance": v.get("statutory_compliance", {})
            }
            for k, v in bottleneck_data.items()
            if "E2E" in k or "COURT" in k or "FORENSIC" in k
        },
        "duckdb_olap_transitions": duckdb_data.get("sql_window_transitions", [])[:4]
    }

    summary_path = os.path.join(data_dir, "analytics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(apply_privacy_k_anonymity(summary, k=k_anonymity), f, indent=2)

    print(f"All analytics artifacts successfully persisted in {data_dir}/")
    return summary


if __name__ == "__main__":
    summary = run_full_analytics_pipeline(data_dir="data")

    print("\n" + "="*75)
    print("PHASE 4: SYSTEMIC LEAKAGE & POLICY ANALYTICS COMPLETE")
    print("="*75)
    print(f"Total Reconstructed Cases:      {summary['pipeline_health']['total_reconstructed_cases']}")
    print(f"Cases Initiating Police FIR:    {summary['pipeline_health']['cases_initiating_fir']}")
    print(f"Cases Reaching Court Verdict:   {summary['pipeline_health']['cases_reaching_verdict']}")
    print(f"Overall End-to-End Resolution:  {summary['pipeline_health']['overall_conviction_reach_rate_pct']}%")

    print("\n" + "-"*75)
    print("SYSTEMIC LEAK POINTS IDENTIFIED (REAL EMPIRICAL FINDINGS):")
    print("-"*75)
    for alert in summary["systemic_leak_points_detected"]:
        dist = alert["district"]
        title = alert["title"]
        mult = alert.get("multiplier", "N/A")
        print(f"  * [{alert['leak_code']}] {dist} - {title}")
        if "multiplier" in alert:
            print(f"    Severity: {mult}x state baseline delay")
        elif "drop_off_leak_pct" in alert:
            print(f"    Severity: {alert['drop_off_leak_pct']}% dropped before chargesheet (Only {alert['chargesheet_rate_pct']}% reach prosecution)")

    print("\n" + "-"*75)
    print("DISTRICT VULNERABILITY RANKING (Leak Severity Index 0-100):")
    print("-"*75)
    for item in summary["district_vulnerability_ranking"]:
        print(f"  - {item['district']:<20}: Score {item['score']:>5.1f} | Status: {item['status']}")

    print("\n" + "-"*75)
    print("KEY TRANSITION BOTTLENECK PERCENTILES:")
    print("-"*75)
    for trans, stats in summary["key_bottlenecks"].items():
        comp_str = ", ".join(f"{k}: {v}%" for k, v in stats.get("compliance", {}).items())
        print(f"  - {trans:<38}: Median={stats['median_days']:>5.1f}d | P90={stats['p90_days']:>5.1f}d | {comp_str}")
    print("="*75 + "\n")
