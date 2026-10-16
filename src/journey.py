"""
journey.py - Case Journey Reconstruction & Process Mining Engine.

Reconstructs the longitudinal journey of violence-against-women cases across
institutional milestones:
FIR -> Medical Exam -> Forensic Report -> Chargesheet -> Trial -> Verdict.

Implements:
1. Event Log Construction:
   - Aggregates linked records per Case Entity into chronologically ordered event streams.
   - Enforces logical process tie-breaking when agency timestamps coincide.
2. Process Mining & Directly-Follows Graph (DFG) Analysis:
   - Integrates PM4Py for DFG discovery and performance mining.
   - Discovers distinct journey path variants and calculates throughput time.
3. Bottleneck & Drop-Off (Leakage) Diagnostics:
   - Identifies start activities (entry points) and end activities (where cases leak out).
   - Computes transition probabilities and duration statistics (mean, median, P90).
   - Stratifies bottlenecks by District and Case Category.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import pm4py

# Activity and stage order definitions
ACTIVITY_MAP = {
    "police_fir": "FIR_REGISTERED",
    "hospital_mlc": "MEDICAL_EXAM",
    "forensic_lab": "FORENSIC_REPORT",
    "prosecutor": "CHARGESHEET_FILED",
    "court": "COURT_VERDICT"
}

STAGE_ORDER = {
    "FIR_REGISTERED": 1,
    "MEDICAL_EXAM": 2,
    "FORENSIC_REPORT": 3,
    "CHARGESHEET_FILED": 4,
    "COURT_VERDICT": 5
}


def build_event_log(data_dir: str = "data") -> pd.DataFrame:
    """
    Constructs a standardized process mining event log from linked case entities
    and agency record metadata.
    """
    ent_path = os.path.join(data_dir, "case_entities.csv")
    df_ent = pd.read_csv(ent_path)

    # Load category lookup from police fir and prosecutor files
    category_lookup = {}
    fir_path = os.path.join(data_dir, "police_fir.csv")
    if os.path.exists(fir_path):
        df_fir = pd.read_csv(fir_path)
        for _, r in df_fir.iterrows():
            if pd.notna(r.get("offense_category")):
                category_lookup[f"police_fir__{r['fir_number']}"] = str(r["offense_category"])

    pros_path = os.path.join(data_dir, "prosecutor.csv")
    if os.path.exists(pros_path):
        df_pros = pd.read_csv(pros_path)
        for _, r in df_pros.iterrows():
            if pd.notna(r.get("case_category")):
                category_lookup[f"prosecutor__{r['prosecution_file_no']}"] = str(r["case_category"])

    # Determine dominant case category per entity
    entity_categories = {}
    for _, row in df_ent.iterrows():
        eid = row["entity_id"]
        uid = row["unique_id"]
        if uid in category_lookup:
            entity_categories[eid] = category_lookup[uid]

    events = []
    for _, row in df_ent.iterrows():
        eid = row["entity_id"]
        agency = row["source_agency"]
        activity = ACTIVITY_MAP.get(agency, agency.upper())

        # Determine best event timestamp
        dt_str = row.get("record_date")
        if not pd.notna(dt_str) or not str(dt_str).strip():
            dt_str = row.get("incident_date")

        if not pd.notna(dt_str) or not str(dt_str).strip():
            continue

        cat = entity_categories.get(eid, "general_violence")

        events.append({
            "case_id": eid,
            "activity": activity,
            "timestamp": str(dt_str).strip(),
            "agency": agency,
            "district": row["district"],
            "case_category": cat,
            "record_id": row["record_id"],
            "stage_order": STAGE_ORDER.get(activity, 99)
        })

    df_events = pd.DataFrame(events)

    # Sort events by case_id, timestamp, and standard logical stage order
    df_events["dt"] = pd.to_datetime(df_events["timestamp"], errors="coerce")
    df_events = df_events.sort_values(by=["case_id", "dt", "stage_order"]).reset_index(drop=True)
    df_events = df_events.drop(columns=["dt", "stage_order"])

    return df_events


def discover_process_variants(df_events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts complete trace variants per case entity and computes frequency and duration.
    """
    # Group activities and timestamps by case_id
    traces = []
    for case_id, grp in df_events.groupby("case_id"):
        act_seq = grp["activity"].tolist()
        trace_str = " -> ".join(act_seq)

        # Throughput time
        dts = pd.to_datetime(grp["timestamp"])
        duration_days = (dts.max() - dts.min()).days

        district = grp["district"].iloc[0]
        cat = grp["case_category"].iloc[0]

        traces.append({
            "case_id": case_id,
            "trace": trace_str,
            "path_length": len(act_seq),
            "duration_days": duration_days,
            "start_activity": act_seq[0],
            "end_activity": act_seq[-1],
            "district": district,
            "case_category": cat
        })

    df_traces = pd.DataFrame(traces)

    total_cases = len(df_traces)
    variant_summary = df_traces.groupby("trace").agg(
        case_count=("case_id", "count"),
        avg_duration_days=("duration_days", "mean"),
        median_duration_days=("duration_days", "median"),
        path_length=("path_length", "first")
    ).reset_index()

    variant_summary["share_pct"] = (variant_summary["case_count"] / total_cases * 100).round(2)
    variant_summary["avg_duration_days"] = variant_summary["avg_duration_days"].round(1)
    variant_summary["median_duration_days"] = variant_summary["median_duration_days"].round(1)
    variant_summary = variant_summary.sort_values(by="case_count", ascending=False).reset_index(drop=True)

    # Top start and end activities
    start_dist = (df_traces["start_activity"].value_counts(normalize=True) * 100).round(2).to_dict()
    end_dist = (df_traces["end_activity"].value_counts(normalize=True) * 100).round(2).to_dict()

    metrics = {
        "total_cases_reconstructed": total_cases,
        "unique_journey_variants": len(variant_summary),
        "start_activities_pct": start_dist,
        "exit_points_pct": end_dist,
        "avg_case_duration_days": round(df_traces["duration_days"].mean(), 1),
        "median_case_duration_days": round(df_traces["duration_days"].median(), 1),
        "top_variants": variant_summary.head(10).to_dict(orient="records")
    }

    return variant_summary, metrics


def compute_directly_follows_graph(df_events: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes Directly-Follows Graph (DFG) transition counts and transition duration statistics.
    Transition: (A -> B) where B directly follows A in a case trace.
    """
    transitions = []

    for case_id, grp in df_events.groupby("case_id"):
        acts = grp["activity"].tolist()
        dts = pd.to_datetime(grp["timestamp"]).tolist()
        district = grp["district"].iloc[0]
        cat = grp["case_category"].iloc[0]

        for i in range(len(acts) - 1):
            src = acts[i]
            tgt = acts[i + 1]
            # Days elapsed
            delta_days = max(0, (dts[i + 1] - dts[i]).days)

            transitions.append({
                "case_id": case_id,
                "from_activity": src,
                "to_activity": tgt,
                "duration_days": delta_days,
                "district": district,
                "case_category": cat
            })

    if not transitions:
        return pd.DataFrame(), {}

    df_trans = pd.DataFrame(transitions)

    # Aggregate overall DFG
    dfg_agg = df_trans.groupby(["from_activity", "to_activity"]).agg(
        transition_count=("case_id", "count"),
        avg_duration_days=("duration_days", "mean"),
        median_duration_days=("duration_days", "median"),
        p90_duration_days=("duration_days", lambda x: np.percentile(x, 90)),
        min_duration_days=("duration_days", "min"),
        max_duration_days=("duration_days", "max")
    ).reset_index()

    dfg_agg["avg_duration_days"] = dfg_agg["avg_duration_days"].round(1)
    dfg_agg["median_duration_days"] = dfg_agg["median_duration_days"].round(1)
    dfg_agg["p90_duration_days"] = dfg_agg["p90_duration_days"].round(1)
    dfg_agg = dfg_agg.sort_values(by="transition_count", ascending=False).reset_index(drop=True)

    # DFG segmented by District (to isolate District 3 and District 6 bottlenecks)
    district_dfg = {}
    for dist, d_grp in df_trans.groupby("district"):
        d_summary = d_grp.groupby(["from_activity", "to_activity"]).agg(
            count=("case_id", "count"),
            avg_days=("duration_days", "mean"),
            median_days=("duration_days", "median")
        ).reset_index()
        d_summary["avg_days"] = d_summary["avg_days"].round(1)
        d_summary["median_days"] = d_summary["median_days"].round(1)
        district_dfg[dist] = d_summary.to_dict(orient="records")

    # DFG segmented by Category
    category_dfg = {}
    for cat, c_grp in df_trans.groupby("case_category"):
        c_summary = c_grp.groupby(["from_activity", "to_activity"]).agg(
            count=("case_id", "count"),
            avg_days=("duration_days", "mean")
        ).reset_index()
        c_summary["avg_days"] = c_summary["avg_days"].round(1)
        category_dfg[cat] = c_summary.to_dict(orient="records")

    summary = {
        "overall_dfg": dfg_agg.to_dict(orient="records"),
        "district_dfg": district_dfg,
        "category_dfg": category_dfg
    }

    return dfg_agg, summary


def discover_pm4py_dfg(df_events: pd.DataFrame) -> Dict[str, Any]:
    """
    Leverages PM4Py process mining engine to discover Directly-Follows Graph (DFG)
    and empirical transition duration performance metrics.
    """
    df_pm = pm4py.format_dataframe(
        df_events.copy(),
        case_id="case_id",
        activity_key="activity",
        timestamp_key="timestamp"
    )

    # PM4Py Frequency DFG
    freq_dfg, start_acts, end_acts = pm4py.discover_dfg(df_pm)

    # PM4Py Performance DFG (durations)
    perf_dfg, _, _ = pm4py.discover_performance_dfg(df_pm)

    freq_transitions = [
        {"from_activity": str(k[0]), "to_activity": str(k[1]), "frequency": int(v)}
        for k, v in freq_dfg.items()
    ]
    freq_transitions = sorted(freq_transitions, key=lambda x: x["frequency"], reverse=True)

    perf_transitions = []
    for k, v in perf_dfg.items():
        if isinstance(v, dict):
            mean_days = round(float(v.get("mean", 0.0)) / 86400.0, 1)
            median_days = round(float(v.get("median", 0.0)) / 86400.0, 1)
            max_days = round(float(v.get("max", 0.0)) / 86400.0, 1)
        else:
            mean_days = round(float(v) / 86400.0, 1)
            median_days = mean_days
            max_days = mean_days

        perf_transitions.append({
            "from_activity": str(k[0]),
            "to_activity": str(k[1]),
            "mean_days": mean_days,
            "median_days": median_days,
            "max_days": max_days
        })
    perf_transitions = sorted(perf_transitions, key=lambda x: x["mean_days"], reverse=True)

    return {
        "start_activities": {str(k): int(v) for k, v in start_acts.items()},
        "end_activities": {str(k): int(v) for k, v in end_acts.items()},
        "frequency_dfg": freq_transitions,
        "performance_dfg": perf_transitions
    }


def run_journey_analysis(data_dir: str = "data") -> Dict[str, Any]:
    """
    Executes complete Case Journey Reconstruction and Process Mining pipeline:
    1. Builds chronological event log from linked entities.
    2. Discovers process variants and throughput times.
    3. Computes performance DFG transition matrix (both Pandas and PM4Py).
    4. Saves event_log.csv, journey_dfg.json, journey_variants.json, journey_pm4py.json, and journey_metrics.json.
    """
    print("Constructing process mining event log from reconstructed case entities...")
    df_events = build_event_log(data_dir=data_dir)
    print(f"Total events in reconstructed process log: {len(df_events)}")
    print(f"Distinct cases with events: {df_events['case_id'].nunique()}")

    # Save Event Log
    event_log_path = os.path.join(data_dir, "event_log.csv")
    df_events.to_csv(event_log_path, index=False)
    print(f"Saved event log to: {event_log_path}")

    # Discover Process Variants
    print("\nDiscovering case journey variants and throughput times...")
    df_variants, variant_metrics = discover_process_variants(df_events)
    print(f"Found {variant_metrics['unique_journey_variants']} distinct journey path variants")

    # Compute Directly-Follows Graph
    print("\nComputing Directly-Follows Graph (DFG) and transition durations...")
    df_dfg, dfg_metrics = compute_directly_follows_graph(df_events)

    # Compute PM4Py DFG
    print("\nExecuting PM4Py process mining discovery...")
    pm4py_metrics = discover_pm4py_dfg(df_events)

    # Save outputs
    variants_path = os.path.join(data_dir, "journey_variants.json")
    with open(variants_path, "w") as f:
        json.dump(variant_metrics, f, indent=2)

    dfg_path = os.path.join(data_dir, "journey_dfg.json")
    with open(dfg_path, "w") as f:
        json.dump(dfg_metrics, f, indent=2)

    pm4py_path = os.path.join(data_dir, "journey_pm4py.json")
    with open(pm4py_path, "w") as f:
        json.dump(pm4py_metrics, f, indent=2)
    print(f"Saved PM4Py process mining metrics to: {pm4py_path}")

    # Overall Journey Summary
    journey_summary = {
        "total_cases": variant_metrics["total_cases_reconstructed"],
        "total_events": len(df_events),
        "avg_duration_days": variant_metrics["avg_case_duration_days"],
        "median_duration_days": variant_metrics["median_case_duration_days"],
        "start_activities_pct": variant_metrics["start_activities_pct"],
        "exit_points_pct": variant_metrics["exit_points_pct"],
        "top_variants": variant_metrics["top_variants"][:5],
        "top_transitions": df_dfg.head(6).to_dict(orient="records") if not df_dfg.empty else [],
        "pm4py_transitions_count": len(pm4py_metrics["frequency_dfg"])
    }

    metrics_path = os.path.join(data_dir, "journey_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(journey_summary, f, indent=2)
    print(f"Saved journey metrics to: {metrics_path}")

    return journey_summary


if __name__ == "__main__":
    summary = run_journey_analysis(data_dir="data")
    print("\n" + "="*70)
    print("PHASE 4: CASE JOURNEY RECONSTRUCTION COMPLETE")
    print("="*70)
    print(f"Total Reconstructed Cases: {summary['total_cases']}")
    print(f"Total Lifecycle Events:    {summary['total_events']}")
    print(f"Average Case Duration:     {summary['avg_duration_days']} days")
    print(f"Median Case Duration:      {summary['median_duration_days']} days")

    print("\nSystem Entry Points (Start Activities):")
    for act, pct in summary["start_activities_pct"].items():
        print(f"  - {act:<22}: {pct:>5.1f}%")

    print("\nSystem Exit / Leak Points (Where cases terminate):")
    for act, pct in summary["exit_points_pct"].items():
        print(f"  - {act:<22}: {pct:>5.1f}%")

    print("\nTop 5 Case Journey Path Variants:")
    for v in summary["top_variants"]:
        print(f"  - [{v['case_count']:>4} cases, {v['share_pct']:>5.1f}%] {v['trace']} (Avg: {v['avg_duration_days']}d)")

    print("\nCore Transition Performance (Directly-Follows Graph):")
    for t in summary["top_transitions"]:
        print(f"  - {t['from_activity']} -> {t['to_activity']}: {t['transition_count']} cases | Avg Delay: {t['avg_duration_days']}d (Median: {t['median_duration_days']}d, P90: {t['p90_duration_days']}d)")
    print("="*70 + "\n")
