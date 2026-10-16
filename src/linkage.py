"""
linkage.py - Privacy-Preserving Record Linkage Engine for Violence-Against-Women Cases.

Links fragmented institutional records (Police FIR, Hospital MLC, Forensic Lab,
Prosecutor, Court) into single-case journey entities WITHOUT a common identifier.

Methodology:
1. Deterministic Normalization:
   - Case standardization, punctuation stripping, title removal (Smt, Mrs, Kumari, etc.)
   - Transliteration normalization (Indian dialectal phonetic variants)
   - Indian surname-first transposition detection and correction
   - Phonetic key generation via Double Metaphone
2. Probabilistic Record Linkage (Fellegi-Sunter model via Splink v5 + DuckDB):
   - Multi-pass blocking rules (district + phonetic surname, district + incident date window, etc.)
   - String distance metrics (Jaro-Winkler) on full name, relative name, locality
   - Date tolerance matching on incident dates
   - Parameter training via Expectation-Maximization (EM) & random sampling (no hardcoded probabilities)
3. Tri-State Decision Thresholding:
   - Auto-accept: match_probability >= 0.85 (with temporal sanity validation)
   - Human-review: 0.50 <= match_probability < 0.85 (saved to data/review_queue.csv)
   - Reject: match_probability < 0.50
4. Graph Clustering:
   - Connected components clustering on validated accepted matches into Case Entities.
"""

import os
import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import jellyfish
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

# Standard Indian surnames for identifying swapped name tokens (e.g. "Sharma Sunita" -> "Sunita Sharma")
KNOWN_SURNAMES = {
    "sharma", "verma", "gupta", "yadav", "singh", "kumar", "devi", "choudhary",
    "shukla", "mishra", "patel", "tiwari", "dubey", "jha", "das", "roy",
    "khan", "ansari", "chauhan", "rathore", "rawat", "joshi", "bhatt", "pandey",
    "chowdhury", "choudhry", "chowdary", "sukla", "yadaw", "goopta", "varma"
}

# Phonetic & Transliteration Normalization Dictionary
TRANSLITERATION_MAP = {
    "suneeta": "sunita", "sunitha": "sunita", "suneetha": "sunita",
    "preya": "priya", "preeya": "priya",
    "puja": "pooja", "poojah": "pooja",
    "aneeta": "anita", "anitha": "anita",
    "kavitha": "kavita", "kaveeta": "kavita",
    "gita": "geeta", "geetah": "geeta",
    "nilam": "neelam", "nelam": "neelam",
    "mina": "meena", "meenah": "meena",
    "laxmi": "lakshmi", "laksmi": "lakshmi",
    "bandana": "vandana", "wandana": "vandana",
    "dipa": "deepa", "deepah": "deepa",
    "rmesh": "ramesh", "ramesha": "ramesh", "rameesh": "ramesh",
    "mohd": "mohammad", "mohamad": "mohammad", "mohammed": "mohammad", "md": "mohammad",
    "chowdhury": "choudhary", "choudhry": "choudhary", "chowdary": "choudhary",
    "sukla": "shukla", "shookla": "shukla",
    "misra": "mishra", "mishar": "mishra",
    "yadaw": "yadav", "jadav": "yadav",
    "varma": "verma", "burman": "verma",
    "goopta": "gupta", "guptha": "gupta",
    "suresha": "suresh", "sooresh": "suresh",
    "rajiesh": "rajesh", "rajesha": "rajesh",
    "deenesh": "dinesh", "dinesha": "dinesh"
}

TITLE_REGEX = re.compile(
    r"^(smt|smt\.|srimati|shrimati|mrs|mrs\.|ms|ms\.|miss|km|kumari|shri|sh\.|mr|mr\.|dr|dr\.)\s+",
    re.IGNORECASE
)


def normalize_name(raw_name: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Cleans, strips titles, normalizes transliterations, corrects name transposition,
    and computes Double Metaphone phonetic representations.

    Returns: (norm_full_name, norm_first_name, norm_last_name, phonetic_first, phonetic_last)
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, None, None, None, None

    s = raw_name.strip().lower()
    s = TITLE_REGEX.sub("", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [t for t in s.split() if t]

    if not tokens:
        return None, None, None, None, None

    norm_tokens = [TRANSLITERATION_MAP.get(t, t) for t in tokens]

    if len(norm_tokens) >= 2 and norm_tokens[0] in KNOWN_SURNAMES and norm_tokens[1] not in KNOWN_SURNAMES:
        norm_tokens[0], norm_tokens[1] = norm_tokens[1], norm_tokens[0]

    first_name = norm_tokens[0] if len(norm_tokens) >= 1 else None
    last_name = norm_tokens[-1] if len(norm_tokens) >= 2 else None
    full_name = " ".join(norm_tokens)

    p_first = jellyfish.metaphone(first_name) if first_name and len(first_name) > 1 else None
    p_last = jellyfish.metaphone(last_name) if last_name and len(last_name) > 1 else None

    p_first = p_first if p_first else None
    p_last = p_last if p_last else None
    last_name = last_name if last_name else None

    return full_name, first_name, last_name, p_first, p_last


def normalize_locality(raw_loc: Any) -> Optional[str]:
    """Standardizes address and locality strings."""
    if not isinstance(raw_loc, str) or not raw_loc.strip():
        return None

    s = raw_loc.strip().lower()
    abbrevs = {
        r"\bngr\b": "nagar",
        r"\bsec\b": "sector",
        r"\brd\b": "road",
        r"\bcol\b": "colony",
        r"\bar\b": "area",
        r"\bmkt\b": "market",
        r"\bvill\b": "village",
        r"\bward\s*\d+\b": "",
        r"\bplot\s*no\b": "",
        r"\bgali\s*no\b": "",
        r"\bnear\b": "",
        r"\bopp\b": "",
        r"\bopposite\b": "",
        r"\bbehind\b": ""
    }
    for pat, rep in abbrevs.items():
        s = re.sub(pat, rep, s)

    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t]
    res = " ".join(tokens)
    return res if res else None


def load_and_harmonize_records(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads all 5 agency tables and creates a unified standardized view for record linkage.
    """
    records = []

    # 1. Police FIR
    fir_path = os.path.join(data_dir, "police_fir.csv")
    if os.path.exists(fir_path):
        df_fir = pd.read_csv(fir_path)
        for _, row in df_fir.iterrows():
            fir_no = str(row["fir_number"])
            full, first, last, p_first, p_last = normalize_name(row.get("complainant_name"))
            rel_full, rel_first, _, p_rel, _ = normalize_name(row.get("father_or_husband"))
            loc = normalize_locality(row.get("locality"))
            inc_date = str(row["incident_date"]) if pd.notna(row.get("incident_date")) and str(row["incident_date"]).strip() else None

            records.append({
                "unique_id": f"police_fir__{fir_no}",
                "source_agency": "police_fir",
                "record_id": fir_no,
                "district": str(row["district"]),
                "norm_full_name": full,
                "norm_first_name": first,
                "norm_last_name": last,
                "phonetic_first": p_first,
                "phonetic_last": p_last,
                "norm_relative_name": rel_full,
                "phonetic_relative": p_rel,
                "norm_locality": loc,
                "incident_date": inc_date,
                "record_date": str(row.get("fir_date")) if pd.notna(row.get("fir_date")) else None
            })

    # 2. Hospital MLC
    mlc_path = os.path.join(data_dir, "hospital_mlc.csv")
    if os.path.exists(mlc_path):
        df_mlc = pd.read_csv(mlc_path)
        for _, row in df_mlc.iterrows():
            mlc_no = str(row["mlc_number"])
            full, first, last, p_first, p_last = normalize_name(row.get("patient_name"))
            rel_full, rel_first, _, p_rel, _ = normalize_name(row.get("guardian_name"))
            loc = normalize_locality(row.get("residential_locality"))
            inc_date = str(row["incident_date"]) if pd.notna(row.get("incident_date")) and str(row["incident_date"]).strip() else None

            records.append({
                "unique_id": f"hospital_mlc__{mlc_no}",
                "source_agency": "hospital_mlc",
                "record_id": mlc_no,
                "district": str(row["district"]),
                "norm_full_name": full,
                "norm_first_name": first,
                "norm_last_name": last,
                "phonetic_first": p_first,
                "phonetic_last": p_last,
                "norm_relative_name": rel_full,
                "phonetic_relative": p_rel,
                "norm_locality": loc,
                "incident_date": inc_date,
                "record_date": str(row.get("examination_date")) if pd.notna(row.get("examination_date")) else None
            })

    # 3. Forensic Lab (FSL)
    fsl_path = os.path.join(data_dir, "forensic_lab.csv")
    if os.path.exists(fsl_path):
        df_fsl = pd.read_csv(fsl_path)
        for _, row in df_fsl.iterrows():
            fsl_no = str(row["fsl_case_number"])
            full, first, last, p_first, p_last = normalize_name(row.get("subject_name"))
            rel_full, rel_first, _, p_rel, _ = normalize_name(row.get("guardian_name"))
            rec_date = str(row["sample_receipt_date"]) if pd.notna(row.get("sample_receipt_date")) and str(row["sample_receipt_date"]).strip() else None

            records.append({
                "unique_id": f"forensic_lab__{fsl_no}",
                "source_agency": "forensic_lab",
                "record_id": fsl_no,
                "district": str(row["district"]),
                "norm_full_name": full,
                "norm_first_name": first,
                "norm_last_name": last,
                "phonetic_first": p_first,
                "phonetic_last": p_last,
                "norm_relative_name": rel_full,
                "phonetic_relative": p_rel,
                "norm_locality": None,
                "incident_date": rec_date,
                "record_date": str(row.get("report_dispatch_date")) if pd.notna(row.get("report_dispatch_date")) else None
            })

    # 4. Prosecutor
    pros_path = os.path.join(data_dir, "prosecutor.csv")
    if os.path.exists(pros_path):
        df_pros = pd.read_csv(pros_path)
        for _, row in df_pros.iterrows():
            pros_no = str(row["prosecution_file_no"])
            full, first, last, p_first, p_last = normalize_name(row.get("victim_name"))
            rel_full, rel_first, _, p_rel, _ = normalize_name(row.get("parent_spouse_name"))
            loc = normalize_locality(row.get("address"))
            inc_date = str(row["incident_date"]) if pd.notna(row.get("incident_date")) and str(row["incident_date"]).strip() else None

            records.append({
                "unique_id": f"prosecutor__{pros_no}",
                "source_agency": "prosecutor",
                "record_id": pros_no,
                "district": str(row["district"]),
                "norm_full_name": full,
                "norm_first_name": first,
                "norm_last_name": last,
                "phonetic_first": p_first,
                "phonetic_last": p_last,
                "norm_relative_name": rel_full,
                "phonetic_relative": p_rel,
                "norm_locality": loc,
                "incident_date": inc_date,
                "record_date": str(row.get("chargesheet_date")) if pd.notna(row.get("chargesheet_date")) else None
            })

    # 5. Court
    court_path = os.path.join(data_dir, "court.csv")
    if os.path.exists(court_path):
        df_court = pd.read_csv(court_path)
        for _, row in df_court.iterrows():
            cnr_no = str(row["cnr_number"])
            full, first, last, p_first, p_last = normalize_name(row.get("petitioner_victim"))
            rel_full, rel_first, _, p_rel, _ = normalize_name(row.get("father_or_husband"))
            loc = normalize_locality(row.get("residence_area"))
            inc_date = str(row["date_of_incident"]) if pd.notna(row.get("date_of_incident")) and str(row["date_of_incident"]).strip() else None

            records.append({
                "unique_id": f"court__{cnr_no}",
                "source_agency": "court",
                "record_id": cnr_no,
                "district": str(row["district"]),
                "norm_full_name": full,
                "norm_first_name": first,
                "norm_last_name": last,
                "phonetic_first": p_first,
                "phonetic_last": p_last,
                "norm_relative_name": rel_full,
                "phonetic_relative": p_rel,
                "norm_locality": loc,
                "incident_date": inc_date,
                "record_date": str(row.get("verdict_date")) if pd.notna(row.get("verdict_date")) else None
            })

    df_harmonized = pd.DataFrame(records)
    return df_harmonized


def build_splink_settings() -> SettingsCreator:
    """
    Configures Fellegi-Sunter model in Splink:
    - Balanced multi-pass blocking rules (district + surname, district + date, district + relative)
    - Jaro-Winkler comparisons on names, relative names, locality
    - Incident date difference with tolerance thresholds
    - Exact match on district & phonetic keys
    """
    blocking_rules = [
        block_on("district", "phonetic_last"),
        block_on("district", "incident_date"),
        block_on("district", "phonetic_relative"),
        block_on("district", "phonetic_first", "norm_locality")
    ]

    comparisons = [
        cl.ExactMatch("district"),
        cl.JaroWinklerAtThresholds("norm_full_name", [0.90, 0.75]),
        cl.ExactMatch("phonetic_first"),
        cl.ExactMatch("phonetic_last"),
        cl.JaroWinklerAtThresholds("norm_relative_name", [0.88, 0.70]),
        cl.AbsoluteDateDifferenceAtThresholds(
            "incident_date",
            input_is_string=True,
            metrics=["day", "day"],
            thresholds=[2, 10],
            datetime_format="%Y-%m-%d"
        ),
        cl.JaroWinklerAtThresholds("norm_locality", [0.85, 0.65])
    ]

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=True
    )
    return settings


def cluster_connected_components(
    nodes: List[str],
    edges: List[Tuple[str, str]]
) -> Dict[str, str]:
    """
    Computes connected components given nodes and accepted edges.
    Returns: mapping of node_id -> entity_id (e.g. 'ENT-00001')
    """
    parent = {n: n for n in nodes}

    def find(x):
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        for node in path:
            parent[node] = x
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for u, v in edges:
        if u in parent and v in parent:
            union(u, v)

    # Group components
    components = {}
    for n in nodes:
        root = find(n)
        if root not in components:
            components[root] = []
        components[root].append(n)

    # Assign canonical entity IDs sorted by size then root name
    sorted_roots = sorted(components.keys(), key=lambda r: (-len(components[r]), r))
    node_to_entity = {}
    for idx, r in enumerate(sorted_roots):
        ent_id = f"ENT-{idx+1:05d}"
        for member in components[r]:
            node_to_entity[member] = ent_id

    return node_to_entity


def run_linkage_pipeline(
    data_dir: str = "data",
    high_threshold: float = 0.85,
    low_threshold: float = 0.50,
    max_training_pairs: int = 50000
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end record linkage pipeline:
    1. Loads and harmonizes all agency records
    2. Trains Fellegi-Sunter model via EM and random sampling (no hardcoded probabilities)
    3. Predicts match probabilities for candidate pairs
    4. Categorizes into Auto-Accept, Human Review Queue, Reject
    5. Clusters auto-accepted matches into Case Entities
    6. Saves linked_pairs.csv, review_queue.csv, and case_entities.csv
    """
    print("Harmonizing agency tables into unified schema...")
    df_harmonized = load_and_harmonize_records(data_dir=data_dir)
    print(f"Total agency records loaded: {len(df_harmonized)}")

    agency_counts = df_harmonized["source_agency"].value_counts().to_dict()
    for ag, cnt in agency_counts.items():
        print(f"  - {ag:<15}: {cnt:>5} records")

    db_api = DuckDBAPI()
    splink_df = db_api.register(df_harmonized, dataset_display_name="harmonized_records")

    settings = build_splink_settings()
    linker = Linker(splink_df, settings)

    # ---------------------------------------------------------
    # EM TRAINING PHASE (No hardcoded match probabilities!)
    # ---------------------------------------------------------
    print("\nEstimating u-probabilities via random sampling...")
    linker.training.estimate_u_using_random_sampling(max_pairs=max_training_pairs)

    print("Estimating m-probabilities via Expectation-Maximization on blocking rules...")
    # Session 1: District + Incident Date (trains names, phonetic keys, relative name, locality)
    em_rule_1 = block_on("district", "incident_date")
    linker.training.estimate_parameters_using_expectation_maximisation(em_rule_1)

    # Session 2: District + Phonetic First + Phonetic Last (trains incident date, relative name, locality)
    em_rule_2 = block_on("district", "phonetic_first", "phonetic_last")
    linker.training.estimate_parameters_using_expectation_maximisation(em_rule_2)

    # Estimate prior match probability
    linker.training.estimate_probability_two_random_records_match(em_rule_2, recall=0.70)

    # ---------------------------------------------------------
    # PREDICTION PHASE
    # ---------------------------------------------------------
    print("\nGenerating pairwise predictions across blocking rules...")
    df_predict = linker.inference.predict()
    predictions_df = df_predict.as_pandas_dataframe()
    print(f"Total candidate pairs evaluated: {len(predictions_df)}")

    # Assign initial link decisions
    conditions = [
        predictions_df["match_probability"] >= high_threshold,
        predictions_df["match_probability"] >= low_threshold
    ]
    choices = ["AUTO_ACCEPT", "HUMAN_REVIEW"]
    predictions_df["link_decision"] = np.select(conditions, choices, default="REJECT")

    # Vectorized temporal sanity guard:
    # Demote auto-accepts that violate domain temporal bounds (incident dates > 20 days apart) to human review
    if "incident_date_l" in predictions_df.columns and "incident_date_r" in predictions_df.columns:
        dt_l = pd.to_datetime(predictions_df["incident_date_l"], errors="coerce")
        dt_r = pd.to_datetime(predictions_df["incident_date_r"], errors="coerce")
        day_diff = (dt_l - dt_r).abs().dt.days
        has_large_date_conflict = day_diff.notna() & (day_diff > 20)
        auto_mask = predictions_df["link_decision"] == "AUTO_ACCEPT"
        predictions_df.loc[auto_mask & has_large_date_conflict, "link_decision"] = "HUMAN_REVIEW"

    decision_counts = predictions_df["link_decision"].value_counts().to_dict()
    auto_accept_count = decision_counts.get("AUTO_ACCEPT", 0)
    human_review_count = decision_counts.get("HUMAN_REVIEW", 0)
    reject_count = decision_counts.get("REJECT", 0)

    print("\nPairwise Decision Distribution:")
    print(f"  - AUTO_ACCEPT  (P >= {high_threshold}): {auto_accept_count:>6} pairs")
    print(f"  - HUMAN_REVIEW ({low_threshold} <= P < {high_threshold}): {human_review_count:>6} pairs")
    print(f"  - REJECT       (P < {low_threshold}): {reject_count:>6} pairs")

    # Save Review Queue
    review_queue_df = predictions_df[predictions_df["link_decision"] == "HUMAN_REVIEW"].copy()
    review_queue_cols = [
        "unique_id_l", "unique_id_r",
        "norm_full_name_l", "norm_full_name_r",
        "district_l", "district_r",
        "incident_date_l", "incident_date_r",
        "norm_relative_name_l", "norm_relative_name_r",
        "norm_locality_l", "norm_locality_r",
        "match_probability", "match_weight"
    ]
    review_queue_cols = [c for c in review_queue_cols if c in review_queue_df.columns]
    review_queue_df = review_queue_df[review_queue_cols]
    review_queue_df["review_status"] = "PENDING_INVESTIGATION"

    review_path = os.path.join(data_dir, "review_queue.csv")
    review_queue_df.to_csv(review_path, index=False)
    print(f"\nSaved human review queue to: {review_path} ({len(review_queue_df)} uncertain pairs)")

    # Save Linked Pairs
    pairs_path = os.path.join(data_dir, "linked_pairs.csv")
    predictions_df.to_csv(pairs_path, index=False)
    print(f"Saved all candidate pairs to: {pairs_path}")

    # ---------------------------------------------------------
    # CLUSTERING PHASE (Case Entity Reconstruction)
    # ---------------------------------------------------------
    print(f"\nClustering records into Case Entities at acceptance threshold {high_threshold}...")
    accepted_pairs = predictions_df[predictions_df["link_decision"] == "AUTO_ACCEPT"]
    edges = list(zip(accepted_pairs["unique_id_l"], accepted_pairs["unique_id_r"]))
    all_nodes = df_harmonized["unique_id"].tolist()

    node_to_entity = cluster_connected_components(all_nodes, edges)

    # Attach entity_id to harmonized records
    clusters_df = df_harmonized.copy()
    clusters_df["entity_id"] = clusters_df["unique_id"].map(node_to_entity)

    # Save Case Entities table
    entities_path = os.path.join(data_dir, "case_entities.csv")
    clusters_df.to_csv(entities_path, index=False)
    print(f"Saved clustered case entities to: {entities_path}")

    total_entities = clusters_df["entity_id"].nunique()
    cluster_sizes = clusters_df.groupby("entity_id")["unique_id"].count()
    size_dist = cluster_sizes.value_counts().sort_index().to_dict()

    print(f"\nTotal Distinct Case Entities Reconstructed: {total_entities}")
    print("Entity Size Distribution (Records per Case Entity):")
    for size, cnt in sorted(size_dist.items()):
        print(f"  - {size}-agency record(s) : {cnt:>5} entities")

    summary = {
        "total_records_harmonized": len(df_harmonized),
        "agency_record_counts": agency_counts,
        "total_pairs_evaluated": len(predictions_df),
        "auto_accept_pairs": auto_accept_count,
        "human_review_pairs": human_review_count,
        "reject_pairs": reject_count,
        "total_case_entities": total_entities,
        "entity_size_distribution": size_dist
    }

    return summary


if __name__ == "__main__":
    summary = run_linkage_pipeline(
        data_dir="data",
        high_threshold=0.85,
        low_threshold=0.50,
        max_training_pairs=50000
    )
    print("\n" + "="*70)
    print("PHASE 2: RECORD LINKAGE COMPLETE")
    print("="*70)
