"""
src/case_explorer.py - End-to-End Case Explorer Engine for CaseLens.

Judicial case exploration engine designed for judges, prosecutors, and policy analysts.
Allows inspecting real end-to-end multi-agency trajectories from the generated dataset
and curated judicial benchmarks.

Key Features & Guarantees:
1. Curated benchmark cases covering:
   - Clean match (full 5-stage agreement)
   - Spelling variants (dialectal & phonetic transliteration invariance)
   - Uncertain pair sent to human review queue (0.50 <= P < 0.85)
   - Similar names correctly kept apart (false-link protection in same district)
   - A missed link (fragmented records / severe corruption leading to broken journey)
   - District 3 forensic delay (extreme FSL turnaround latency)
   - District 6 chargesheet drop-off (investigation completed, chargesheet never filed)
2. Random case retrieval from the 5,000 generated synthetic multi-agency cases.
3. Fragmented source records displayed exactly as they appear in each agency system
   (different IDs, differing spellings, ages, and dates, with no common identifier).
4. Pairwise probabilistic match scoring with per-field log-odds reasons.
5. Reconstructed journey timeline with days between stages and drop/stall point identification.
6. Server-side role enforcement:
   - Viewers see masked names (e.g. S***a S***a).
   - Analysts & Admins see pseudonymised IDs (e.g. ANON-A1B2).
7. Read-only with zero data side effects: never writes or stores anything to disk.
"""

import os
import re
import hashlib
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import jellyfish

from src.case_simulator import score_case_pair, SCENARIOS
from src.linkage import normalize_name, normalize_locality


# ==============================================================================
# 1. Privacy Masking & Pseudonymisation Utilities (Server-Side RBAC)
# ==============================================================================

def mask_name(name: Optional[str]) -> str:
    """
    Masks personal names for Viewer role (e.g. 'Sunita Sharma' -> 'S***a S***a').
    Preserves first and last letter where possible, masking internal characters with asterisks.
    """
    if not name or not str(name).strip() or str(name).strip().lower() in ["nan", "none", "null"]:
        return "Not Disclosed"
    parts = str(name).strip().split()
    masked = []
    for p in parts:
        clean_p = re.sub(r"[^\w]", "", p)
        if len(clean_p) <= 2:
            masked.append(clean_p[0] + "*" if len(clean_p) > 0 else "*")
        else:
            masked.append(clean_p[0] + ("*" * (len(clean_p) - 2)) + clean_p[-1])
    return " ".join(masked)


def pseudonymise_name(name: Optional[str], prefix: str = "ANON") -> str:
    """
    Generates a deterministic pseudonymised identifier for Analyst/Admin role
    (e.g. 'Sunita Sharma' -> 'ANON-SUBJ-8B3E').
    """
    if not name or not str(name).strip() or str(name).strip().lower() in ["nan", "none", "null"]:
        return "ANON-UNRECORDED"
    token = hashlib.sha256(str(name).strip().lower().encode("utf-8")).hexdigest()[:6].upper()
    return f"{prefix}-{token}"


def apply_role_privacy(text: Optional[str], role: str, prefix: str = "ANON") -> str:
    """Applies role-appropriate privacy transform to a name value."""
    if not text:
        return "N/A"
    r = str(role or "viewer").strip().lower()
    if r == "viewer":
        return mask_name(text)
    return pseudonymise_name(text, prefix=prefix)


# ==============================================================================
# 2. Curated Judicial Examples Definitions
# ==============================================================================

CURATED_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "clean_match": {
        "id": "clean_match",
        "title": "1. Clean Match: Full 5-Stage Institutional Journey",
        "category": "Baseline Linkage",
        "district": "D2_Chandranagar",
        "description": "Standard ideal case across Police FIR, Hospital MLC, Forensic Lab, Prosecution, and District Court. Records align with high probabilistic confidence across all stages.",
        "one_line_verdict": "Fully Linked Journey: All 5 institutional milestones successfully reached in statutory compliance.",
        "verdict_type": "success",
        "why": "High agreement across full name, Double Metaphone phonetic tokens, guardian name, district, and dates. Pairwise posterior probabilities exceed 0.95 across all agency transitions.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2023-D2-0001",
                "name": "Shashi Choudhary",
                "guardian": "Sanjay Choudhary",
                "age": 30,
                "district": "D2_Chandranagar",
                "locality": "Indira Market",
                "incident_date": "2023-04-14",
                "record_date": "2023-04-17",
                "details": "Domestic Violence complaint filed at PS Chandranagar Central"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2023-130-0001",
                "name": "Shashi Choudhary",
                "guardian": "Sanjay Choudhary",
                "age": 30,
                "district": "D2_Chandranagar",
                "locality": "Behind Water Tank, Indira Mkt",
                "incident_date": "2023-04-15",
                "record_date": "2023-04-16",
                "details": "Medico-legal examination completed within 24h at District Civil Hospital"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2023-D2-0001",
                "name": "Shashi Choudhary",
                "guardian": "Chowdary Sanjay",
                "age": 30,
                "district": "D2_Chandranagar",
                "locality": None,
                "incident_date": "2023-04-15",
                "record_date": "2023-05-26",
                "details": "Biological Serology report dispatched in 36 days by Regional FSL"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PP-CHG-2023-D2-0001",
                "name": "Choudhary Shashi",
                "guardian": "Sanjay Choudhary",
                "age": 30,
                "district": "D2_Chandranagar",
                "locality": "Indira Market, Chandranagar",
                "incident_date": "2023-04-15",
                "record_date": "2023-07-25",
                "details": "Section 173 CrPC chargesheet submitted by Public Prosecutor Office"
            },
            {
                "agency": "District Court",
                "agency_code": "court",
                "record_id": "CNR-D2-2024-00001",
                "name": "Shashi Choudhary",
                "guardian": "Sanjay Choudhary",
                "age": 31,
                "district": "D2_Chandranagar",
                "locality": "Shop 12, Indira Mkt",
                "incident_date": "2023-04-15",
                "record_date": "2024-04-18",
                "details": "Conviction recorded at District Sessions Court Chandranagar (367 days total)"
            }
        ]
    },

    "spelling_variants": {
        "id": "spelling_variants",
        "title": "2. Spelling Variants: Phonetic & Dialectal Invariance",
        "category": "Algorithmic Robustness",
        "district": "D3_Rudrapur",
        "description": "Examines dialectal phonetic spelling differences (Suneeta / Sunita, Rmesh / Ramesh, Mohd / Mohammad) across disparate agencies without shared IDs.",
        "one_line_verdict": "Successfully Linked: Normalization layer resolved transliteration differences (P >= 0.94).",
        "verdict_type": "success",
        "why": "Double Metaphone phonetic encoding and Indian transliteration tables harmonize variations ('Suneeta' == 'Sunita', 'Rmesh' == 'Ramesh'), maintaining high match weights despite surface character discrepancies.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-1102",
                "name": "Suneeta Sharma",
                "guardian": "Rmesh Sharma",
                "age": 24,
                "district": "D3_Rudrapur",
                "locality": "Gandhi Colony",
                "incident_date": "2024-02-14",
                "record_date": "2024-02-15",
                "details": "Sec 354 IPC registered at Rudrapur Police Station"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-1088",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "age": 24,
                "district": "D3_Rudrapur",
                "locality": "Gandhi Col",
                "incident_date": "2024-02-14",
                "record_date": "2024-02-15",
                "details": "Medical exam conducted under standardized clinical spelling"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PP-CHG-2024-D3-0499",
                "name": "Sunita Sharma",
                "guardian": "Ramesh Sharma",
                "age": 24,
                "district": "D3_Rudrapur",
                "locality": "Gandhi Colony",
                "incident_date": "2024-02-14",
                "record_date": "2024-04-05",
                "details": "Chargesheet submitted to Magistrate Court in 49 days"
            }
        ]
    },

    "uncertain_review": {
        "id": "uncertain_review",
        "title": "3. Uncertain Pair: Routed to Human Review Queue",
        "category": "Human-in-the-Loop Safeguard",
        "district": "D4_Kalyanpur",
        "description": "Candidate pair falls into the uncertainty band (0.50 <= P < 0.85). System holds link for human adjudicator review rather than blindly accepting or discarding.",
        "one_line_verdict": "Held for Adjudication: Confidence score 0.656 routed to confidential Human Review Queue.",
        "verdict_type": "review",
        "why": "Records share district ('D4_Kalyanpur') and surname token ('Sharma'), but first names have phonetic divergence ('Deepika' vs 'Deepa'), guardian is omitted in FIR, and reported incident dates differ by 17 days. Posterior probability sits at 0.656.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2023-D4-3367",
                "name": "Deepika Sharma",
                "guardian": None,
                "age": 28,
                "district": "D4_Kalyanpur",
                "locality": None,
                "incident_date": "2023-06-25",
                "record_date": "2023-06-26",
                "details": "FIR logged with missing guardian name"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D4-4555",
                "name": "Deepa Sharma",
                "guardian": "Ram Lal",
                "age": 28,
                "district": "D4_Kalyanpur",
                "locality": None,
                "incident_date": "2023-07-12",
                "record_date": "2023-08-30",
                "details": "FSL requisition with 17-day incident recall shift"
            }
        ]
    },

    "similar_names_kept_apart": {
        "id": "similar_names_kept_apart",
        "title": "4. False-Link Protection: Similar Names Correctly Kept Apart",
        "category": "False Positive Defense",
        "district": "D3_Rudrapur",
        "description": "Two distinct individuals sharing a common first name ('Sunita') in the same district must never be mistakenly merged into one survivor trajectory.",
        "one_line_verdict": "Definitively Disconnected: Conflicting guardians and 194-day date divergence triggered absolute rejection.",
        "verdict_type": "danger",
        "why": "Despite identical first name and district, surname tokens ('Devi' vs 'Sharma'), guardian names ('Ramesh Chand' vs 'Suresh Kumar'), localities, and dates (194 days apart) produce heavy negative weights dropping score to P < 0.05.",
        "records": [
            {
                "agency": "Police FIR (Person A)",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-0210",
                "name": "Sunita Devi",
                "guardian": "Ramesh Chand",
                "age": 26,
                "district": "D3_Rudrapur",
                "locality": "Model Town",
                "incident_date": "2024-02-10",
                "record_date": "2024-02-11",
                "details": "Case registered under Section 376 IPC"
            },
            {
                "agency": "Hospital MLC (Person B)",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-0940",
                "name": "Sunita Sharma",
                "guardian": "Suresh Kumar",
                "age": 34,
                "district": "D3_Rudrapur",
                "locality": "Raja Bazar",
                "incident_date": "2024-08-22",
                "record_date": "2024-08-23",
                "details": "Unrelated medical examination in same district 6 months later"
            }
        ]
    },

    "missed_link": {
        "id": "missed_link",
        "title": "5. Missed Link: Data Fragmentation Caused False Negative",
        "category": "Pipeline Sensitivity",
        "district": "D1_Suryapur",
        "description": "Illustrates how missing institutional metadata (omitted guardian, unstandardized locality, and 12-day date recall shift) can cause an algorithm to miss linking a valid trial record.",
        "one_line_verdict": "Link Missed (False Negative): Missing guardian and large date shift dropped score below 0.50.",
        "verdict_type": "warning",
        "why": "The Police FIR recorded only single name 'Priya' without surname or guardian. When Court petition was filed 180 days later under 'Priya Kumari' with a 12-day shifted incident date, the link weight fell below 0.50 threshold.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D1-0988",
                "name": "Priya",
                "guardian": None,
                "age": 22,
                "district": "D1_Suryapur",
                "locality": "Shivaji Nagar",
                "incident_date": "2024-03-01",
                "record_date": "2024-03-02",
                "details": "Complainant arrived alone; guardian field left unpopulated"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D1-0902",
                "name": "Priya Kumari",
                "guardian": "Mahesh Kumar",
                "age": 22,
                "district": "D1_Suryapur",
                "locality": "Shivaji Ngr",
                "incident_date": "2024-03-01",
                "record_date": "2024-03-02",
                "details": "Medical examination completed"
            },
            {
                "agency": "District Court",
                "agency_code": "court",
                "record_id": "CNR-D1-2024-00412",
                "name": "Priya Verma",
                "guardian": "M. Kumar",
                "age": 22,
                "district": "D1_Suryapur",
                "locality": "Ward 7",
                "incident_date": "2024-03-14",
                "record_date": "2024-10-18",
                "details": "Trial record divergent in reported incident date (+13 days) and surname"
            }
        ]
    },

    "district_3_forensic_delay": {
        "id": "district_3_forensic_delay",
        "title": "6. District 3: Severe Forensic Laboratory Latency Bottleneck",
        "category": "Systemic Bottleneck",
        "district": "D3_Rudrapur",
        "description": "Case reached trial, but forensic laboratory analysis suffered extreme turnaround latency (276 days), breaching the 30-day statutory benchmark and stalling the justice pipeline.",
        "one_line_verdict": "Institutional Delay Flagged: Forensic analysis delayed chargesheet filing by 276 days in District 3.",
        "verdict_type": "warning",
        "why": "All 5 agency records link with high confidence (P > 0.95), but process mining analysis reveals a 276-day gap between specimen receipt and FSL report dispatch (9.2x statutory benchmark breach).",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D3-0055",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "age": 27,
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2024-01-11",
                "details": "Sec 376 IPC FIR registered promptly"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D3-0048",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "age": 27,
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2024-01-11",
                "details": "Forensic evidence kit secured within 24 hours"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D3-0012",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "age": 27,
                "district": "D3_Rudrapur",
                "locality": None,
                "incident_date": "2024-01-10",
                "record_date": "2024-10-12",
                "details": "FSL report stalled in backlog; dispatched after 274 days!"
            },
            {
                "agency": "Prosecution Chargesheet",
                "agency_code": "prosecutor",
                "record_id": "PP-CHG-2024-D3-0801",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "age": 27,
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2024-11-04",
                "details": "Chargesheet delayed awaiting delayed FSL chemical analysis"
            },
            {
                "agency": "District Court",
                "agency_code": "court",
                "record_id": "CNR-D3-2025-00502",
                "name": "Rekha Yadav",
                "guardian": "Jagdish Yadav",
                "age": 28,
                "district": "D3_Rudrapur",
                "locality": "Industrial Area",
                "incident_date": "2024-01-10",
                "record_date": "2025-05-18",
                "details": "Trial completed 493 days after incident due to FSL bottleneck"
            }
        ]
    },

    "district_6_chargesheet_drop": {
        "id": "district_6_chargesheet_drop",
        "title": "7. District 6: Police-to-Prosecution Chargesheet Drop-Off",
        "category": "Policy Attrition Leak",
        "district": "D6_Shivgarh",
        "description": "Survivor reported incident and underwent medical/forensic examination, but investigation stalled and no chargesheet was ever filed, cutting off survivor access to trial.",
        "one_line_verdict": "Critical Policy Leak: Investigation closed after FSL; no chargesheet filed in District 6.",
        "verdict_type": "warning",
        "why": "Records up to milestone 3 link with P > 0.95, but trajectory terminates abruptly. Over 280 days have elapsed without chargesheet filing, violating CrPC Section 173 60-day mandate.",
        "records": [
            {
                "agency": "Police FIR",
                "agency_code": "police_fir",
                "record_id": "FIR-2024-D6-0144",
                "name": "Meena Kumari",
                "guardian": "Deepak Kumar",
                "age": 23,
                "district": "D6_Shivgarh",
                "locality": "Subhash Nagar",
                "incident_date": "2024-01-15",
                "record_date": "2024-01-16",
                "details": "FIR registered at Shivgarh Central PS"
            },
            {
                "agency": "Hospital MLC",
                "agency_code": "hospital_mlc",
                "record_id": "MLC-2024-D6-0129",
                "name": "Meena Kumari",
                "guardian": "Deepak Kumar",
                "age": 23,
                "district": "D6_Shivgarh",
                "locality": "Subhash Nagar",
                "incident_date": "2024-01-15",
                "record_date": "2024-01-16",
                "details": "Medical examination completed at Sub-District Hospital"
            },
            {
                "agency": "Forensic Lab (FSL)",
                "agency_code": "forensic_lab",
                "record_id": "FSL-2024-D6-0088",
                "name": "Meena Kumari",
                "guardian": "Deepak Kumar",
                "age": 23,
                "district": "D6_Shivgarh",
                "locality": None,
                "incident_date": "2024-01-15",
                "record_date": "2024-02-12",
                "details": "Biological evidence dispatched to police IO within 28 days"
            }
        ]
    }
}


# ==============================================================================
# 3. Case Data Loader Service (In-Memory Fast Caching)
# ==============================================================================

class CaseExplorerService:
    """
    Loads synthetic case trajectories from generated CSVs and curated examples.
    Calculates pairwise Fellegi-Sunter scoring, journey timeline with elapsed days,
    and applies role-based masking (viewer) or pseudonymisation (analyst/admin).
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Workspace root relative
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base, "data")
        self.data_dir = data_dir
        self._cache: Dict[str, pd.DataFrame] = {}
        self._gt_cases: Optional[List[str]] = None

    def _get_df(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            p = os.path.join(self.data_dir, f"{name}.csv")
            if os.path.exists(p):
                self._cache[name] = pd.read_csv(p)
            else:
                self._cache[name] = pd.DataFrame()
        return self._cache[name]

    def get_all_case_ids(self) -> List[str]:
        if self._gt_cases is None:
            gt = self._get_df("ground_truth")
            if "case_id" in gt.columns:
                self._gt_cases = gt["case_id"].dropna().astype(str).tolist()
            else:
                self._gt_cases = []
        return self._gt_cases

    def get_curated_examples_summary(self) -> List[Dict[str, Any]]:
        """Returns metadata for all 7 curated examples for selector rendering."""
        res = []
        for k, v in CURATED_EXAMPLES.items():
            res.append({
                "id": v["id"],
                "title": v["title"],
                "category": v.get("category", "Curated Example"),
                "district": v.get("district", "N/A"),
                "description": v.get("description", ""),
                "one_line_verdict": v.get("one_line_verdict", ""),
                "verdict_type": v.get("verdict_type", "neutral"),
                "total_records": len(v.get("records", []))
            })
        return res

    def get_random_case_id(self) -> str:
        all_ids = self.get_all_case_ids()
        if all_ids:
            return random.choice(all_ids)
        return "clean_match"

    def load_case_from_dataset(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Extracts fragmented multi-agency source records for a case_id from generated CSVs."""
        gt = self._get_df("ground_truth")
        if gt.empty or "case_id" not in gt.columns:
            return None

        m = gt[gt["case_id"] == case_id]
        if m.empty:
            return None

        row = m.iloc[0]
        records: List[Dict[str, Any]] = []

        # 1. Police FIR
        fir_id = str(row.get("fir_id", "")).strip()
        if fir_id and fir_id.lower() != "nan":
            df_fir = self._get_df("police_fir")
            if not df_fir.empty and "fir_number" in df_fir.columns:
                r_m = df_fir[df_fir["fir_number"] == fir_id]
                if not r_m.empty:
                    f = r_m.iloc[0]
                    records.append({
                        "agency": "Police FIR",
                        "agency_code": "police_fir",
                        "record_id": fir_id,
                        "name": str(f.get("complainant_name")) if pd.notna(f.get("complainant_name")) else None,
                        "guardian": str(f.get("father_or_husband")) if pd.notna(f.get("father_or_husband")) else None,
                        "age": int(f.get("age")) if pd.notna(f.get("age")) else None,
                        "district": str(f.get("district")) if pd.notna(f.get("district")) else None,
                        "locality": str(f.get("locality")) if pd.notna(f.get("locality")) else None,
                        "incident_date": str(f.get("incident_date")) if pd.notna(f.get("incident_date")) else None,
                        "record_date": str(f.get("fir_date")) if pd.notna(f.get("fir_date")) else None,
                        "details": f"Offense: {f.get('offense_category', 'General')} | Station: {f.get('police_station', 'N/A')}"
                    })

        # 2. Hospital MLC
        mlc_id = str(row.get("mlc_id", "")).strip()
        if mlc_id and mlc_id.lower() != "nan":
            df_mlc = self._get_df("hospital_mlc")
            if not df_mlc.empty and "mlc_number" in df_mlc.columns:
                r_m = df_mlc[df_mlc["mlc_number"] == mlc_id]
                if not r_m.empty:
                    m_rec = r_m.iloc[0]
                    records.append({
                        "agency": "Hospital MLC",
                        "agency_code": "hospital_mlc",
                        "record_id": mlc_id,
                        "name": str(m_rec.get("patient_name")) if pd.notna(m_rec.get("patient_name")) else None,
                        "guardian": str(m_rec.get("guardian_name")) if pd.notna(m_rec.get("guardian_name")) else None,
                        "age": None,
                        "dob": str(m_rec.get("dob")) if pd.notna(m_rec.get("dob")) else None,
                        "district": str(m_rec.get("district")) if pd.notna(m_rec.get("district")) else None,
                        "locality": str(m_rec.get("residential_locality")) if pd.notna(m_rec.get("residential_locality")) else None,
                        "incident_date": str(m_rec.get("incident_date")) if pd.notna(m_rec.get("incident_date")) else None,
                        "record_date": str(m_rec.get("examination_date")) if pd.notna(m_rec.get("examination_date")) else None,
                        "details": f"Exam: {m_rec.get('injury_type', 'Clinical')} | Hospital: {m_rec.get('hospital_name', 'N/A')}"
                    })

        # 3. Forensic Lab (FSL)
        fsl_id = str(row.get("fsl_id", "")).strip()
        if fsl_id and fsl_id.lower() != "nan":
            df_fsl = self._get_df("forensic_lab")
            if not df_fsl.empty and "fsl_case_number" in df_fsl.columns:
                r_m = df_fsl[df_fsl["fsl_case_number"] == fsl_id]
                if not r_m.empty:
                    fsl_rec = r_m.iloc[0]
                    records.append({
                        "agency": "Forensic Lab (FSL)",
                        "agency_code": "forensic_lab",
                        "record_id": fsl_id,
                        "name": str(fsl_rec.get("subject_name")) if pd.notna(fsl_rec.get("subject_name")) else None,
                        "guardian": str(fsl_rec.get("guardian_name")) if pd.notna(fsl_rec.get("guardian_name")) else None,
                        "age": int(fsl_rec.get("age")) if pd.notna(fsl_rec.get("age")) else None,
                        "district": str(fsl_rec.get("district")) if pd.notna(fsl_rec.get("district")) else None,
                        "locality": None,
                        "incident_date": str(row.get("incident_date")) if pd.notna(row.get("incident_date")) else None,
                        "record_date": str(fsl_rec.get("report_dispatch_date")) if pd.notna(fsl_rec.get("report_dispatch_date")) else None,
                        "details": f"Division: {fsl_rec.get('lab_division', 'N/A')} | Evidence: {fsl_rec.get('evidence_type', 'Biological')}"
                    })

        # 4. Prosecution Chargesheet
        pros_id = str(row.get("prosecutor_id", "")).strip()
        if pros_id and pros_id.lower() != "nan":
            df_pros = self._get_df("prosecutor")
            if not df_pros.empty and "prosecution_file_no" in df_pros.columns:
                r_m = df_pros[df_pros["prosecution_file_no"] == pros_id]
                if not r_m.empty:
                    p_rec = r_m.iloc[0]
                    records.append({
                        "agency": "Prosecution Chargesheet",
                        "agency_code": "prosecutor",
                        "record_id": pros_id,
                        "name": str(p_rec.get("victim_name")) if pd.notna(p_rec.get("victim_name")) else None,
                        "guardian": str(p_rec.get("parent_spouse_name")) if pd.notna(p_rec.get("parent_spouse_name")) else None,
                        "age": int(p_rec.get("age")) if pd.notna(p_rec.get("age")) else None,
                        "district": str(p_rec.get("district")) if pd.notna(p_rec.get("district")) else None,
                        "locality": str(p_rec.get("address")) if pd.notna(p_rec.get("address")) else None,
                        "incident_date": str(p_rec.get("incident_date")) if pd.notna(p_rec.get("incident_date")) else None,
                        "record_date": str(p_rec.get("chargesheet_date")) if pd.notna(p_rec.get("chargesheet_date")) else None,
                        "details": f"Category: {p_rec.get('case_category', 'General')} | Office: {p_rec.get('prosecutor_office', 'N/A')}"
                    })

        # 5. District Court
        crt_id = str(row.get("court_id", "")).strip()
        if crt_id and crt_id.lower() != "nan":
            df_crt = self._get_df("court")
            if not df_crt.empty and "cnr_number" in df_crt.columns:
                r_m = df_crt[df_crt["cnr_number"] == crt_id]
                if not r_m.empty:
                    c_rec = r_m.iloc[0]
                    records.append({
                        "agency": "District Court",
                        "agency_code": "court",
                        "record_id": crt_id,
                        "name": str(c_rec.get("petitioner_victim")) if pd.notna(c_rec.get("petitioner_victim")) else None,
                        "guardian": str(c_rec.get("father_or_husband")) if pd.notna(c_rec.get("father_or_husband")) else None,
                        "dob": str(c_rec.get("dob")) if pd.notna(c_rec.get("dob")) else None,
                        "district": str(c_rec.get("district")) if pd.notna(c_rec.get("district")) else None,
                        "locality": str(c_rec.get("residence_area")) if pd.notna(c_rec.get("residence_area")) else None,
                        "incident_date": str(c_rec.get("date_of_incident")) if pd.notna(c_rec.get("date_of_incident")) else None,
                        "record_date": str(c_rec.get("verdict_date")) if pd.notna(c_rec.get("verdict_date")) else None,
                        "details": f"Status: {c_rec.get('case_status', 'Disposed')} | Court: {c_rec.get('court_name', 'N/A')}"
                    })

        true_district = str(row.get("true_district", "N/A"))
        journey_path = str(row.get("journey_path", "N/A"))
        category = str(row.get("case_category", "N/A"))

        return {
            "id": case_id,
            "title": f"Case {case_id}: {category.replace('_', ' ').title()} ({true_district})",
            "category": "Generated Dataset Case",
            "district": true_district,
            "description": f"Multi-agency trajectory from generated dataset with recorded journey '{journey_path}'.",
            "records": records
        }

    def build_case_explorer_data(
        self,
        case_id_or_key: str,
        role: str = "viewer"
    ) -> Dict[str, Any]:
        """
        Builds full end-to-end case exploration package:
        (a) Fragmented source records
        (b) Pairwise match probability and reasons
        (c) Reconstructed journey timeline with days between stages and drop/stall point
        (d) One-line judicial verdict
        (e) Server-side role privacy enforcement
        """
        # 1. Resolve raw case data
        case_raw = None
        if case_id_or_key in CURATED_EXAMPLES:
            case_raw = CURATED_EXAMPLES[case_id_or_key]
        else:
            case_raw = self.load_case_from_dataset(case_id_or_key)

        if not case_raw:
            # Fallback to clean_match
            case_raw = CURATED_EXAMPLES["clean_match"]

        raw_records = case_raw.get("records", [])
        total_records = len(raw_records)

        # 2. Pairwise Scoring Engine (Fellegi-Sunter)
        pairwise_results = []
        auto_accepts = 0
        human_reviews = 0
        rejects = 0

        for i in range(total_records):
            for j in range(i + 1, total_records):
                ra = raw_records[i]
                rb = raw_records[j]
                score_res = score_case_pair(ra, rb)
                dec = score_res["decision"]

                if dec == "AUTO_ACCEPT":
                    auto_accepts += 1
                elif dec == "HUMAN_REVIEW":
                    human_reviews += 1
                else:
                    rejects += 1

                # Role-adjusted evidence explanations
                evidence_cleaned = []
                for ev in score_res["evidence"]:
                    detail_str = ev.get("detail", "")
                    # Protect raw names in evidence strings based on role
                    if ra.get("name") and ra["name"] in detail_str:
                        detail_str = detail_str.replace(ra["name"], apply_role_privacy(ra["name"], role, prefix="SUBJ"))
                    if rb.get("name") and rb["name"] in detail_str:
                        detail_str = detail_str.replace(rb["name"], apply_role_privacy(rb["name"], role, prefix="SUBJ"))
                    if ra.get("guardian") and ra["guardian"] in detail_str:
                        detail_str = detail_str.replace(ra["guardian"], apply_role_privacy(ra["guardian"], role, prefix="GUARD"))
                    if rb.get("guardian") and rb["guardian"] in detail_str:
                        detail_str = detail_str.replace(rb["guardian"], apply_role_privacy(rb["guardian"], role, prefix="GUARD"))

                    evidence_cleaned.append({
                        "field": ev.get("field"),
                        "status": ev.get("status"),
                        "weight": ev.get("weight"),
                        "detail": detail_str
                    })

                rec_a_label = f"{ra['agency']} ({ra.get('record_id', 'ID')})"
                rec_b_label = f"{rb['agency']} ({rb.get('record_id', 'ID')})"

                pairwise_results.append({
                    "record_a": rec_a_label,
                    "record_b": rec_b_label,
                    "match_probability": score_res["match_probability"],
                    "match_weight": score_res["match_weight"],
                    "decision": dec,
                    "evidence": evidence_cleaned
                })

        # 3. Timeline Reconstruction with Days Between Stages & Bottleneck Detection
        canonical_order = ["police_fir", "hospital_mlc", "forensic_lab", "prosecutor", "court"]
        stage_names = {
            "police_fir": "1. Police FIR",
            "hospital_mlc": "2. Hospital MLC",
            "forensic_lab": "3. Forensic Lab (FSL)",
            "prosecutor": "4. Chargesheet Filed",
            "court": "5. Court Verdict"
        }

        records_map = {r["agency_code"]: r for r in raw_records if r.get("agency_code")}
        timeline_steps = []
        bottlenecks = []
        prev_date: Optional[datetime] = None
        prev_stage_name: Optional[str] = None
        is_dropped = False
        drop_point: Optional[str] = None
        stall_stage: Optional[str] = None

        for idx, code in enumerate(canonical_order):
            s_label = stage_names[code]
            if code in records_map:
                r_item = records_map[code]
                d_str = r_item.get("record_date") or r_item.get("incident_date")
                parsed_d: Optional[datetime] = None
                days_from_prev = None

                if d_str:
                    try:
                        parsed_d = datetime.strptime(str(d_str).strip(), "%Y-%m-%d")
                        if prev_date is not None:
                            days_from_prev = max(0, (parsed_d - prev_date).days)
                        prev_date = parsed_d
                        prev_stage_name = s_label
                    except Exception:
                        pass

                step_data = {
                    "stage": s_label,
                    "agency_code": code,
                    "status": "COMPLETED",
                    "date": d_str,
                    "record_id": r_item.get("record_id"),
                    "days_from_previous": days_from_prev,
                    "elapsed_label": f"+{days_from_prev} days from {prev_stage_name}" if days_from_prev is not None and prev_stage_name else "Inception",
                    "notes": r_item.get("details") or ""
                }
                timeline_steps.append(step_data)
            else:
                # Stage omitted / not reached
                step_data = {
                    "stage": s_label,
                    "agency_code": code,
                    "status": "DROPPED / NOT REACHED",
                    "date": None,
                    "record_id": "NOT REACHED",
                    "days_from_previous": None,
                    "elapsed_label": "Halted",
                    "notes": "Case did not proceed to this institutional milestone"
                }
                timeline_steps.append(step_data)
                if not is_dropped and idx > 0 and canonical_order[idx - 1] in records_map:
                    is_dropped = True
                    drop_point = stage_names[canonical_order[idx - 1]]

        # Evaluate Latency Violations & Bottlenecks
        # Check FSL delay
        if "forensic_lab" in records_map:
            fsl_rec = records_map["forensic_lab"]
            inc_s = fsl_rec.get("incident_date")
            rec_s = fsl_rec.get("record_date")
            if inc_s and rec_s:
                try:
                    d_inc = datetime.strptime(inc_s, "%Y-%m-%d")
                    d_rep = datetime.strptime(rec_s, "%Y-%m-%d")
                    fsl_days = (d_rep - d_inc).days
                    if fsl_days > 60:
                        stall_stage = "Forensic Science Laboratory"
                        bottlenecks.append({
                            "code": "LP-01-FORENSIC-DELAY",
                            "stage": "3. Forensic Lab (FSL)",
                            "delay_days": fsl_days,
                            "benchmark_days": 30,
                            "severity": "CRITICAL" if fsl_days > 120 else "WARNING",
                            "title": "Severe Forensic Turnaround Latency",
                            "detail": f"{fsl_days} days to complete evidence examination (breaches 30-day statutory benchmark by {fsl_days / 30.0:.1f}x)."
                        })
                except Exception:
                    pass

        # Check FIR to Chargesheet delay
        if "police_fir" in records_map and "prosecutor" in records_map:
            fir_rec = records_map["police_fir"]
            pros_rec = records_map["prosecutor"]
            d_fir_s = fir_rec.get("record_date") or fir_rec.get("incident_date")
            d_cs_s = pros_rec.get("record_date")
            if d_fir_s and d_cs_s:
                try:
                    d_fir = datetime.strptime(d_fir_s, "%Y-%m-%d")
                    d_cs = datetime.strptime(d_cs_s, "%Y-%m-%d")
                    cs_days = (d_cs - d_fir).days
                    if cs_days > 60:
                        stall_stage = stall_stage or "Police Investigation"
                        bottlenecks.append({
                            "code": "LP-02-CHARGESHEET-DELAY",
                            "stage": "4. Chargesheet Filed",
                            "delay_days": cs_days,
                            "benchmark_days": 60,
                            "severity": "CRITICAL" if cs_days > 120 else "WARNING",
                            "title": "CrPC 173(1A) 60-Day Mandate Exceeded",
                            "detail": f"Chargesheet took {cs_days} days to file after FIR (statutory limit is 60 days)."
                        })
                except Exception:
                    pass
        elif is_dropped and "police_fir" in records_map and "prosecutor" not in records_map:
            stall_stage = stall_stage or "Investigation to Prosecution Transition"
            bottlenecks.append({
                "code": "LP-02-POLICE-ATTRITION",
                "stage": "4. Chargesheet Filed",
                "delay_days": None,
                "benchmark_days": 60,
                "severity": "CRITICAL",
                "title": "Police-to-Prosecution Case Attrition Leak",
                "detail": "Case terminated after investigation without chargesheet submission, ending survivor legal recourse."
            })

        # 4. Formulate One-Line Judicial Verdict
        if "one_line_verdict" in case_raw and case_raw["one_line_verdict"]:
            verdict = case_raw["one_line_verdict"]
            verdict_type = case_raw.get("verdict_type", "neutral")
        else:
            # Algorithmic verdict formulation for general cases
            if len(pairwise_results) == 0:
                verdict = "Single Record Case: Insufficient cross-agency records to construct linkage."
                verdict_type = "neutral"
            elif rejects == len(pairwise_results):
                verdict = "Definitively Disconnected: False-link protection prevented erroneous merger."
                verdict_type = "danger"
            elif is_dropped:
                verdict = f"Case Attrition Leak: Journey halted after {drop_point} with no court adjudication."
                verdict_type = "warning"
            elif any(b["code"] == "LP-01-FORENSIC-DELAY" for b in bottlenecks):
                verdict = f"Linked with Institutional Delay: Forensic turnaround backlog delayed pipeline ({bottlenecks[0]['delay_days']} days)."
                verdict_type = "warning"
            elif auto_accepts == len(pairwise_results):
                verdict = f"Fully Linked Journey: All {total_records} multi-agency records verified with high confidence (P >= 0.85)."
                verdict_type = "success"
            elif human_reviews > 0:
                verdict = "Uncertain Candidate Links: Held for human adjudication in Review Queue."
                verdict_type = "review"
            else:
                verdict = f"Reconstructed Trajectory: {auto_accepts} auto-accepted links, {human_reviews} uncertain, {rejects} rejected."
                verdict_type = "neutral"

        # 5. Apply Server-Side Role Privacy Enforcement on Output Records
        protected_records = []
        for r in raw_records:
            pr = dict(r)
            raw_n = pr.get("name")
            raw_g = pr.get("guardian")

            pr["name"] = apply_role_privacy(raw_n, role=role, prefix="SUBJ")
            pr["guardian"] = apply_role_privacy(raw_g, role=role, prefix="GUARD")
            if "dob" in pr and role == "viewer":
                # Year only for viewer
                if pr["dob"] and len(str(pr["dob"])) >= 4:
                    pr["dob"] = str(pr["dob"])[:4] + "-**-**"

            protected_records.append(pr)

        return {
            "case_id": case_raw.get("id", case_id_or_key),
            "title": case_raw.get("title", f"Case {case_id_or_key}"),
            "category": case_raw.get("category", "General Trajectory"),
            "district": case_raw.get("district", "N/A"),
            "description": case_raw.get("description", ""),
            "role_view": "masked" if str(role).lower() == "viewer" else "pseudonymised",
            "caller_role": str(role).lower(),
            "synthetic_data": True,
            "synthetic_label": "All data shown is synthetic. No real person or case.",
            "one_line_verdict": verdict,
            "verdict_type": verdict_type,
            "why": case_raw.get("why", "Evaluated against Fellegi-Sunter probabilistic comparison matrix and statutory process milestones."),
            "records": protected_records,
            "pairwise_results": pairwise_results,
            "timeline": {
                "steps": timeline_steps,
                "bottlenecks": bottlenecks,
                "is_dropped": is_dropped,
                "drop_point": drop_point,
                "stall_stage": stall_stage,
                "completed_count": sum(1 for s in timeline_steps if s["status"] == "COMPLETED"),
                "total_milestones": 5
            },
            "summary": {
                "total_source_records": total_records,
                "total_pairwise_combinations": len(pairwise_results),
                "auto_accept_links": auto_accepts,
                "human_review_links": human_reviews,
                "rejected_links": rejects
            }
        }


# Global Singleton Service
_CASE_EXPLORER_SERVICE: Optional[CaseExplorerService] = None

def get_case_explorer_service() -> CaseExplorerService:
    global _CASE_EXPLORER_SERVICE
    if _CASE_EXPLORER_SERVICE is None:
        _CASE_EXPLORER_SERVICE = CaseExplorerService()
    return _CASE_EXPLORER_SERVICE
