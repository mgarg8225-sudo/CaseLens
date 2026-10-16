"""
generate_data.py - Synthetic VAW Case Journey Data Generator.

Generates realistic, privacy-sensitive violence-against-women (VAW) case journeys
across 5 independent public agencies (Police, Hospital, Forensic Lab, Prosecutor, Court)
without shared case identifiers.

Injects realistic drop-offs, district-level leak points, cultural name variations,
transliterations, format swaps, typos, missing values, and duplicate records.
"""

import os
import random
import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from faker import Faker

# Fixed seed for perfect reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker('en_IN')
Faker.seed(RANDOM_SEED)

# Districts & Case Categories
DISTRICTS = [
    "D1_Suryapur",
    "D2_Chandranagar",
    "D3_Rudrapur",      # LEAK POINT 1: Forensic Lab delays (~3.5x normal)
    "D4_Kalyanpur",
    "D5_Mayaganj",
    "D6_Shivgarh",       # LEAK POINT 2: Police-to-Chargesheet drop-off (~70% leak)
    "D7_Anandpur",       # LEAK POINT 3: Court trial delay (>2x normal)
    "D8_Devkalan"
]

CASE_CATEGORIES = [
    "domestic_violence",
    "sexual_assault",
    "dowry_harassment",
    "stalking",
    "trafficking"
]

# District-specific localities for realistic address simulation
LOCALITIES_BY_DISTRICT = {
    "D1_Suryapur": ["Civil Lines", "Old City", "Gandhi Chowk", "Shanti Nagar", "Shastri Market", "Sector 4", "Ward 7"],
    "D2_Chandranagar": ["Lake View", "Station Road", "Nehru Nagar", "Subhash Colony", "Model Town", "Indira Market"],
    "D3_Rudrapur": ["Industrial Area Ph-1", "Mill Gate", "Rampur Ward 3", "Kali Mandir Gali", "Azad Nagar", "Kotwali Chowk"],
    "D4_Kalyanpur": ["Vikas Nagar", "Green Park", "Railway Colony", "Patel Nagar", "Kabir Chowk", "Anand Vihar"],
    "D5_Mayaganj": ["Riverbank Basti", "Bazaar Samiti", "Ganj Road", "Transport Nagar", "Ward 12", "Shivaji Nagar"],
    "D6_Shivgarh": ["Garhi Ward", "Fort Road", "Cantt Area", "Purana Thana", "Krishna Colony", "Bypass Road"],
    "D7_Anandpur": ["University Road", "Civil Court Area", "Pragati Vihar", "Shambhu Ghat", "Teachers Colony"],
    "D8_Devkalan": ["Dehat Ward 1", "Mandi Road", "Kisan Nagar", "Shanti Vihar", "Block Colony", "Sector 9"]
}

# Common transliteration mappings for Indian names
TRANSLITERATION_MAP = {
    "Sunita": ["Suneeta", "Sunitha", "Suneetha"],
    "Priya": ["Preya", "Preeya"],
    "Pooja": ["Puja", "Poojah"],
    "Anita": ["Aneeta", "Anitha"],
    "Kavita": ["Kavitha", "Kaveeta"],
    "Geeta": ["Gita", "Geetah"],
    "Neelam": ["Nilam", "Nelam"],
    "Meena": ["Mina", "Meenah"],
    "Lakshmi": ["Laxmi", "Laksmi"],
    "Vandana": ["Bandana", "Wandana"],
    "Deepa": ["Dipa", "Deepah"],
    "Ramesh": ["Rmesh", "Ramesha", "Rameesh"],
    "Mohammad": ["Mohd", "Mohammed", "Mohamad", "Md"],
    "Choudhary": ["Chowdhury", "Choudhry", "Chowdary"],
    "Shukla": ["Sukla", "Shookla"],
    "Yadav": ["Yadaw", "Jadav"],
    "Verma": ["Varma", "Burman"],
    "Gupta": ["Goopta", "Guptha"],
    "Mishra": ["Misra", "Mishar"],
    "Suresh": ["Suresha", "Sooresh"],
    "Rajesh": ["Rajeesh", "Rajesha"],
    "Dinesh": ["Deenesh", "Dinesha"]
}

FIRST_NAMES = [
    "Sunita", "Priya", "Pooja", "Anita", "Rekha", "Kavita", "Geeta", "Neelam",
    "Meena", "Lakshmi", "Vandana", "Deepa", "Poonam", "Shabana", "Rubina",
    "Anjali", "Renu", "Shashi", "Ritu", "Sarita", "Manju", "Seema", "Mamta",
    "Usha", "Pinky", "Jyoti", "Aarti", "Radha", "Suman", "Maya", "Kamla",
    "Nisha", "Swati", "Rashmi", "Farida", "Preeti", "Kiran", "Sapna"
]

MALE_NAMES = [
    "Ramesh", "Suresh", "Mukesh", "Rajesh", "Dinesh", "Manoj", "Mohammad",
    "Sanjay", "Ajay", "Vijay", "Sunil", "Anil", "Ashok", "Vinod", "Ram",
    "Shyam", "Harish", "Naresh", "Satish", "Mahesh", "Amit", "Sumit",
    "Arvind", "Pradeep", "Deepak", "Jagdish", "Rakesh", "Praveen", "Santosh"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Yadav", "Singh", "Kumar", "Devi", "Choudhary",
    "Shukla", "Mishra", "Patel", "Tiwari", "Dubey", "Jha", "Das", "Roy",
    "Khan", "Ansari", "Chauhan", "Rathore", "Rawat", "Joshi", "Bhatt", "Pandey"
]


def perturb_name(first_name: str, last_name: str) -> Tuple[str, str, str]:
    """
    Applies realistic spelling variations, transliterations, initials, or name-swapping.
    Returns: (output_first, output_last, formatted_full_name)
    """
    f = first_name
    l = last_name

    # 1. Transliteration variant on first name
    if f in TRANSLITERATION_MAP and random.random() < 0.25:
        f = random.choice(TRANSLITERATION_MAP[f])
    elif random.random() < 0.08:
        # Typo: swap two adjacent letters or drop one
        if len(f) > 4:
            idx = random.randint(1, len(f) - 2)
            if random.random() < 0.5:
                f = f[:idx] + f[idx+1] + f[idx] + f[idx+2:]
            else:
                f = f[:idx] + f[idx+1:]

    # 2. Transliteration variant on last name
    if l in TRANSLITERATION_MAP and random.random() < 0.25:
        l = random.choice(TRANSLITERATION_MAP[l])
    elif random.random() < 0.08:
        if len(l) > 4:
            idx = random.randint(1, len(l) - 2)
            l = l[:idx] + l[idx+1:]

    # 3. Formatting style
    r = random.random()
    if r < 0.08:
        # Swapped: "Sharma Sunita"
        full = f"{l} {f}"
    elif r < 0.15:
        # Initial First: "S. Sharma"
        full = f"{f[0]}. {l}"
    elif r < 0.20:
        # Initial Last: "Sunita S."
        full = f"{f} {l[0]}."
    elif r < 0.24:
        # First name only (common in institutional records)
        full = f
    else:
        # Standard: "Sunita Sharma"
        full = f"{f} {l}"

    return f, l, full


def perturb_relative_name(rel_name: str) -> str:
    """Applies transliteration or initials to father/husband name."""
    if not rel_name or rel_name == "Unknown":
        return ""
    parts = rel_name.split()
    if len(parts) >= 2:
        f, l = parts[0], parts[1]
        if f in TRANSLITERATION_MAP and random.random() < 0.25:
            f = random.choice(TRANSLITERATION_MAP[f])
        if l in TRANSLITERATION_MAP and random.random() < 0.25:
            l = random.choice(TRANSLITERATION_MAP[l])
        r = random.random()
        if r < 0.10:
            return f"{f[0]}. {l}"
        elif r < 0.20:
            return f"{l} {f}"
        return f"{f} {l}"
    return rel_name


def perturb_address(locality: str, district: str) -> str:
    """Applies realistic address formatting, abbreviations, or landmark noise."""
    r = random.random()
    loc = locality
    abbrev_map = {
        "Nagar": "Ngr",
        "Sector": "Sec",
        "Road": "Rd",
        "Colony": "Col",
        "Area": "Ar",
        "Market": "Mkt"
    }
    for k, v in abbrev_map.items():
        if k in loc and random.random() < 0.35:
            loc = loc.replace(k, v)

    if r < 0.25:
        landmarks = ["Near Kali Mandir", "Opp Govt School", "Behind Water Tank", "Plot No 42", "Gali No 3", "Shop 12"]
        return f"{random.choice(landmarks)}, {loc}"
    elif r < 0.50:
        return f"Ward {random.randint(1, 25)}, {loc}"
    elif r < 0.75:
        return f"{loc}, {district.split('_')[1]}"
    else:
        return loc


def perturb_date(d: datetime.date, swap_prob: float = 0.05, jitter_days: int = 2) -> datetime.date:
    """
    Applies small date errors (jitter +/- 1-2 days, or day/month swap when day <= 12).
    """
    # Jitter
    if random.random() < 0.20:
        offset = random.randint(-jitter_days, jitter_days)
        d = d + datetime.timedelta(days=offset)

    # Day-month swap if ambiguous
    if swap_prob > 0 and random.random() < swap_prob:
        if d.day <= 12 and d.month <= 12 and d.day != d.month:
            try:
                d = datetime.date(d.year, d.day, d.month)
            except ValueError:
                pass
    return d


def generate_dataset(num_cases: int = 5000, output_dir: str = "data") -> Dict[str, Any]:
    """
    Main generator function.
    Creates 5 agency tables and 1 ground truth table.
    """
    os.makedirs(output_dir, exist_ok=True)

    ground_truth_records = []
    fir_records = []
    mlc_records = []
    fsl_records = []
    prosecutor_records = []
    court_records = []

    # Keep track of duplicates
    duplicate_counter = 0

    base_start_date = datetime.date(2023, 1, 1)

    print(f"Generating {num_cases} synthetic VAW case journeys across 8 districts...")

    for i in range(1, num_cases + 1):
        case_id = f"CASE-{i:05d}"
        district = random.choice(DISTRICTS)
        category = random.choice(CASE_CATEGORIES)
        locality = random.choice(LOCALITIES_BY_DISTRICT[district])

        true_first = random.choice(FIRST_NAMES)
        true_last = random.choice(LAST_NAMES)
        true_full = f"{true_first} {true_last}"

        true_rel_first = random.choice(MALE_NAMES)
        true_rel_last = true_last if random.random() < 0.85 else random.choice(LAST_NAMES)
        true_rel_name = f"{true_rel_first} {true_rel_last}"
        relation_type = "Husband" if category in ["domestic_violence", "dowry_harassment"] else random.choice(["Father", "Husband"])

        # Age and DOB
        true_age = int(np.clip(np.random.normal(27, 7), 16, 65))
        # Incident date between Jan 2023 and Dec 2024
        day_offset = random.randint(0, 600)
        incident_date = base_start_date + datetime.timedelta(days=day_offset)
        true_dob = incident_date - datetime.timedelta(days=int(true_age * 365.25) + random.randint(0, 300))

        # ----------------------------------------------------
        # JOURNEY SIMULATION & DROP-OFF LOGIC
        # ----------------------------------------------------
        # Stage 1: Police FIR
        # 94% of reported incidents have an FIR registered
        has_fir = random.random() < 0.94
        fir_date = None
        fir_id = None

        if has_fir:
            fir_delay = max(0, int(np.random.exponential(2)))
            fir_date = incident_date + datetime.timedelta(days=fir_delay)
            fir_id = f"FIR-{fir_date.year}-{district[:2]}-{i:04d}"

        # Stage 2: Hospital MLC (Medical Examination)
        # Rates depend on category
        mlc_prob = {
            "sexual_assault": 0.92,
            "domestic_violence": 0.82,
            "dowry_harassment": 0.65,
            "trafficking": 0.45,
            "stalking": 0.30
        }[category]
        has_mlc = random.random() < mlc_prob
        mlc_date = None
        mlc_id = None

        if has_mlc:
            mlc_delay = max(0, int(np.random.exponential(1.5)))
            mlc_date = incident_date + datetime.timedelta(days=mlc_delay)
            mlc_id = f"MLC-{mlc_date.year}-{random.randint(100, 999)}-{i:04d}"

        # Stage 3: Forensic Lab (FSL)
        # Primarily for sexual assault, severe domestic violence/dowry, trafficking
        fsl_prob = {
            "sexual_assault": 0.88,
            "domestic_violence": 0.35,
            "dowry_harassment": 0.50,
            "trafficking": 0.40,
            "stalking": 0.10
        }[category]
        has_fsl = (has_fir or has_mlc) and (random.random() < fsl_prob)
        fsl_date = None
        fsl_receipt_date = None
        fsl_id = None

        if has_fsl:
            # Baseline delay from FIR/MLC to FSL report dispatch: ~38 days
            # LEAK POINT 1: District 3 (Rudrapur) delay is ~3.5x normal (mean ~135 days!)
            if district == "D3_Rudrapur":
                fsl_turnaround = int(np.random.normal(135, 20))
            else:
                fsl_turnaround = int(np.random.normal(38, 8))
            fsl_turnaround = max(10, fsl_turnaround)

            ref_date = fir_date if fir_date else mlc_date
            fsl_receipt_date = ref_date + datetime.timedelta(days=random.randint(2, 5))
            fsl_date = fsl_receipt_date + datetime.timedelta(days=fsl_turnaround)
            fsl_id = f"FSL-{fsl_date.year}-{district[:2]}-{i:04d}"

        # Stage 4: Prosecution Chargesheet Filing
        # Requires FIR.
        # LEAK POINT 2: District 6 (Shivgarh) chargesheet drop-off is severe!
        # Normal chargesheet rate is ~68%, but in D6 it drops to ~28%.
        # Stalking also drops heavily before prosecution (~30%).
        has_chargesheet = False
        chargesheet_date = None
        prosecutor_id = None

        if has_fir:
            if district == "D6_Shivgarh":
                cs_prob = 0.28  # LEAK POINT 2
            elif category == "stalking":
                cs_prob = 0.32  # Category specific leak
            else:
                cs_prob = 0.70

            has_chargesheet = random.random() < cs_prob

        if has_chargesheet:
            # Chargesheet delay depends on forensic dispatch if forensic was involved
            base_ref = fsl_date if (has_fsl and fsl_date > fir_date) else fir_date
            cs_delay = max(15, int(np.random.normal(65, 20)))
            chargesheet_date = base_ref + datetime.timedelta(days=cs_delay)
            prosecutor_id = f"PP-CHG-{chargesheet_date.year}-{district[:2]}-{i:04d}"

        # Stage 5: Court Verdict / Trial Conclusion
        # Requires chargesheet
        has_court = False
        verdict_date = None
        court_id = None
        trial_outcome = None

        if has_chargesheet:
            # ~55% of chargesheeted cases reach final verdict in the observation window
            has_court = random.random() < 0.55

        if has_court:
            # LEAK POINT 3: District 7 (Anandpur) court backlog: trial delay >2x normal
            if district == "D7_Anandpur":
                trial_days = int(np.random.normal(620, 80))
            else:
                trial_days = int(np.random.normal(280, 50))
            trial_days = max(90, trial_days)

            verdict_date = chargesheet_date + datetime.timedelta(days=trial_days)
            court_id = f"CNR-{district[:2]}-{verdict_date.year}-{i:05d}"

            outcomes = ["Conviction", "Acquittal", "Compromised/Withdrawn", "Abated"]
            weights = [0.26, 0.52, 0.18, 0.04]
            trial_outcome = np.random.choice(outcomes, p=weights)

        # Build Ground Truth record
        journey_path = " -> ".join([
            s for s, flag in [
                ("FIR", has_fir),
                ("MLC", has_mlc),
                ("FSL", has_fsl),
                ("CHARGESHEET", has_chargesheet),
                ("VERDICT", has_court)
            ] if flag
        ])

        gt_record = {
            "case_id": case_id,
            "true_first_name": true_first,
            "true_last_name": true_last,
            "true_full_name": true_full,
            "true_relative_name": true_rel_name,
            "true_relation_type": relation_type,
            "true_age": true_age,
            "true_dob": str(true_dob),
            "true_district": district,
            "true_locality": locality,
            "case_category": category,
            "incident_date": str(incident_date),
            "has_fir": has_fir,
            "has_mlc": has_mlc,
            "has_fsl": has_fsl,
            "has_chargesheet": has_chargesheet,
            "has_court": has_court,
            "journey_path": journey_path,
            "fir_id": fir_id or "",
            "mlc_id": mlc_id or "",
            "fsl_id": fsl_id or "",
            "prosecutor_id": prosecutor_id or "",
            "court_id": court_id or "",
            "fir_date": str(fir_date) if fir_date else "",
            "mlc_date": str(mlc_date) if mlc_date else "",
            "fsl_date": str(fsl_date) if fsl_date else "",
            "chargesheet_date": str(chargesheet_date) if chargesheet_date else "",
            "verdict_date": str(verdict_date) if verdict_date else "",
            "trial_outcome": trial_outcome or ""
        }
        ground_truth_records.append(gt_record)

        # ----------------------------------------------------
        # AGENCY TABLE GENERATION WITH NOISE INJECTION
        # ----------------------------------------------------

        # Helper: introduce missingness (10-20%)
        def maybe_missing(val: Any, p: float = 0.14) -> Any:
            return "" if random.random() < p else val

        # 1. Police FIR record
        if has_fir:
            f, l, full = perturb_name(true_first, true_last)
            rel = perturb_relative_name(true_rel_name)
            p_inc = perturb_date(incident_date)
            p_fir = perturb_date(fir_date, swap_prob=0.03)
            addr = perturb_address(locality, district)
            age_val = true_age if random.random() > 0.12 else ""

            fir_row = {
                "fir_number": fir_id,
                "complainant_name": full,
                "father_or_husband": maybe_missing(rel, 0.15),
                "age": age_val,
                "locality": maybe_missing(addr, 0.12),
                "district": district,
                "incident_date": str(p_inc),
                "fir_date": str(p_fir),
                "offense_category": category,
                "police_station": f"PS_{district.split('_')[1]}_{random.choice(['East', 'West', 'Central', 'Rural'])}"
            }
            fir_records.append(fir_row)

            # Small duplicate rate (~2.5% supplemental FIR record)
            if random.random() < 0.025:
                duplicate_counter += 1
                dup_fir = fir_row.copy()
                dup_fir["fir_number"] = f"{fir_id}-SUPP"
                dup_fir["complainant_name"] = f"{true_first} {true_last}" # slight spelling/format variation
                fir_records.append(dup_fir)

        # 2. Hospital MLC record
        if has_mlc:
            f, l, full = perturb_name(true_first, true_last)
            rel = perturb_relative_name(true_rel_name)
            p_inc = perturb_date(incident_date)
            p_exam = perturb_date(mlc_date, swap_prob=0.04)
            addr = perturb_address(locality, district)
            # Hospital uses DOB instead of age
            dob_val = str(perturb_date(true_dob, swap_prob=0.02, jitter_days=365)) if random.random() > 0.15 else ""

            injury_types = ["Blunt Trauma", "Contusion / Laceration", "Psychological Trauma", "Severe Assault Marks", "Normal / Under Observation"]
            mlc_row = {
                "mlc_number": mlc_id,
                "patient_name": full,
                "guardian_name": maybe_missing(rel, 0.18),
                "dob": dob_val,
                "residential_locality": maybe_missing(addr, 0.14),
                "district": district,
                "incident_date": str(p_inc),
                "examination_date": str(p_exam),
                "injury_type": random.choice(injury_types),
                "hospital_name": f"District_Civil_Hospital_{district.split('_')[1]}"
            }
            mlc_records.append(mlc_row)

            # Hospital re-exam duplicate (~1.5%)
            if random.random() < 0.015:
                duplicate_counter += 1
                dup_mlc = mlc_row.copy()
                dup_mlc["mlc_number"] = f"{mlc_id}-RE"
                mlc_records.append(dup_mlc)

        # 3. Forensic Lab record
        if has_fsl:
            f, l, full = perturb_name(true_first, true_last)
            rel = perturb_relative_name(true_rel_name)
            p_rec = perturb_date(fsl_receipt_date)
            p_rep = perturb_date(fsl_date, swap_prob=0.03)
            age_val = true_age if random.random() > 0.20 else ""

            ev_types = ["DNA Profile", "Toxicology Screen", "Biological Serology", "Physical Trace", "Digital Forensics"]
            fsl_row = {
                "fsl_case_number": fsl_id,
                "subject_name": full,
                "guardian_name": maybe_missing(rel, 0.22),
                "age": age_val,
                "district": district,
                "sample_receipt_date": str(p_rec),
                "report_dispatch_date": str(p_rep),
                "evidence_type": random.choice(ev_types),
                "lab_division": f"Regional_FSL_{district.split('_')[1]}"
            }
            fsl_records.append(fsl_row)

        # 4. Prosecutor record
        if has_chargesheet:
            f, l, full = perturb_name(true_first, true_last)
            rel = perturb_relative_name(true_rel_name)
            p_inc = perturb_date(incident_date)
            p_cs = perturb_date(chargesheet_date, swap_prob=0.03)
            addr = perturb_address(locality, district)
            age_val = true_age if random.random() > 0.12 else ""

            prosecutor_row = {
                "prosecution_file_no": prosecutor_id,
                "victim_name": full,
                "parent_spouse_name": maybe_missing(rel, 0.16),
                "age": age_val,
                "address": maybe_missing(addr, 0.10),
                "district": district,
                "incident_date": str(p_inc),
                "chargesheet_date": str(p_cs),
                "case_category": category,
                "prosecutor_office": f"PP_Office_{district.split('_')[1]}"
            }
            prosecutor_records.append(prosecutor_row)

        # 5. Court record
        if has_court:
            f, l, full = perturb_name(true_first, true_last)
            rel = perturb_relative_name(true_rel_name)
            p_inc = perturb_date(incident_date)
            p_ver = perturb_date(verdict_date, swap_prob=0.04)
            addr = perturb_address(locality, district)
            # Court has DOB or Age
            dob_val = str(perturb_date(true_dob, swap_prob=0.02, jitter_days=365)) if random.random() > 0.18 else ""

            court_row = {
                "cnr_number": court_id,
                "petitioner_victim": full,
                "father_or_husband": maybe_missing(rel, 0.15),
                "dob": dob_val,
                "residence_area": maybe_missing(addr, 0.12),
                "district": district,
                "date_of_incident": str(p_inc),
                "verdict_date": str(p_ver),
                "case_status": trial_outcome,
                "court_name": f"District_Sessions_Court_{district.split('_')[1]}"
            }
            court_records.append(court_row)

    # Convert to DataFrames
    df_gt = pd.DataFrame(ground_truth_records)
    df_fir = pd.DataFrame(fir_records)
    df_mlc = pd.DataFrame(mlc_records)
    df_fsl = pd.DataFrame(fsl_records)
    df_pros = pd.DataFrame(prosecutor_records)
    df_court = pd.DataFrame(court_records)

    # Save CSVs
    gt_path = os.path.join(output_dir, "ground_truth.csv")
    fir_path = os.path.join(output_dir, "police_fir.csv")
    mlc_path = os.path.join(output_dir, "hospital_mlc.csv")
    fsl_path = os.path.join(output_dir, "forensic_lab.csv")
    pros_path = os.path.join(output_dir, "prosecutor.csv")
    court_path = os.path.join(output_dir, "court.csv")

    df_gt.to_csv(gt_path, index=False)
    df_fir.to_csv(fir_path, index=False)
    df_mlc.to_csv(mlc_path, index=False)
    df_fsl.to_csv(fsl_path, index=False)
    df_pros.to_csv(pros_path, index=False)
    df_court.to_csv(court_path, index=False)

    # Compute Quality Metrics & Leak Point Validation
    summary = compute_quality_summary(df_gt, df_fir, df_mlc, df_fsl, df_pros, df_court, duplicate_counter)
    return summary


def compute_quality_summary(
    df_gt: pd.DataFrame,
    df_fir: pd.DataFrame,
    df_mlc: pd.DataFrame,
    df_fsl: pd.DataFrame,
    df_pros: pd.DataFrame,
    df_court: pd.DataFrame,
    duplicate_counter: int
) -> Dict[str, Any]:
    """Computes comprehensive quality metrics, missingness, and leak point verification."""
    agency_dfs = {
        "police_fir": df_fir,
        "hospital_mlc": df_mlc,
        "forensic_lab": df_fsl,
        "prosecutor": df_pros,
        "court": df_court
    }

    agency_stats = {}
    for name, df in agency_dfs.items():
        missing_dict = {}
        for col in df.columns:
            # Empty strings or nulls
            empty_count = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
            missing_dict[col] = round((empty_count / len(df)) * 100, 2)
        agency_stats[name] = {
            "row_count": len(df),
            "missing_pct": missing_dict
        }

    # Leak Point 1: Forensic turnaround by district
    df_fsl_calc = df_fsl.copy()
    df_fsl_calc["receipt_dt"] = pd.to_datetime(df_fsl_calc["sample_receipt_date"])
    df_fsl_calc["dispatch_dt"] = pd.to_datetime(df_fsl_calc["report_dispatch_date"])
    df_fsl_calc["delay_days"] = (df_fsl_calc["dispatch_dt"] - df_fsl_calc["receipt_dt"]).dt.days
    fsl_delay_by_district = df_fsl_calc.groupby("district")["delay_days"].mean().round(1).to_dict()

    # Leak Point 2: FIR to Chargesheet rate by district
    cs_by_district = df_gt.groupby("true_district").agg(
        total_firs=("has_fir", "sum"),
        total_chargesheets=("has_chargesheet", "sum")
    )
    cs_by_district["chargesheet_rate_pct"] = (
        (cs_by_district["total_chargesheets"] / cs_by_district["total_firs"]) * 100
    ).round(1)
    fir_to_cs_rate_dict = cs_by_district["chargesheet_rate_pct"].to_dict()

    # Overall Drop-off funnel
    n_total = len(df_gt)
    funnel = {
        "Total Cases (Reported)": n_total,
        "FIR Registered": int(df_gt["has_fir"].sum()),
        "Medical Exam (MLC)": int(df_gt["has_mlc"].sum()),
        "Forensic Exam (FSL)": int(df_gt["has_fsl"].sum()),
        "Chargesheet Filed": int(df_gt["has_chargesheet"].sum()),
        "Court Verdict Reached": int(df_gt["has_court"].sum())
    }

    summary = {
        "total_true_cases": n_total,
        "agency_stats": agency_stats,
        "total_injected_duplicates": duplicate_counter,
        "funnel": funnel,
        "fsl_delay_by_district": fsl_delay_by_district,
        "fir_to_chargesheet_rate_by_district": fir_to_cs_rate_dict
    }

    return summary


if __name__ == "__main__":
    summary = generate_dataset(num_cases=5000, output_dir="data")
    print("\n" + "="*70)
    print("PHASE 1: SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print(f"Total True Cases Generated: {summary['total_true_cases']}")
    print(f"Total Injected Duplicates: {summary['total_injected_duplicates']}")
    print("\nAgency Record Counts & Missingness:")
    for ag, st in summary["agency_stats"].items():
        print(f"  - {ag:<15}: {st['row_count']:>5} records")
        for col, pct in st["missing_pct"].items():
            if pct > 0:
                print(f"      * {col:<22}: {pct:>5.1f}% missing")

    print("\nFunnel Progression Across Stages:")
    for stage, count in summary["funnel"].items():
        pct = (count / summary['total_true_cases']) * 100
        print(f"  - {stage:<25}: {count:>5} ({pct:>5.1f}%)")

    print("\nInjected Leak Point 1 (Forensic Delay Days by District):")
    for dist, days in summary["fsl_delay_by_district"].items():
        highlight = " <-- [LEAK POINT 1: ~3.5x Delay]" if "Rudrapur" in dist else ""
        print(f"  - {dist:<18}: {days:>5.1f} days{highlight}")

    print("\nInjected Leak Point 2 (FIR -> Chargesheet Conversion % by District):")
    for dist, pct in summary["fir_to_chargesheet_rate_by_district"].items():
        highlight = " <-- [LEAK POINT 2: Severe Drop-off]" if "Shivgarh" in dist else ""
        print(f"  - {dist:<18}: {pct:>5.1f}%{highlight}")
    print("="*70 + "\n")
