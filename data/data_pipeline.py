"""
UK Recruitment Data Pipeline

Generates synthetic candidate data calibrated to real UK ONS/LFS statistics.
Sources:
  - ONS Labour Force Survey (LFS) 2023
  - Annual Population Survey (APS) 2023
  - EHRC Workforce Equality Data 2023
  - NOMIS Labour Market Statistics

Optional: Install 'sdv' for more statistically realistic synthetic data.
  pip install sdv
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.dependencies import SDV_AVAILABLE

if SDV_AVAILABLE:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

# These are the real UK ONS/LFS Distributions 

UK_GENDER_DIST = {"Male": 0.51, "Female": 0.49}

UK_ETHNICITY_DIST = {
    "White British":           0.799,
    "White Other":             0.052,
    "Asian/Asian British":     0.076,
    "Black/African/Caribbean": 0.034,
    "Mixed/Multiple":          0.023,
    "Other":                   0.016,
}

UK_AGE_DIST = {
    "16-24": 0.148,
    "25-34": 0.229,
    "35-44": 0.225,
    "45-54": 0.221,
    "55-64": 0.177,
}

UK_DISABILITY_RATE = 0.175

UK_EDUCATION_DIST = {
    "No Qualification":   0.07,
    "GCSE/Equivalent":    0.23,
    "A-Level/Equivalent": 0.22,
    "Higher Education":   0.11,
    "Degree or Above":    0.37,
}

UK_REGIONS = {
    "London":                0.152,
    "South East":            0.138,
    "North West":            0.112,
    "East of England":       0.094,
    "West Midlands":         0.088,
    "Yorkshire & Humber":    0.086,
    "South West":            0.084,
    "East Midlands":         0.074,
    "North East":            0.042,
    "Wales":                 0.049,
    "Scotland":              0.085,
    "Northern Ireland":      0.030,
    "East Midlands (other)": 0.006,
}

UK_JOB_ROLES = [
    "Software Developer", "Marketing Executive", "Sales Representative",
    "HR Coordinator", "Financial Analyst", "Project Manager",
    "Customer Service Advisor", "Operations Manager", "Data Analyst",
    "Business Development Manager", "UX Designer", "Accountant",
    "Recruitment Consultant", "Office Manager", "IT Support Specialist",
]

UK_UNIVERSITIES = [
    "University of Oxford", "University of Cambridge", "Imperial College London",
    "University College London", "University of Manchester", "University of Edinburgh",
    "University of Bristol", "King's College London", "University of Warwick",
    "University of Leeds", "University of Sheffield", "University of Birmingham",
    "Nottingham Trent University", "Manchester Metropolitan University",
    "Leeds Beckett University", "London Metropolitan University",
    "University of the West of England", "Coventry University",
    "Open University", "Aston University",
]

RUSSELL_GROUP = {
    "University of Oxford", "University of Cambridge", "Imperial College London",
    "University College London", "University of Manchester", "University of Edinburgh",
    "University of Bristol", "King's College London", "University of Warwick",
    "University of Leeds", "University of Sheffield", "University of Birmingham",
}

# Helpers functions

# This function is for generating weighted random choices
def weighted_choice(distribution: dict) -> str:
    keys    = list(distribution.keys())
    weights = np.array(list(distribution.values()), dtype=float)
    weights = weights / weights.sum()
    return np.random.choice(keys, p=weights)


def sample_age_from_band(band: str) -> int:
    lo, hi = map(int, band.split("-"))
    return random.randint(lo, hi)


def generate_name(gender: str) -> str:
    male_first = [
        "James", "Oliver", "Harry", "George", "Noah", "Jack", "Charlie",
        "Jacob", "Alfie", "Freddie", "Mohammed", "Aiden", "Ethan", "Liam",
        "Daniel", "Samuel", "Alexander", "Benjamin", "William", "Thomas",
    ]
    female_first = [
        "Olivia", "Amelia", "Isla", "Ava", "Emily", "Isabella", "Mia",
        "Poppy", "Ella", "Lily", "Sophie", "Grace", "Freya", "Charlotte",
        "Evie", "Layla", "Hannah", "Scarlett", "Chloe", "Jessica",
    ]
    last_names = [
        "Smith", "Jones", "Williams", "Taylor", "Brown", "Davies", "Evans",
        "Wilson", "Thomas", "Roberts", "Johnson", "Lewis", "Walker", "Robinson",
        "Wood", "Thompson", "White", "Watson", "Jackson", "Wright",
        "Patel", "Khan", "Ali", "Ahmed", "Singh", "Sharma", "Okafor",
        "Mensah", "Diallo", "Osei", "Kim", "Chen", "Wang", "Li",
    ]
    first = random.choice(male_first if gender == "Male" else female_first)
    last  = random.choice(last_names)
    return f"{first} {last}"


def compute_cv_score(row: dict, introduce_bias: bool = False) -> float:
    score = 0.0

    edu_scores = {
        "No Qualification":   0,
        "GCSE/Equivalent":    8,
        "A-Level/Equivalent": 15,
        "Higher Education":   22,
        "Degree or Above":    30,
    }
    score += edu_scores.get(row["education_level"], 0)

    if row.get("university") in RUSSELL_GROUP:
        score += 5

    score += min(row["years_experience"] * 2.5, 25)

    gap_penalty = min(row["employment_gap_years"] * 5, 10)
    score -= gap_penalty

    score += row["keyword_match_score"] * 20
    score += np.random.normal(0, 3)

    if introduce_bias:
        if row["gender"] == "Female":
            score -= np.random.uniform(3, 7)
        if row["ethnicity"] not in ("White British", "White Other"):
            score -= np.random.uniform(4, 9)
        if row["age"] > 45:
            score -= np.random.uniform(2, 6)
        if row["has_disability"]:
            score -= np.random.uniform(2, 5)

    return round(np.clip(score, 0, 100), 2)


def make_hiring_decision(cv_score: float, threshold: float = 55.0) -> int:
    prob = 1 / (1 + np.exp(-(cv_score - threshold) / 8))
    return int(np.random.random() < prob)

# Core Generator Functions

def _generate_base(n_candidates: int, introduce_bias: bool, seed: int) -> pd.DataFrame:
    """Built-in ONS-calibrated generator (no optional dependencies required)."""
    np.random.seed(seed)
    random.seed(seed)

    records = []
    for i in range(n_candidates):
        gender        = weighted_choice(UK_GENDER_DIST)
        age_band      = weighted_choice(UK_AGE_DIST)
        age           = sample_age_from_band(age_band)
        ethnicity     = weighted_choice(UK_ETHNICITY_DIST)
        region        = weighted_choice(UK_REGIONS)
        education     = weighted_choice(UK_EDUCATION_DIST)
        has_disability = np.random.random() < UK_DISABILITY_RATE

        max_exp         = max(0, age - 18)
        years_experience = min(int(np.random.exponential(scale=6)), max_exp)

        university = None
        if education == "Degree or Above":
            university = random.choice(UK_UNIVERSITIES)

        gap_prob = 0.15
        if age > 45:              gap_prob += 0.10
        if has_disability:        gap_prob += 0.08
        if gender == "Female" and 28 <= age <= 42:
                                  gap_prob += 0.12
        employment_gap_years = (
            round(min(np.random.exponential(1.5), 10), 1)
            if np.random.random() < gap_prob else 0
        )

        keyword_match_score = round(np.random.beta(2, 3), 3)
        job_role            = random.choice(UK_JOB_ROLES)

        row = {
            "candidate_id":        f"UK-{i+1:05d}",
            "name":                generate_name(gender),
            "gender":              gender,
            "age":                 age,
            "age_band":            age_band,
            "ethnicity":           ethnicity,
            "region":              region,
            "has_disability":      has_disability,
            "education_level":     education,
            "university":          university,
            "years_experience":    years_experience,
            "employment_gap_years": employment_gap_years,
            "keyword_match_score": keyword_match_score,
            "job_role":            job_role,
        }

        row["cv_score"] = compute_cv_score(row, introduce_bias=introduce_bias)
        row["hired"]    = make_hiring_decision(row["cv_score"])

        records.append(row)

    return pd.DataFrame(records)


def _generate_with_sdv(n_candidates: int, introduce_bias: bool, seed: int) -> pd.DataFrame:
    """
    SDV-enhanced generator.
    Generates a seed dataset with the built-in method, fits a GaussianCopula
    synthesizer to learn statistical relationships, then samples a larger dataset.
    """
    print("[Data Pipeline] SDV detected — generating seed dataset...")
    seed_size = min(500, n_candidates)
    seed_df   = _generate_base(seed_size, introduce_bias, seed)

    # SDV works on numeric/categorical columns only — drop name/candidate_id
    sdv_cols  = [
        "gender", "age", "age_band", "ethnicity", "region", "has_disability",
        "education_level", "years_experience", "employment_gap_years",
        "keyword_match_score", "job_role", "cv_score", "hired",
    ]
    seed_input = seed_df[sdv_cols].copy()
    seed_input["has_disability"] = seed_input["has_disability"].astype(str)

    print("[Data Pipeline] Fitting SDV GaussianCopulaSynthesizer...")
    metadata    = SingleTableMetadata()
    metadata.detect_from_dataframe(seed_input)
    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    synthesizer.fit(seed_input)

    print(f"[Data Pipeline] Sampling {n_candidates} records from SDV...")
    synth_df = synthesizer.sample(num_rows=n_candidates)

    # Restore types
    synth_df["has_disability"] = synth_df["has_disability"].map(
        lambda x: True if str(x).lower() == "true" else False
    )
    synth_df["hired"]          = synth_df["hired"].round().clip(0, 1).astype(int)
    synth_df["cv_score"]       = synth_df["cv_score"].clip(0, 100).round(2)
    synth_df["candidate_id"]   = [f"UK-SDV-{i+1:05d}" for i in range(len(synth_df))]
    synth_df["name"]           = [
        generate_name(g) for g in synth_df["gender"]
    ]
    synth_df["university"]     = None

    return synth_df


#  Public API for generating dataset

def generate_dataset(
    n_candidates: int = 2000,
    introduce_bias: bool = False,
    seed: int = 42, #seed set to 42 to have a consistent result
    use_sdv: bool = False,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate a synthetic UK recruitment dataset.

    Parameters
    ----------
    n_candidates : int
        Number of candidate records to generate.
    introduce_bias : bool
        If True, injects demographic bias into CV scoring.
    seed : int
        Random seed for reproducibility.
    use_sdv : bool
        If True and SDV is installed, uses GaussianCopula for richer synthesis.
        Falls back to built-in generator if SDV is not installed.
    output_path : str
        If provided, saves CSV to this path.
    """
    if use_sdv and SDV_AVAILABLE:
        df = _generate_with_sdv(n_candidates, introduce_bias, seed)
    else:
        if use_sdv and not SDV_AVAILABLE:
            print("[Data Pipeline] SDV not installed — using built-in generator.")
            print("               To enable SDV: pip install sdv")
        df = _generate_base(n_candidates, introduce_bias, seed)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[✓] Dataset saved: {output_path} ({len(df)} rows)")

    return df


def generate_both_datasets(
    output_dir: str = "data/",
    use_sdv: bool = False,
) -> tuple:
    """Generate both fair and biased datasets."""
    print("Generating fair dataset (no demographic bias)...")
    fair_df = generate_dataset(
        n_candidates=2000, introduce_bias=False, seed=42, use_sdv=use_sdv,
        output_path=os.path.join(output_dir, "uk_recruitment_fair.csv"),
    )

    print("Generating biased dataset (with demographic bias injected)...")
    biased_df = generate_dataset(
        n_candidates=2000, introduce_bias=True, seed=42, use_sdv=use_sdv,
        output_path=os.path.join(output_dir, "uk_recruitment_biased.csv"),
    )

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator":    "SDV GaussianCopula" if (use_sdv and SDV_AVAILABLE) else "Built-in ONS-calibrated",
        "sources": [
            "ONS Labour Force Survey 2023",
            "Annual Population Survey 2023",
            "EHRC Workforce Equality Data 2023",
            "NOMIS Labour Market Statistics",
        ],
        "n_fair":   len(fair_df),
        "n_biased": len(biased_df),
        "note": (
            "Synthetic data generated from real UK demographic distributions. "
            "No real personal data used."
        ),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Both datasets ready in '{output_dir}'")
    print(f"Fair dataset hire rate:   {fair_df['hired'].mean():.1%}")
    print(f"Biased dataset hire rate: {biased_df['hired'].mean():.1%}")
    return fair_df, biased_df


if __name__ == "__main__":
    generate_both_datasets()
