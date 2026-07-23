"""
Data Audit

Checks training data for:
  - Demographic imbalances (representation gaps)
  - Proxy features (variables that may encode protected characteristics)
  - Label bias (differential outcomes by group)
  - Missing data patterns
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


PROTECTED_CHARACTERISTICS = ["gender", "age", "ethnicity", "has_disability"]

# Features that may act as proxies for protected characteristics in UK context
KNOWN_PROXIES = {
    "region": "ethnicity/socioeconomic status (ONS spatial analysis)",
    "university": "socioeconomic background / ethnicity (Sutton Trust, 2023)",
    "employment_gap_years": "gender (parental leave), disability",
    "postcode": "ethnicity, socioeconomic status",
    "school_type": "socioeconomic background",
    "name": "ethnicity / gender (NLP-based inference risk)",
}


class DataAuditor:
    """
    Audits a recruitment dataset for fairness and data quality issues.
    """

    def __init__(self, df: pd.DataFrame, label_col: str = "hired"):
        self.df = df.copy()
        self.label_col = label_col
        self.results = {}

    # Representation Analysis 

    def check_representation(self) -> dict:
        """
        Check if protected groups are adequately represented in the dataset.
        Returns a dict of {feature: {group: count, pct, flag}}
        """
        rep = {}
        for col in PROTECTED_CHARACTERISTICS:
            if col not in self.df.columns:
                continue
            counts = self.df[col].value_counts()
            total = len(self.df)
            group_stats = {}
            for grp, cnt in counts.items():
                pct = cnt / total
                flag = "⚠️ Under-represented" if pct < 0.05 else "✓ OK"
                group_stats[str(grp)] = {
                    "count": int(cnt),
                    "pct": round(pct * 100, 2),
                    "flag": flag,
                }
            rep[col] = group_stats
        self.results["representation"] = rep
        return rep

    # Proxy Feature Detection 

    def check_proxy_features(self) -> dict:
        """
        Flags features present in the dataset that may act as demographic proxies.
        Also checks correlation between non-protected features and protected ones.
        """
        detected = {}

        # Check known proxies
        for feature, reason in KNOWN_PROXIES.items():
            if feature in self.df.columns:
                detected[feature] = {
                    "risk": "HIGH" if feature in ["name", "postcode", "region"] else "MEDIUM",
                    "reason": reason,
                    "recommendation": (
                        f"Consider removing or transforming '{feature}' "
                        f"before model training."
                    ),
                }

        # Check correlation between numeric features and protected characteristics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        protected_numeric = [c for c in PROTECTED_CHARACTERISTICS if c in numeric_cols]
        other_numeric = [c for c in numeric_cols if c not in PROTECTED_CHARACTERISTICS + [self.label_col]]

        correlations = {}
        for prot in protected_numeric:
            for feat in other_numeric:
                try:
                    corr = abs(self.df[prot].corr(self.df[feat]))
                    if corr > 0.3:
                        correlations[f"{feat} ↔ {prot}"] = {
                            "correlation": round(corr, 3),
                            "risk": "HIGH" if corr > 0.6 else "MEDIUM",
                        }
                except Exception:
                    pass

        self.results["proxy_features"] = {"known_proxies": detected, "correlations": correlations}
        return self.results["proxy_features"]

    # Label Bias Analysis 

    def check_label_bias(self) -> dict:
        """
        Checks differential outcome rates across protected groups.
        Computes hire rates and flags large disparities.
        """
        if self.label_col not in self.df.columns:
            return {}

        label_bias = {}
        overall_rate = self.df[self.label_col].mean()

        for col in PROTECTED_CHARACTERISTICS:
            if col not in self.df.columns:
                continue
            rates = self.df.groupby(col)[self.label_col].mean()
            group_results = {}
            for grp, rate in rates.items():
                disparity = rate / overall_rate if overall_rate > 0 else 1.0
                flag = "OK"
                if disparity < 0.8:
                    flag = "🔴 Disparate Impact Detected"
                elif disparity < 0.9:
                    flag = "🟡 Minor Disparity"
                group_results[str(grp)] = {
                    "hire_rate": round(rate * 100, 2),
                    "disparity_ratio": round(disparity, 3),
                    "flag": flag,
                }
            label_bias[col] = {
                "overall_hire_rate": round(overall_rate * 100, 2),
                "groups": group_results,
            }

        self.results["label_bias"] = label_bias
        return label_bias

    # Missing Data Analysis 

    def check_missing_data(self) -> dict:
        """
        Checks for missing values and whether missingness correlates with demographics.
        """
        missing = {}
        for col in self.df.columns:
            n_missing = self.df[col].isna().sum()
            if n_missing > 0:
                pct = n_missing / len(self.df) * 100
                missing[col] = {
                    "n_missing": int(n_missing),
                    "pct_missing": round(pct, 2),
                    "flag": "⚠️ Review" if pct > 5 else "Acceptable",
                }
        self.results["missing_data"] = missing
        return missing

    # Full Audit Report 

    def run_full_audit(self) -> dict:
        """Runs all audit checks and returns a combined results dict."""
        print("[Data Auditor] Running representation check...")
        self.check_representation()

        print("[Data Auditor] Detecting proxy features...")
        self.check_proxy_features()

        print("[Data Auditor] Checking label bias...")
        self.check_label_bias()

        print("[Data Auditor] Checking missing data...")
        self.check_missing_data()

        # Summary flags
        flags = []
        lb = self.results.get("label_bias", {})
        for char, data in lb.items():
            for grp, info in data.get("groups", {}).items():
                if "🔴" in info["flag"]:
                    flags.append(f"Disparate impact on {char}={grp} (ratio={info['disparity_ratio']})")

        pf = self.results.get("proxy_features", {})
        for feat, info in pf.get("known_proxies", {}).items():
            if info["risk"] == "HIGH":
                flags.append(f"High-risk proxy feature detected: '{feat}'")

        self.results["summary_flags"] = flags
        self.results["audit_passed"] = len([f for f in flags if "Disparate" in f or "HIGH" in f]) == 0

        print(f"\n[Data Auditor] Complete. {len(flags)} issue(s) flagged.")
        return self.results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.data_pipeline import generate_dataset

    df = generate_dataset(n_candidates=500, introduce_bias=True)
    auditor = DataAuditor(df)
    results = auditor.run_full_audit()

    print("Label Bias Summary ")
    for char, data in results["label_bias"].items():
        print(f"{char} (overall hire rate: {data['overall_hire_rate']}%):")
        for grp, info in data["groups"].items():
            print(f"{grp}: {info['hire_rate']}% hired | {info['flag']}")
