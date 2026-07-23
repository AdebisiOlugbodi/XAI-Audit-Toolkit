"""
Bias Detection & Fairness Metrics

Computes UK-regulation-aligned fairness metrics.

By default uses built-in implementations.
If 'fairlearn' or 'aif360' are installed, automatically adds extended metrics.

  pip install fairlearn aif360
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from modules.dependencies import FAIRLEARN_AVAILABLE, AIF360_AVAILABLE

if FAIRLEARN_AVAILABLE:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate,
    )

if AIF360_AVAILABLE:
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    except Exception:
        AIF360_AVAILABLE = False

PROTECTED_CHARACTERISTICS = ["gender", "ethnicity", "age_band", "has_disability"]

THRESHOLDS = {
    "disparate_impact_ratio": {"green": 0.90, "amber": 0.80},
    "demographic_parity_diff": {"green": 0.05, "amber": 0.10},
    "equal_opportunity_diff":  {"green": 0.05, "amber": 0.10},
}


def rag_rating(metric_name: str, value: float) -> str:
    t = THRESHOLDS.get(metric_name, {})
    if metric_name == "disparate_impact_ratio":
        if value >= t.get("green", 0.90): return "🟢 GREEN"
        elif value >= t.get("amber", 0.80): return "🟡 AMBER"
        else: return "🔴 RED"
    else:
        if value <= t.get("green", 0.05): return "🟢 GREEN"
        elif value <= t.get("amber", 0.10): return "🟡 AMBER"
        else: return "🔴 RED"


class BiasDetector:
    """
    Computes fairness metrics for a trained recruitment model's predictions.
    Automatically uses fairlearn/aif360 if installed.
    """

    def __init__(self, df, y_true_col="hired", y_pred_col="predicted_hired", score_col="cv_score"):
        self.df          = df.copy()
        self.y_true_col  = y_true_col
        self.y_pred_col  = y_pred_col
        self.score_col   = score_col
        self.results     = {}

    # ── Built-in Metrics ─────────────────────────────────────────────────────

    def disparate_impact_ratio(self, feature: str, privileged_group: str) -> dict:
        col    = self.y_pred_col if self.y_pred_col in self.df.columns else self.y_true_col
        groups = self.df.groupby(feature)[col].mean()
        priv_rate = groups.get(privileged_group, None)
        if priv_rate is None or priv_rate == 0:
            return {}
        results = {}
        for grp, rate in groups.items():
            if grp == privileged_group:
                continue
            ratio = rate / priv_rate
            results[str(grp)] = {
                "hire_rate":       round(rate * 100, 2),
                "privileged_rate": round(priv_rate * 100, 2),
                "dir":             round(ratio, 4),
                "rag":             rag_rating("disparate_impact_ratio", ratio),
                "interpretation":  f"{grp} candidates hired at {ratio:.1%} the rate of {privileged_group}",
            }
        return results

    def demographic_parity_difference(self, feature: str) -> dict:
        col   = self.y_pred_col if self.y_pred_col in self.df.columns else self.y_true_col
        rates = self.df.groupby(feature)[col].mean()
        dpd   = rates.max() - rates.min()
        return {
            "max_hire_rate_group": str(rates.idxmax()),
            "min_hire_rate_group": str(rates.idxmin()),
            "max_rate":            round(rates.max() * 100, 2),
            "min_rate":            round(rates.min() * 100, 2),
            "demographic_parity_difference": round(dpd, 4),
            "rag":                 rag_rating("demographic_parity_diff", dpd),
        }

    def equalised_odds(self, feature: str) -> dict:
        if self.y_pred_col not in self.df.columns:
            return {"error": "Predictions column not found"}
        results     = {}
        tpr_values  = []
        fpr_values  = []
        for grp, group_df in self.df.groupby(feature):
            y_true = group_df[self.y_true_col]
            y_pred = group_df[self.y_pred_col]
            if len(y_true.unique()) < 2 or len(y_pred.unique()) < 2:
                continue
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tpr_values.append(tpr)
                fpr_values.append(fpr)
                results[str(grp)] = {"tpr": round(tpr, 4), "fpr": round(fpr, 4), "n": len(group_df)}
            except Exception:
                pass
        if tpr_values:
            tpr_diff = max(tpr_values) - min(tpr_values)
            fpr_diff = max(fpr_values) - min(fpr_values)
            results["summary"] = {
                "tpr_difference":         round(tpr_diff, 4),
                "fpr_difference":         round(fpr_diff, 4),
                "equal_opportunity_rag":  rag_rating("equal_opportunity_diff", tpr_diff),
            }
        return results

    def score_distribution_by_group(self, feature: str) -> dict:
        if self.score_col not in self.df.columns:
            return {}
        results = {}
        for grp, group_df in self.df.groupby(feature):
            scores = group_df[self.score_col].dropna()
            results[str(grp)] = {
                "mean_score":   round(scores.mean(), 2),
                "median_score": round(scores.median(), 2),
                "std_score":    round(scores.std(), 2),
                "n":            len(scores),
            }
        means = {k: v["mean_score"] for k, v in results.items()}
        if means:
            gap = max(means.values()) - min(means.values())
            results["summary"] = {
                "score_gap": round(gap, 2),
                "flag": "🔴 Large score gap — investigate bias" if gap > 5 else "🟢 Acceptable",
            }
        return results

    # ── Fairlearn Metrics (optional) ─────────────────────────────────────────

    def fairlearn_metrics(self, feature: str) -> dict:
        """Extended metrics using fairlearn. Only called if fairlearn is installed."""
        if not FAIRLEARN_AVAILABLE:
            return {"note": "Install fairlearn for extended metrics: pip install fairlearn"}

        if self.y_pred_col not in self.df.columns:
            return {"error": "Predictions column not found"}

        try:
            y_true    = self.df[self.y_true_col]
            y_pred    = self.df[self.y_pred_col]
            sensitive = self.df[feature].astype(str)

            dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
            eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

            return {
                "source":                        "fairlearn",
                "demographic_parity_difference": round(float(dpd), 4),
                "equalized_odds_difference":     round(float(eod), 4),
                "rag_dpd": rag_rating("demographic_parity_diff", abs(dpd)),
                "rag_eod": rag_rating("equal_opportunity_diff", abs(eod)),
            }
        except Exception as e:
            return {"error": f"Fairlearn metrics failed: {e}"}

    # ── AIF360 Metrics (optional) ────────────────────────────────────────────

    def aif360_metrics(self, feature: str, privileged_group: str) -> dict:
        """Extended metrics using IBM AI Fairness 360. Only called if aif360 is installed."""
        if not AIF360_AVAILABLE:
            return {"note": "Install aif360 for extended metrics: pip install aif360"}

        try:
            df_aif = self.df[[feature, self.y_true_col]].copy()
            df_aif[feature] = (df_aif[feature] == privileged_group).astype(int)

            dataset = BinaryLabelDataset(
                df=df_aif,
                label_names=[self.y_true_col],
                protected_attribute_names=[feature],
            )
            metric = BinaryLabelDatasetMetric(
                dataset,
                privileged_groups=[{feature: 1}],
                unprivileged_groups=[{feature: 0}],
            )
            return {
                "source":                    "aif360",
                "disparate_impact":          round(float(metric.disparate_impact()), 4),
                "statistical_parity_diff":   round(float(metric.statistical_parity_difference()), 4),
                "consistency":               round(float(metric.consistency()[0]), 4),
            }
        except Exception as e:
            return {"error": f"AIF360 metrics failed: {e}"}

    # ── Full Report ───────────────────────────────────────────────────────────

    def run_full_bias_report(self) -> dict:
        """Runs all bias checks across all protected characteristics."""
        privileged = {
            "gender":        "Male",
            "ethnicity":     "White British",
            "age_band":      "25-34",
            "has_disability": False,
        }

        # Print which optional libraries are active
        if FAIRLEARN_AVAILABLE:
            print("[Bias Detector] ✅ fairlearn detected — extended metrics enabled")
        else:
            print("[Bias Detector] ℹ️  fairlearn not installed — using built-in metrics")
            print("               Install with: pip install fairlearn")

        if AIF360_AVAILABLE:
            print("[Bias Detector] ✅ aif360 detected — IBM fairness metrics enabled")
        else:
            print("[Bias Detector] ℹ️  aif360 not installed — using built-in metrics")
            print("               Install with: pip install aif360")

        report = {}

        for feature in PROTECTED_CHARACTERISTICS:
            if feature not in self.df.columns:
                continue
            print(f"[Bias Detector] Analysing: {feature}")
            priv = privileged.get(feature)

            feature_report = {}
            feature_report["disparate_impact_ratio"] = (
                self.disparate_impact_ratio(feature, str(priv)) if priv is not None else {}
            )
            feature_report["demographic_parity"]    = self.demographic_parity_difference(feature)
            feature_report["equalised_odds"]        = self.equalised_odds(feature)
            feature_report["score_distributions"]   = self.score_distribution_by_group(feature)

            # Optional extended metrics
            feature_report["fairlearn_metrics"] = self.fairlearn_metrics(feature)
            if priv is not None:
                feature_report["aif360_metrics"] = self.aif360_metrics(feature, str(priv))

            report[feature] = feature_report

        # Overall compliance
        rag_counts = {"🟢 GREEN": 0, "🟡 AMBER": 0, "🔴 RED": 0}
        for feat_data in report.values():
            dp  = feat_data.get("demographic_parity", {})
            rag = dp.get("rag", "")
            if rag in rag_counts:
                rag_counts[rag] += 1

        if rag_counts["🔴 RED"] > 0:
            overall = "🔴 RED — Significant bias detected. Regulatory action required."
        elif rag_counts["🟡 AMBER"] > 0:
            overall = "🟡 AMBER — Some disparities detected. Review and monitor."
        else:
            overall = "🟢 GREEN — No significant bias detected."

        report["overall_compliance"] = {
            "rag_summary":  overall,
            "green_count":  rag_counts["🟢 GREEN"],
            "amber_count":  rag_counts["🟡 AMBER"],
            "red_count":    rag_counts["🔴 RED"],
            "fairlearn_active": FAIRLEARN_AVAILABLE,
            "aif360_active":    AIF360_AVAILABLE,
            "regulatory_references": [
                "Equality Act 2010 — s.19 Indirect Discrimination",
                "ICO AI Auditing Framework (2023)",
                "EHRC Technical Guidance on Employment",
                "4/5ths Rule (widely adopted in UK practice)",
            ],
        }

        self.results = report
        return report
