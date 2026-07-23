"""
Model Explainability

Provides per-decision and global model explanations.

By default uses scikit-learn permutation importance (no extra dependencies).
If 'shap' is installed, automatically upgrades to SHAP explanations.

  pip install shap
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

from modules.dependencies import SHAP_AVAILABLE, TORCH_AVAILABLE
if TORCH_AVAILABLE:
    from modules.neural_network import PyTorchNeuralNetwork

if SHAP_AVAILABLE:
    import shap

FEATURE_LABELS = {
    "years_experience":       "Years of Work Experience",
    "keyword_match_score":    "CV Keyword Match",
    "employment_gap_years":   "Employment Gap (Years)",
    "cv_score":               "Overall CV Score",
    "age":                    "Age",
    "has_disability":         "Disability Status",
    "education_level":        "Education Level",
    "university":             "University Attended",
    "region":                 "Region of Residence",
}


class ExplainabilityEngine:
    """
    Generates local (per-decision) and global (model-level) explanations.
    Automatically uses SHAP if installed, otherwise falls back to
    permutation importance and coefficient-based explanations.
    """

    def __init__(self, model, feature_names: list, X_train=None, y_train=None):
        self.model              = model
        self.feature_names      = feature_names
        self.X_train            = X_train
        self.y_train            = y_train
        self._global_importance = None
        self._shap_explainer    = None
        # Store aligned training data for fallback — X_train/y_train must always match
        self._X_train_ref = X_train
        self._y_train_ref = y_train

        # Set up SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                if isinstance(model, RandomForestClassifier):
                    self._shap_explainer = shap.TreeExplainer(model)
                    print("[Explainability] ✅ Using SHAP TreeExplainer (Random Forest)")
                elif isinstance(model, LogisticRegression):
                    self._shap_explainer = shap.LinearExplainer(model, X_train)
                    print("[Explainability] ✅ Using SHAP LinearExplainer (Logistic Regression)")
                else:
                    self._shap_explainer = shap.KernelExplainer(
                        model.predict_proba, shap.sample(X_train, 50)
                    )
                    print("[Explainability] ✅ Using SHAP KernelExplainer (generic model)")
            except Exception as e:
                print(f"[Explainability] SHAP setup failed ({e}) — using fallback.")
                self._shap_explainer = None
        else:
            print("[Explainability] ℹ️  SHAP not installed — using permutation importance.")
            print("               Install with: pip install shap")

    # ── Global Importance ────────────────────────────────────────────────────

    def compute_global_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Computes global feature importance.
        Uses SHAP mean absolute values if available, otherwise permutation importance.
        X and y must be the same split (both test or both train — never mixed).
        """
        if SHAP_AVAILABLE and self._shap_explainer is not None:
            importance_df = self._shap_global_importance(X)
        else:
            importance_df = self._permutation_global_importance(X, y)

        self._global_importance = importance_df
        return importance_df

    def _shap_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Global importance from mean absolute SHAP values."""
        try:
            shap_values = self._shap_explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class for binary
            mean_abs = np.abs(shap_values).mean(axis=0)
            df = pd.DataFrame({
                "feature":         self.feature_names,
                "importance_mean": mean_abs,
                "importance_std":  np.abs(shap_values).std(axis=0),
                "method":          "SHAP",
            }).sort_values("importance_mean", ascending=False)
        except Exception as e:
            print(f"[Explainability] SHAP global importance failed ({e}) — using fallback.")
            # Use stored training data to ensure X and y are always aligned
            if self._X_train_ref is not None and self._y_train_ref is not None:
                return self._permutation_global_importance(self._X_train_ref, self._y_train_ref)
            return self._permutation_global_importance(X, self.y_train)

        df["label"] = df["feature"].map(
            lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title())
        )
        total = df["importance_mean"].clip(lower=0).sum()
        df["pct"] = (df["importance_mean"].clip(lower=0) / total * 100).round(1) if total > 0 else 0
        return df

    def _permutation_global_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Global importance from sklearn permutation importance."""
        result = permutation_importance(
            self.model, X, y, n_repeats=10, random_state=42, scoring="roc_auc"
        )
        df = pd.DataFrame({
            "feature":         self.feature_names,
            "importance_mean": result.importances_mean,
            "importance_std":  result.importances_std,
            "method":          "Permutation Importance",
        }).sort_values("importance_mean", ascending=False)
        df["label"] = df["feature"].map(
            lambda f: FEATURE_LABELS.get(f, f.replace("_", " ").title())
        )
        total = df["importance_mean"].clip(lower=0).sum()
        df["pct"] = (df["importance_mean"].clip(lower=0) / total * 100).round(1) if total > 0 else 0
        return df

    def global_explanation_text(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Returns a plain-English summary of what drives model decisions."""
        imp    = self.compute_global_importance(X, y)
        top3   = imp.head(3)
        method = imp["method"].iloc[0] if "method" in imp.columns else "analysis"

        lines = [f"The recruitment model makes its decisions primarily based on "
                 f"(using {method}):\n"]
        for i, row in enumerate(top3.itertuples(), 1):
            lines.append(
                f"  {i}. {row.label} — accounts for approximately {row.pct:.0f}% of the decision weight"
            )

        remaining = imp.iloc[3:]
        if len(remaining) > 0:
            others = ", ".join(remaining["label"].tolist())
            lines.append(f"\n  Other contributing factors: {others}")

        return "\n".join(lines)

    # ── Local (Per-Candidate) Explanation ────────────────────────────────────

    def explain_decision(self, candidate_row: pd.Series, decision: int) -> dict:
        """
        Generates a human-readable explanation for an individual hiring decision.
        Uses SHAP values if available, otherwise coefficient/importance-based fallback.
        """
        if self._global_importance is None and self.X_train is not None:
            self.compute_global_importance(self.X_train, self.y_train)

        if SHAP_AVAILABLE and self._shap_explainer is not None:
            contributions = self._shap_local_contributions(candidate_row)
        elif isinstance(self.model, LogisticRegression):
            contributions = self._logistic_contributions(candidate_row)
        elif isinstance(self.model, RandomForestClassifier):
            contributions = self._rf_contributions(candidate_row)
        elif TORCH_AVAILABLE and isinstance(self.model, PyTorchNeuralNetwork):
            contributions = self._nn_contributions(candidate_row)
        else:
            contributions = self._permutation_contributions(candidate_row)

        contributions_sorted = sorted(
            contributions, key=lambda x: abs(x["contribution"]), reverse=True
        )

        outcome      = "shortlisted" if decision == 1 else "not shortlisted"
        top_positive = [c for c in contributions_sorted if c["contribution"] > 0][:2]
        top_negative = [c for c in contributions_sorted if c["contribution"] < 0][:2]

        if SHAP_AVAILABLE and self._shap_explainer:
            method_note = "SHAP values"
        elif TORCH_AVAILABLE and isinstance(self.model, PyTorchNeuralNetwork):
            method_note = "neural network feature importance"
        else:
            method_note = "feature analysis"
        lines = [f"This candidate was {outcome} (based on {method_note}).", ""]

        if top_positive:
            pos_texts = [f"{c['label']} ({c['value_text']})" for c in top_positive]
            lines.append(
                f"Positive factors: {' and '.join(pos_texts)} contributed to their score."
            )
        if top_negative:
            neg_texts = [f"{c['label']} ({c['value_text']})" for c in top_negative]
            lines.append(
                f"Areas of concern: {' and '.join(neg_texts)} reduced their score."
            )

        lines.append(
            "\nNote: This explanation is generated automatically. "
            "All hiring decisions should be reviewed by a qualified HR professional."
        )

        return {
            "decision":      outcome,
            "decision_code": decision,
            "plain_english": "\n".join(lines),
            "factors":       contributions_sorted,
            "candidate_id":  candidate_row.get("candidate_id", "Unknown"),
            "method": (
                "SHAP" if (SHAP_AVAILABLE and self._shap_explainer)
                else "Neural Network" if (TORCH_AVAILABLE and isinstance(self.model, PyTorchNeuralNetwork))
                else "Permutation Importance"
            ),
        }

    def _shap_local_contributions(self, row: pd.Series) -> list:
        """Per-candidate contributions using SHAP values."""
        try:
            feat_vals = pd.DataFrame(
                [row[self.feature_names].values], columns=self.feature_names
            ).fillna(0)
            shap_values = self._shap_explainer.shap_values(feat_vals)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            values = shap_values[0]

            contributions = []
            for i, feat in enumerate(self.feature_names):
                val    = row.get(feat, 0)
                contrib = float(values[i])
                contributions.append({
                    "feature":   feat,
                    "label":     FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                    "value":     val,
                    "value_text": self._format_value(feat, val),
                    "contribution": round(contrib, 4),
                    "direction": "positive" if contrib > 0 else "negative",
                })
            return contributions
        except Exception as e:
            print(f"[Explainability] SHAP local explanation failed ({e}) — using fallback.")
            return self._select_fallback_contributions(row)

    def _logistic_contributions(self, row: pd.Series) -> list:
        coefs = self.model.coef_[0]
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val    = row.get(feat, 0) if feat in row.index else 0
            if pd.isna(val): val = 0
            contrib = float(coefs[i] * val)
            contributions.append({
                "feature":    feat,
                "label":      FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "value":      val,
                "value_text": self._format_value(feat, val),
                "contribution": round(contrib, 4),
                "direction":  "positive" if contrib > 0 else "negative",
            })
        return contributions

    def _rf_contributions(self, row: pd.Series) -> list:
        importances = self.model.feature_importances_
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val      = row.get(feat, 0) if feat in row.index else 0
            if pd.isna(val): val = 0
            mean_val = self.X_train[feat].mean() if self.X_train is not None else 1
            contrib  = float(importances[i] * (val - mean_val))
            contributions.append({
                "feature":    feat,
                "label":      FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "value":      val,
                "value_text": self._format_value(feat, val),
                "contribution": round(contrib, 4),
                "direction":  "positive" if contrib > 0 else "negative",
            })
        return contributions


    def _select_fallback_contributions(self, row: pd.Series) -> list:
        """Picks the right contribution method based on the actual model type."""
        if isinstance(self.model, LogisticRegression):
            return self._logistic_contributions(row)
        elif isinstance(self.model, RandomForestClassifier):
            return self._rf_contributions(row)
        elif TORCH_AVAILABLE and isinstance(self.model, PyTorchNeuralNetwork):
            return self._nn_contributions(row)
        else:
            return self._permutation_contributions(row)

    def _nn_contributions(self, row: pd.Series) -> list:
        """
        Per-candidate contributions for PyTorchNeuralNetwork.
        Uses the model's internal feature_importances_ (permutation-based),
        scaled by how far each feature value is from the training mean.
        """
        importances = self.model.feature_importances_
        contributions = []
        for i, feat in enumerate(self.feature_names):
            val      = row.get(feat, 0) if feat in row.index else 0
            if pd.isna(val): val = 0
            mean_val = self.X_train[feat].mean() if self.X_train is not None else 0
            # Direction: positive if above average (likely helps), negative if below
            contrib  = float(importances[i] * (val - mean_val))
            contributions.append({
                "feature":      feat,
                "label":        FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "value":        val,
                "value_text":   self._format_value(feat, val),
                "contribution": round(contrib, 4),
                "direction":    "positive" if contrib > 0 else "negative",
            })
        return contributions

    def _permutation_contributions(self, row: pd.Series) -> list:
        if self._global_importance is None:
            return []
        contributions = []
        for _, imp_row in self._global_importance.iterrows():
            feat   = imp_row["feature"]
            val    = row.get(feat, 0)
            if pd.isna(val): val = 0
            contrib = float(imp_row["importance_mean"] * np.sign(val))
            contributions.append({
                "feature":    feat,
                "label":      FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "value":      val,
                "value_text": self._format_value(feat, val),
                "contribution": round(contrib, 4),
                "direction":  "positive" if contrib > 0 else "negative",
            })
        return contributions

    def _format_value(self, feature: str, value) -> str:
        if feature == "years_experience":    return f"{int(value)} year(s)"
        elif feature == "keyword_match_score": return f"{value * 100:.0f}% match"
        elif feature == "employment_gap_years": return f"{value:.1f} year(s) gap" if value > 0 else "No gap"
        elif feature == "cv_score":          return f"{value:.1f}/100"
        elif feature == "has_disability":    return "Declared disability" if value else "No declared disability"
        elif feature == "age":               return f"Age {int(value)}"
        else:                                return str(value)

    def explain_batch(self, X: pd.DataFrame, predictions: np.ndarray) -> list:
        explanations = []
        for i, (idx, row) in enumerate(X.iterrows()):
            exp = self.explain_decision(row, int(predictions[i]))
            explanations.append(exp)
        return explanations

    def get_shap_values(self, X: pd.DataFrame):
        """
        Returns raw SHAP values for a dataset.
        Returns None if SHAP is not installed.
        Used by the Streamlit dashboard for SHAP summary plots.
        """
        if not SHAP_AVAILABLE or self._shap_explainer is None:
            return None
        try:
            shap_values = self._shap_explainer.shap_values(X)
            if isinstance(shap_values, list):
                return shap_values[1]
            return shap_values
        except Exception:
            return None
