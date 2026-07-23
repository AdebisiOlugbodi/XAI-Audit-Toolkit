"""
XAI Audit Toolkit — Main Orchestrator
=======================================
Runs the full audit pipeline end-to-end.

Usage:
  python main.py --generate-data --bias --company "Acme Ltd" --role "Software Developer"
  python main.py --dataset data/uk_recruitment_biased.csv
  python main.py --generate-data --use-sdv           # use SDV if installed
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from modules.dependencies import print_dependency_status, SHAP_AVAILABLE, TORCH_AVAILABLE
if TORCH_AVAILABLE:
    from modules.neural_network import PyTorchNeuralNetwork
from data.data_pipeline import generate_both_datasets, generate_dataset
from modules.data_auditor import DataAuditor
from modules.bias_detector import BiasDetector
from modules.explainability import ExplainabilityEngine
from modules.report_generator import AuditReportGenerator

FEATURE_COLS = ["years_experience", "keyword_match_score", "employment_gap_years"]
LABEL_COL    = "hired"


def train_model(df, model_type="logistic"):
    X = df[FEATURE_COLS].fillna(0)
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "random_forest":
        model      = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "Random Forest"
    elif model_type == "neural_network":
        if not TORCH_AVAILABLE:
            print("[Warning] PyTorch not installed — falling back to Logistic Regression.")
            print("          Install with: pip install torch")
            model      = LogisticRegression(max_iter=500, random_state=42)
            model_name = "Logistic Regression (fallback)"
        else:
            model      = PyTorchNeuralNetwork(epochs=100, lr=0.001, batch_size=32, verbose=True)
            model_name = "Neural Network (PyTorch)"
    else:
        model      = LogisticRegression(max_iter=500, random_state=42)
        model_name = "Logistic Regression"

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        auc = None

    auc_str = f"{auc:.3f}" if auc else "N/A"
    print(f"[Model] Trained {model_name} | Accuracy: {acc*100:.1f}% | AUC: {auc_str}")

    model_info = {
        "name":                  f"CV Screening Model ({model_name})",
        "type":                  model_name,
        "version":               "1.0",
        "training_date":         datetime.now().strftime("%Y-%m-%d"),
        "features":              FEATURE_COLS,
        "training_dataset":      "Synthetic UK Recruitment Data (ONS-calibrated)",
        "n_train":               len(X_train),
        "accuracy":              f"{acc*100:.1f}%",
        "roc_auc":               auc_str,
        "explainability_method": "SHAP" if SHAP_AVAILABLE else "Permutation Importance",
        "architecture":          "Input→64→32→16→1 (BatchNorm+Dropout)" if model_type == "neural_network" and TORCH_AVAILABLE else "N/A",
    }
    return model, X_train, X_test, y_train, y_test, model_info

def run_audit(df, company_name="UK SME Ltd", job_role="General Vacancy",
              model_type="logistic", output_dir="reports/"):

    print("  XAI Audit Toolkit — UK SME Recruitment Compliance")

    print("\n[1/5] Training screening model...")
    model, X_train, X_test, y_train, y_test, model_info = train_model(df, model_type)

    X_full = df[FEATURE_COLS].fillna(0)
    df = df.copy()
    df["predicted_hired"] = model.predict(X_full)
    df["predicted_score"] = model.predict_proba(X_full)[:, 1]

    print("\n[2/5] Running data audit...")
    auditor      = DataAuditor(df, label_col=LABEL_COL)
    data_results = auditor.run_full_audit()

    print("\n[3/5] Running bias detection...")
    detector     = BiasDetector(df, y_true_col=LABEL_COL,
                                y_pred_col="predicted_hired", score_col="cv_score")
    bias_results = detector.run_full_bias_report()

    print("\n[4/5] Generating explanations...")
    engine            = ExplainabilityEngine(model, FEATURE_COLS, X_train=X_train, y_train=y_train)
    global_explanation = engine.global_explanation_text(X_train, y_train)

    sample_df       = df.sample(min(10, len(df)), random_state=42)
    sample_explanations = []
    for _, row in sample_df.iterrows():
        feat_row = row[FEATURE_COLS + ["candidate_id"]].copy()
        exp = engine.explain_decision(feat_row, int(row["predicted_hired"]))
        sample_explanations.append(exp)

    print("\n[5/5] Generating audit report...")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(
        output_dir, f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    )

    generator = AuditReportGenerator(company_name=company_name, job_role=job_role)
    report    = generator.generate(
        model_info=model_info,
        data_audit_results=data_results,
        bias_results=bias_results,
        explainability_summary=global_explanation,
        candidate_examples=sample_explanations,
        output_path=report_path,
    )

    json_path = report_path.replace(".md", "_data.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_info":        model_info,
            "overall_compliance": bias_results.get("overall_compliance", {}),
            "data_flags":        data_results.get("summary_flags", []),
            "audit_passed":      data_results.get("audit_passed", False),
        }, f, indent=2, default=str)

    rag = bias_results.get("overall_compliance", {}).get("rag_summary", "Unknown")
    print("  Audit Complete!")
    print(f"  Report:     {report_path}")
    print(f"  Data:       {json_path}")
    print(f"  Compliance: {rag}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="XAI Audit Toolkit for UK SME Recruitment")
    parser.add_argument("--dataset",       help="Path to CSV dataset", default=None)
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic datasets")
    parser.add_argument("--bias",          action="store_true", help="Use biased dataset (for testing)")
    parser.add_argument("--use-sdv",       action="store_true", help="Use SDV for data generation (if installed)")
    parser.add_argument("--company",       default="UK SME Ltd")
    parser.add_argument("--role",          default="General Vacancy")
    parser.add_argument("--model",         default="logistic", choices=["logistic", "random_forest", "neural_network"])
    parser.add_argument("--output",        default="reports/")
    args = parser.parse_args()

    # Always show dependency status at startup
    print_dependency_status()

    if args.generate_data:
        print("Generating synthetic UK recruitment datasets...")
        fair_df, biased_df = generate_both_datasets(
            output_dir="data/", use_sdv=args.use_sdv
        )
        df = biased_df if args.bias else fair_df
        print(f"Using {'biased' if args.bias else 'fair'} dataset ({len(df)} records)")
    elif args.dataset:
        print(f"Loading dataset: {args.dataset}")
        df = pd.read_csv(args.dataset)
        print(f"Loaded {len(df)} records")
    else:
        print("No dataset specified. Generating fair dataset...")
        df = generate_dataset(n_candidates=1000, introduce_bias=False,
                              seed=42, use_sdv=args.use_sdv)

    run_audit(df, company_name=args.company, job_role=args.role,
              model_type=args.model, output_dir=args.output)


if __name__ == "__main__":
    main()
