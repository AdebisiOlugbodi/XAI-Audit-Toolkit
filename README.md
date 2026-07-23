# Explainable AI Audit Toolkit for Bias Detection and Regulatory Compliance in UK SME Recruitment

> **Explainable AI Audit Toolkit for Bias Detection and Regulatory Compliance in UK SME Recruitment** — Detecting bias, generating explanations, and producing regulatory audit reports for UK SME recruitment processes.

---

## Overview

This toolkit helps UK small and medium enterprises (SMEs) comply with:
- **Equality Act 2010** — detecting indirect discrimination in automated CV screening
- **UK GDPR / DPA 2018** — transparency and right to explanation
- **ICO AI Auditing Framework (2023)** — explainability requirements

It is the artefact for the research project:
> *"Explainable AI Audit Toolkit for Bias Detection and Regulatory Compliance in UK SME Recruitment"*

---

## Project Structure

```
xai_audit_toolkit/
├── data/
│   ├── data_pipeline.py          # ONS-calibrated synthetic data generator
│   ├── uk_recruitment_fair.csv   # Generated fair dataset (2000 records)
│   └── uk_recruitment_biased.csv # Generated biased dataset (for testing)
│
├── modules/
│   ├── dependencies.py           # Central optional library checker
│   ├── neural_network.py         # PyTorch neural network model (optional)
│   ├── data_auditor.py           # Representation, proxy, label bias checks
│   ├── bias_detector.py          # DIR, demographic parity, equalised odds
│   ├── explainability.py         # Per-decision & global explanations
│   └── report_generator.py       # Markdown audit report generator
│
├── ui/
│   └── dashboard.py              # Streamlit web dashboard
│
├── reports/                      # Auto-generated audit reports (leave empty)
│
├── ethics_doc/
│   └── ethics_justification.md   # Full ethics statement for university submission
│
├── main.py                       # CLI orchestrator
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Set Up a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a Full Audit (CLI)

```bash
# Logistic Regression (default)
python main.py --generate-data --company "Your Company" --role "Software Developer"

# Random Forest
python main.py --generate-data --model random_forest --company "Your Company" --role "Software Developer"

# Neural Network (requires PyTorch — see Optional Libraries below)
python main.py --generate-data --model neural_network --company "Your Company" --role "Software Developer"

# Use biased dataset for bias detection testing
python main.py --generate-data --bias --model neural_network --company "Your Company" --role "Software Developer"

# Audit your own CSV dataset
python main.py --dataset data/uk_recruitment_fair.csv --company "Your Company" --role "Software Developer"

# Use SDV for richer synthetic data generation (requires sdv)
python main.py --generate-data --use-sdv --company "Your Company" --role "Software Developer"
```

### 4. Launch the Web Dashboard

```bash
streamlit run ui/dashboard.py
```

Then open http://localhost:8501 in your browser.

---

## CLI Reference

| Flag | Options | Description |
|------|---------|-------------|
| `--generate-data` | — | Generate synthetic UK datasets |
| `--dataset` | path/to/file.csv | Use your own CSV instead |
| `--bias` | — | Use biased dataset (for testing bias detection) |
| `--model` | `logistic` `random_forest` `neural_network` | Model type to train |
| `--use-sdv` | — | Use SDV for data generation (if installed) |
| `--company` | "Company Name" | Company name for the audit report |
| `--role` | "Job Role" | Job role being audited |
| `--output` | path/to/folder | Where to save reports (default: `reports/`) |

---

## Models

Three models are available across both the CLI and the Streamlit dashboard:

### Logistic Regression (default)
- No extra dependencies required
- Fast training, highly interpretable coefficients
- Best for: baseline audits, quick checks

```bash
python main.py --generate-data --model logistic
```

### Random Forest
- No extra dependencies required
- Handles non-linear relationships, robust to outliers
- Best for: more realistic recruitment screening scenarios

```bash
python main.py --generate-data --model random_forest
```

### Neural Network (PyTorch)
- Requires PyTorch: `pip install torch`
- 3-hidden-layer feedforward network with BatchNorm and Dropout
- Architecture: `Input → 64 → 32 → 16 → 1`
- Handles class imbalance automatically via weighted BCE loss
- Best for: demonstrating deep learning in a compliance context

```bash
# Install PyTorch first
pip install torch

# Then run
python main.py --generate-data --model neural_network --company "Your Company" --role "Your Role"
```

In the Streamlit dashboard, "Neural Network (PyTorch)" appears automatically in the Model Type dropdown once PyTorch is installed.

---

## Optional Libraries

The toolkit works out of the box with no optional libraries. Each optional library enhances a specific module and is detected automatically at startup.

| Library | What It Adds | Install |
|---------|-------------|---------|
| `torch` | PyTorch neural network model | `pip install torch` |
| `shap` | More accurate SHAP explanations | `pip install shap` |
| `fairlearn` | Extended fairness metrics (DPD, EOD) | `pip install fairlearn` |
| `aif360` | IBM AI Fairness 360 metrics | `pip install aif360` |
| `sdv` | Synthetic Data Vault data generation | `pip install sdv` |

Install all at once:
```bash
pip install torch shap fairlearn aif360 sdv
```

> **Note:** `torch` is a large download (~800MB). Install only when needed.

When running the CLI, the dependency status is printed at startup:
```
── Optional Dependencies ──────────────────────────────────────────
  torch        ✅ Installed
  shap         ✅ Installed
  fairlearn    ❌ Not installed — fallback active
  aif360       ❌ Not installed — fallback active
  sdv          ❌ Not installed — fallback active
──────────────────────────────────────────────────────────────────
```

---

## Modules

### `dependencies.py`
Central checker for all optional libraries. Exposes convenience booleans (`TORCH_AVAILABLE`, `SHAP_AVAILABLE`, etc.) imported by every other module. All graceful degradation logic flows from here.

### `neural_network.py`
PyTorch neural network wrapped in a fully sklearn-compatible interface. Implements `fit()`, `predict()`, `predict_proba()`, and `feature_importances_` so it plugs into the pipeline identically to the other models. Only loaded when PyTorch is installed.

### `data_auditor.py`
- Representation analysis — under-represented groups flagged at < 5%
- Proxy feature detection — region, university, employment gap, name
- Label bias — hire rate disparities by protected characteristic
- Missing data patterns

### `bias_detector.py`
- **Disparate Impact Ratio** — 4/5ths rule (< 0.80 = 🔴 RED)
- **Demographic Parity Difference** — max hire rate gap across groups
- **Equalised Odds** — TPR/FPR parity check
- **Score Distribution** — mean CV score gaps by protected group
- RAG (Red/Amber/Green) compliance ratings
- Extended metrics via `fairlearn` and `aif360` if installed

### `explainability.py`
- Global feature importance — SHAP if installed, permutation importance as fallback
- Per-candidate plain-English decision explanations
- Model-aware routing: uses SHAP → neural network importance → RF importance → logistic coefficients depending on what is available
- Factor contribution charts
- Compliant with UK GDPR Article 22 (right to explanation)

### `report_generator.py`
- Model card (includes architecture for neural network)
- Data audit summary
- Fairness metrics with RAG ratings
- Optional library status shown in report
- Recommendations
- Candidate explanation appendix

---

## Data Strategy

Synthetic data is calibrated to real UK statistics:

| Source | Data Used |
|--------|-----------|
| ONS Labour Force Survey 2023 | Gender & age distributions |
| ONS Annual Population Survey 2023 | Ethnicity breakdowns |
| NOMIS 2023 | Regional employment distributions |
| ONS Disability & Employment 2023 | Disability prevalence |
| EHRC Workforce Equality Data 2023 | Differential employment rates |

No real personal data is used at any stage. See `ethics_doc/ethics_justification.md` for the full ethics justification.

---

## Regulatory Alignment

| Regulation | Coverage |
|-----------|----------|
| Equality Act 2010, s.19 | Indirect discrimination / disparate impact |
| UK GDPR Article 22 | Automated decision-making rights |
| Data Protection Act 2018 | Lawful processing of recruitment data |
| ICO AI Auditing Framework | Transparency & explainability |
| EHRC Technical Guidance | Employer recruitment obligations |
| EU AI Act (2024) | Awareness — recruitment AI classified as high-risk |

---

## Ethics

This project follows ethical guidelines for AI fairness research:
- No real personal data used at any stage
- Synthetic data grounded in public ONS/APS statistics
- Biased dataset clearly labelled as intentionally constructed for testing bias detection only
- Full ethics justification in `ethics_doc/ethics_justification.md`

---

## Citation

If you use this toolkit in your research:

```
University of Wolverhampton. 
Explainable AI Audit Toolkit for Bias Detection and Regulatory Compliance in UK SME Recruitment.
Student Name: Adebisi Adebowale Olugbodi
Dissertation Supervisor: Olawande Olusegun
Available at: [repository URL]
August, 2026. 

```

---

## Licence

For academic and research use. Not for commercial deployment without review by a qualified employment law specialist.
