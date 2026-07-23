"""
XAI Audit Toolkit — Streamlit Dashboard

SME-friendly web interface for running recruitment AI audits.
Run with: streamlit run ui/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.dependencies import (
    check_optional_dependencies,
    get_dependency_status_list,
    SHAP_AVAILABLE,
    TORCH_AVAILABLE,
)
if TORCH_AVAILABLE:
    from modules.neural_network import PyTorchNeuralNetwork
from data.data_pipeline import generate_dataset
from modules.data_auditor import DataAuditor
from modules.bias_detector import BiasDetector
from modules.explainability import ExplainabilityEngine
from modules.report_generator import AuditReportGenerator

if SHAP_AVAILABLE:
    import shap

# ── Theme-aware matplotlib defaults ────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor":   "none",
    "axes.edgecolor":   "gray",
    "axes.labelcolor":  "gray",
    "xtick.color":      "gray",
    "ytick.color":      "gray",
    "text.color":       "gray",
    "grid.alpha":       0.3,
})

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Audit Toolkit — UK SME Recruitment",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
    }
    .rag-green { background:#d4edda; border-left:4px solid #28a745; padding:0.8rem; border-radius:6px; }
    .rag-amber { background:#fff3cd; border-left:4px solid #ffc107; padding:0.8rem; border-radius:6px; }
    .rag-red   { background:#f8d7da; border-left:4px solid #dc3545; padding:0.8rem; border-radius:6px; }
    .section-header { font-size:1.4rem; font-weight:700; color:#1e3a5f; margin:1.5rem 0 0.5rem 0; }
    .explanation-box {
        border: 1px solid rgba(128,128,128,0.3);
        border-left: 4px solid #2d6a9f;
        border-radius:8px; padding:1.2rem; font-size:0.95rem; line-height:1.6;
    }
    .dep-installed   { color:#28a745; font-weight:600; }
    .dep-missing     { color:#888;    font-weight:400; }
    .stButton>button { background:#2d6a9f; color:white; border-radius:8px; border:none; font-weight:600; }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ["years_experience", "keyword_match_score", "employment_gap_years"]
LABEL_COL    = "hired"

# ── Session state init ────────────────────────────────────────────────────────
for key in ["df", "model", "X_train", "X_test", "y_train", "y_test",
            "model_info", "data_results", "bias_results", "engine",
            "global_explanation", "audit_done", "company_name", "job_role"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["audit_done"] is None:
    st.session_state["audit_done"] = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    company_name = st.text_input("Company Name", value="Acme Digital Ltd")
    job_role     = st.text_input("Job Role",     value="Software Developer")

    st.markdown("### Data Source")
    data_source = st.radio("Choose dataset", [
        "Generate Fair (No Bias)",
        "Generate Biased (Bias Testing)",
        "Upload My Own CSV",
    ])
    uploaded_file = None
    if data_source == "Upload My Own CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    n_records  = st.slider("Synthetic Records", 500, 3000, 1000, 100)
    model_options = ["Logistic Regression", "Random Forest"]
    if TORCH_AVAILABLE:
        model_options.append("Neural Network (PyTorch)")
    else:
        st.caption("PyTorch not installed — Neural Network unavailable.\n`pip install torch` to enable.")
    model_type = st.selectbox("Model Type", model_options)

    # SDV option — only show if sdv is installed
    deps = check_optional_dependencies()
    use_sdv = False
    if deps["sdv"]:
        use_sdv = st.checkbox("Use SDV for data generation ✅", value=False)
    else:
        st.caption("SDV not installed — using built-in generator.\n`pip install sdv` to enable.")

    run_btn = st.button("🚀 Run Audit", use_container_width=True)

    # ── Optional Library Status ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔧 Optional Libraries")
    dep_list = get_dependency_status_list()
    for dep in dep_list:
        if dep["available"]:
            st.markdown(f'<span class="dep-installed">✅ {dep["library"]}</span> — {dep["description"]}',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="dep-missing">❌ {dep["library"]}</span> — {dep["description"]}',
                        unsafe_allow_html=True)
            st.caption(f"`{dep['install']}`")

    st.markdown("---")
    st.caption(
        "XAI Audit Toolkit v1.0\n\n"
        "Aligned with:\n"
        "- Equality Act 2010\n"
        "- UK GDPR / DPA 2018\n"
        "- ICO AI Auditing Framework\n\n"
        "Data calibrated to ONS/LFS 2023."
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:2rem;">🔍 XAI Audit Toolkit</h1>
    <p style="margin:0.5rem 0 0 0;opacity:0.9;font-size:1.1rem;">
        Explainable AI Compliance for UK SME Recruitment
    </p>
</div>
""", unsafe_allow_html=True)

# ── Run Audit ─────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running full audit pipeline..."):
        # Load data
        if data_source == "Upload My Own CSV" and uploaded_file:
            df = pd.read_csv(uploaded_file)
        elif data_source == "Generate Biased (Bias Testing)":
            df = generate_dataset(n_candidates=n_records, introduce_bias=True,
                                  seed=42, use_sdv=use_sdv)
        else:
            df = generate_dataset(n_candidates=n_records, introduce_bias=False,
                                  seed=42, use_sdv=use_sdv)

        # Train model
        X = df[FEATURE_COLS].fillna(0)
        y = df[LABEL_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "Neural Network (PyTorch)":
            if TORCH_AVAILABLE:
                model = PyTorchNeuralNetwork(epochs=100, lr=0.001, batch_size=32, verbose=False)
            else:
                st.warning("PyTorch not installed — falling back to Logistic Regression. Run: pip install torch")
                model = LogisticRegression(max_iter=500, random_state=42)
        else:
            model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train, y_train)

        df = df.copy()
        df["predicted_hired"] = model.predict(X)
        df["predicted_score"] = model.predict_proba(X)[:, 1]

        y_pred_test  = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec  = recall_score(y_test, y_pred_test, zero_division=0)
        f1   = f1_score(y_test, y_pred_test, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
        tn_v, fp_v, fn_v, tp_v = cm.ravel()

        try:
            auc = roc_auc_score(y_test, y_proba_test)
            auc_str = f"{auc:.4f}"
        except Exception:
            auc_str = "N/A"

        model_info = {
            "name":                  f"CV Screening ({model_type})",
            "type":                  model_type,
            "version":               "1.0",
            "training_date":         datetime.now().strftime("%Y-%m-%d"),
            "features":              FEATURE_COLS,
            "n_train":               len(X_train),
            "n_test":                len(X_test),
            "accuracy":              f"{acc*100:.2f}%",
            "precision":             f"{prec:.4f}",
            "recall":                f"{rec:.4f}",
            "f1_score":              f"{f1:.4f}",
            "roc_auc":               auc_str,
            "tp":                    int(tp_v),
            "fp":                    int(fp_v),
            "tn":                    int(tn_v),
            "fn":                    int(fn_v),
            "training_dataset":      data_source,
            "explainability_method": "SHAP" if SHAP_AVAILABLE else "Permutation Importance",
            "architecture": "Input→64→32→16→1 (BatchNorm+Dropout)" if model_type == "Neural Network (PyTorch)" and TORCH_AVAILABLE else "N/A",
        }

        auditor      = DataAuditor(df, label_col=LABEL_COL)
        data_results = auditor.run_full_audit()

        detector     = BiasDetector(df, y_true_col=LABEL_COL,
                                    y_pred_col="predicted_hired", score_col="cv_score")
        bias_results = detector.run_full_bias_report()

        engine            = ExplainabilityEngine(model, FEATURE_COLS, X_train=X_train, y_train=y_train)
        global_explanation = engine.global_explanation_text(X_train, y_train)

        st.session_state.update({
            "df": df, "model": model,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "model_info": model_info,
            "data_results": data_results,
            "bias_results": bias_results,
            "engine": engine,
            "global_explanation": global_explanation,
            "audit_done": True,
            "company_name": company_name,
            "job_role": job_role,
        })
    st.success("✅ Audit complete!")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.get("audit_done"):
    df             = st.session_state["df"]
    model          = st.session_state["model"]
    model_info     = st.session_state["model_info"]
    data_results   = st.session_state["data_results"]
    bias_results   = st.session_state["bias_results"]
    engine         = st.session_state["engine"]
    global_exp     = st.session_state["global_explanation"]
    X_train        = st.session_state["X_train"]
    X_test         = st.session_state["X_test"]
    y_train        = st.session_state["y_train"]
    y_test         = st.session_state["y_test"]
    cn             = st.session_state.get("company_name", company_name)
    jr             = st.session_state.get("job_role", job_role)
    compliance     = bias_results.get("overall_compliance", {})
    rag_text       = compliance.get("rag_summary", "Unknown")

    # Compliance banner
    css = "rag-red" if "🔴" in rag_text else ("rag-amber" if "🟡" in rag_text else "rag-green")
    st.markdown(f'<div class="{css}"><b>Overall Compliance: {rag_text}</b></div>',
                unsafe_allow_html=True)
    st.markdown("")

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Candidates",  f"{len(df):,}")
    k2.metric("Hired",             f"{int(df[LABEL_COL].sum()):,}",
              f"{df[LABEL_COL].mean()*100:.1f}% rate")
    k3.metric("Accuracy",          model_info["accuracy"])
    k4.metric("ROC-AUC",           model_info["roc_auc"])
    k5.metric("F1-Score",          model_info["f1_score"])
    k6.metric("Issues Flagged",    len(data_results.get("summary_flags", [])))

    st.markdown("---")
    tabs = st.tabs(["📋 Dataset", "📊 Bias Detection", "🔍 Data Audit",
                    "💡 Explainability", "👤 Candidate Lookup", "📄 Report"])

    # ── TAB 0: Dataset ────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<p class="section-header">Generated Dataset Overview</p>',
                    unsafe_allow_html=True)

        # ── Dataset info banner ───────────────────────────────────────────
        bias_flag = "⚠️ BIASED — Demographic Penalties Injected" if "Bias" in data_source else "✅ FAIR — No Demographic Penalties"
        st.info(f"**Dataset:** {data_source}  |  **Records:** {len(df):,}  |  **Features:** {len(df.columns)}  |  **Status:** {bias_flag}")

        # ── Model performance card ────────────────────────────────────────
        st.markdown("#### 🤖 Model Performance on Hold-Out Test Set")
        st.caption(f"All metrics computed on {model_info['n_test']} held-out test records "
                   f"(20% stratified split, never seen during training)")

        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Accuracy",  model_info["accuracy"])
        mc2.metric("Precision", model_info["precision"],
                   help="Of all predicted hires, what fraction were truly hireable?")
        mc3.metric("Recall",    model_info["recall"],
                   help="Of all truly hireable candidates, what fraction did the model find?")
        mc4.metric("F1-Score",  model_info["f1_score"],
                   help="Harmonic mean of Precision and Recall")
        mc5.metric("ROC-AUC",   model_info["roc_auc"],
                   help="Area under ROC curve — threshold-independent performance measure")
        mc6.metric("Train Records", f"{model_info['n_train']:,}")

        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        cm_col1, cm_col2 = st.columns([1, 2])
        with cm_col1:
            cm_data = {
                "": ["Predicted: NOT HIRED", "Predicted: HIRED"],
                "Actual: NOT HIRED": [model_info["tn"], model_info["fp"]],
                "Actual: HIRED":     [model_info["fn"], model_info["tp"]],
            }
            cm_df = pd.DataFrame(cm_data).set_index("")
            st.dataframe(cm_df, use_container_width=True)
            st.caption(f"TP={model_info['tp']}  FP={model_info['fp']}  "
                       f"TN={model_info['tn']}  FN={model_info['fn']}")

        with cm_col2:
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax = plt.subplots(figsize=(4, 3))
            cm_vals = np.array([[model_info["tn"], model_info["fp"]],
                                 [model_info["fn"], model_info["tp"]]])
            im = ax.imshow(cm_vals, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, ax=ax)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred: 0", "Pred: 1"])
            ax.set_yticklabels(["Act: 0", "Act: 1"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm_vals[i, j]),
                            ha="center", va="center", fontsize=14, fontweight="bold",
                            color="white" if cm_vals[i, j] > cm_vals.max()/2 else "black")
            ax.set_title("Confusion Matrix", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # ── Dataset preview ───────────────────────────────────────────────
        st.markdown("#### 📄 Dataset Preview (First 20 Records)")
        display_cols = ["candidate_id", "gender", "age", "ethnicity", "region",
                        "has_disability", "education_level", "years_experience",
                        "employment_gap_years", "keyword_match_score",
                        "cv_score", "hired", "predicted_hired", "predicted_score"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].head(20), use_container_width=True, hide_index=True)

        # ── Feature statistics ────────────────────────────────────────────
        st.markdown("#### 📊 Feature Statistics")
        feat_cols = ["years_experience", "keyword_match_score",
                     "employment_gap_years", "cv_score"]
        feat_cols = [c for c in feat_cols if c in df.columns]
        stat_df = df[feat_cols].describe().round(4)
        st.dataframe(stat_df, use_container_width=True)

        # ── Score distribution ────────────────────────────────────────────
        st.markdown("#### 📈 CV Score Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram of cv_score
        axes[0].hist(df[df["hired"]==0]["cv_score"], bins=30, alpha=0.6,
                     label="Not Hired", color="#dc3545")
        axes[0].hist(df[df["hired"]==1]["cv_score"], bins=30, alpha=0.7,
                     label="Hired", color="#28a745")
        axes[0].set_xlabel("CV Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("CV Score Distribution by Hire Outcome")
        axes[0].legend()

        # Hire rate bar chart
        hire_by_gender = df.groupby("gender")["hired"].mean() * 100
        axes[1].bar(hire_by_gender.index, hire_by_gender.values,
                    color=["#2d6a9f", "#e05c8a"], edgecolor="white")
        axes[1].set_ylabel("Hire Rate (%)")
        axes[1].set_title("Hire Rate by Gender")
        for i, (g, v) in enumerate(hire_by_gender.items()):
            axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Download button ────────────────────────────────────────────────
        st.markdown("#### ⬇️ Download Dataset")
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Full Dataset as CSV",
            data=csv_data,
            file_name=f"xai_audit_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Downloads the complete generated dataset including predictions"
        )

    # ── TAB 1: Bias Detection ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<p class="section-header">Fairness Metrics by Protected Characteristic</p>',
                    unsafe_allow_html=True)

        # Show which libraries contributed
        fl_active  = compliance.get("fairlearn_active", False)
        ai_active  = compliance.get("aif360_active", False)
        lib_status = []
        if fl_active:  lib_status.append("✅ fairlearn")
        else:          lib_status.append("❌ fairlearn (not installed)")
        if ai_active:  lib_status.append("✅ aif360")
        else:          lib_status.append("❌ aif360 (not installed)")
        st.caption("Libraries active for this audit: " + "  |  ".join(lib_status))

        for feat in ["gender", "ethnicity", "age_band", "has_disability"]:
            if feat not in df.columns:
                continue
            with st.expander(f"📌 {feat.replace('_',' ').title()}", expanded=(feat == "gender")):
                feat_data = bias_results.get(feat, {})
                dp        = feat_data.get("demographic_parity", {})

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    dpd = dp.get("demographic_parity_difference")
                    rag = dp.get("rag", "")
                    if dpd is not None:
                        color = "#28a745" if "🟢" in rag else ("#ffc107" if "🟡" in rag else "#dc3545")
                        st.markdown(f"""
                        <div style="text-align:center;padding:1rem;border:1px solid rgba(128,128,128,0.2);border-radius:8px;">
                            <div style="font-size:2.5rem;font-weight:800;color:{color};">{dpd:.3f}</div>
                            <div style="font-size:0.9rem;opacity:0.7;">Demographic Parity Difference</div>
                            <div style="margin-top:0.5rem;">{rag}</div>
                        </div>""", unsafe_allow_html=True)

                    # Fairlearn extended metric if available
                    fl = feat_data.get("fairlearn_metrics", {})
                    if fl and "note" not in fl and "error" not in fl:
                        st.markdown("**Fairlearn:**")
                        st.markdown(f"DPD: `{fl.get('demographic_parity_difference','?')}` {fl.get('rag_dpd','')}")
                        st.markdown(f"EOD: `{fl.get('equalized_odds_difference','?')}` {fl.get('rag_eod','')}")

                    # AIF360 metric if available
                    ai = feat_data.get("aif360_metrics", {})
                    if ai and "note" not in ai and "error" not in ai:
                        st.markdown("**AIF360:**")
                        st.markdown(f"Disparate Impact: `{ai.get('disparate_impact','?')}`")
                        st.markdown(f"Stat. Parity Diff: `{ai.get('statistical_parity_diff','?')}`")

                with col_b:
                    hire_rates = df.groupby(feat)[LABEL_COL].mean() * 100
                    fig, ax    = plt.subplots(figsize=(6, 3))
                    colors     = ["#dc3545" if str(g) == str(dp.get("min_hire_rate_group")) else "#2d6a9f"
                                  for g in hire_rates.index]
                    bars = ax.barh([str(g) for g in hire_rates.index], hire_rates.values,
                                   color=colors, edgecolor="white", height=0.6)
                    ax.set_xlabel("Hire Rate (%)")
                    ax.set_title(f"Hire Rate by {feat.replace('_',' ').title()}", fontweight="bold")
                    ax.axvline(hire_rates.mean(), color="#666", linestyle="--", alpha=0.7, label="Average")
                    ax.legend(fontsize=8)
                    for bar, val in zip(bars, hire_rates.values):
                        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                                f"{val:.1f}%", va="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # DIR table
                dir_data = feat_data.get("disparate_impact_ratio", {})
                if dir_data:
                    dir_df = pd.DataFrame([
                        {"Group": grp, "Hire Rate": f"{info['hire_rate']}%",
                         "Privileged Rate": f"{info['privileged_rate']}%",
                         "DIR": info["dir"], "Status": info["rag"]}
                        for grp, info in dir_data.items()
                    ])
                    st.dataframe(dir_df, use_container_width=True, hide_index=True)

    # ── TAB 2: Data Audit ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<p class="section-header">Data Quality & Representation Audit</p>',
                    unsafe_allow_html=True)
        st.subheader("Demographic Representation")
        rep      = data_results.get("representation", {})
        rep_cols = st.columns(min(len(rep), 4))
        for i, (char, groups) in enumerate(rep.items()):
            with rep_cols[i % 4]:
                labels = list(groups.keys())
                sizes  = [g["pct"] for g in groups.values()]
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                colors  = plt.cm.Blues(np.linspace(0.4, 0.85, len(labels)))
                wedges, _, _ = ax.pie(sizes, labels=None, autopct="%1.0f%%",
                                      colors=colors, startangle=90,
                                      textprops={"fontsize": 8})
                ax.set_title(char.replace("_", " ").title(), fontsize=10, fontweight="bold")
                ax.legend(wedges, [l[:15] for l in labels], loc="lower center",
                          fontsize=7, bbox_to_anchor=(0.5, -0.15), ncol=2)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.subheader("⚠️ Proxy Feature Risks")
        pf = data_results.get("proxy_features", {}).get("known_proxies", {})
        if pf:
            pf_df = pd.DataFrame([
                {"Feature": k, "Risk": v["risk"], "Reason": v["reason"]}
                for k, v in pf.items()
            ])
            st.dataframe(pf_df, use_container_width=True, hide_index=True)
        else:
            st.success("No known proxy features detected.")

        flags = data_results.get("summary_flags", [])
        if flags:
            st.subheader("🚩 Audit Flags")
            for f in flags:
                st.warning(f)
        else:
            st.success("✅ No critical data issues detected.")

    # ── TAB 3: Explainability ─────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<p class="section-header">Model Explainability</p>',
                    unsafe_allow_html=True)

        method_badge = "✅ SHAP" if SHAP_AVAILABLE else "ℹ️ Permutation Importance (install shap for upgrade)"
        st.caption(f"Explanation method: {method_badge}")

        st.markdown("### How does the model make decisions?")
        st.markdown(
            f'<div class="explanation-box">{global_exp.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Feature Importance")
        imp_df = engine.compute_global_importance(X_test, y_test)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(imp_df["label"], imp_df["importance_mean"].clip(lower=0),
                       color="#2d6a9f", edgecolor="white", height=0.6)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance ({imp_df['method'].iloc[0]})", fontweight="bold")
        for bar, val in zip(bars, imp_df["importance_mean"].clip(lower=0)):
            if val > 0.001:
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # SHAP summary plot (only if SHAP installed)
        if SHAP_AVAILABLE:
            st.markdown("### SHAP Summary Plot")
            st.caption("SHAP values show the impact of each feature on individual predictions.")
            try:
                shap_vals = engine.get_shap_values(X_test)
                if shap_vals is not None:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.summary_plot(shap_vals, X_test, feature_names=FEATURE_COLS,
                                      show=False, plot_type="bar")
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.caption(f"SHAP plot unavailable: {e}")
        else:
            st.info("💡 Install `shap` to enable SHAP summary plots: `pip install shap`")

        st.markdown("### CV Score Distribution by Gender")
        fig, ax = plt.subplots(figsize=(8, 4))
        for gender, group in df.groupby("gender"):
            ax.hist(group["cv_score"], bins=30, alpha=0.6, label=gender,
                    color=("#2d6a9f" if gender == "Male" else "#e05c8a"))
        ax.set_xlabel("CV Score")
        ax.set_ylabel("Count")
        ax.set_title("CV Score Distributions by Gender")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 4: Candidate Lookup ───────────────────────────────────────────
    with tabs[4]:
        st.markdown('<p class="section-header">Individual Candidate Explanation</p>',
                    unsafe_allow_html=True)
        selected_id = st.selectbox("Select Candidate ID", df["candidate_id"].tolist())

        if selected_id:
            row  = df[df["candidate_id"] == selected_id].iloc[0]
            pred = int(row["predicted_hired"])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Candidate Profile**")
                profile_data = {
                    "Gender":        row.get("gender", "—"),
                    "Age":           row.get("age", "—"),
                    "Ethnicity":     row.get("ethnicity", "—"),
                    "Region":        row.get("region", "—"),
                    "Education":     row.get("education_level", "—"),
                    "Experience":    f"{row.get('years_experience', 0)} yrs",
                    "Employment Gap": f"{row.get('employment_gap_years', 0):.1f} yrs",
                    "CV Score":      f"{row.get('cv_score', 0):.1f}/100",
                    "Decision":      "✅ Shortlisted" if pred == 1 else "❌ Not Shortlisted",
                }
                st.dataframe(
                    pd.DataFrame(list(profile_data.items()), columns=["Field", "Value"]),
                    use_container_width=True, hide_index=True
                )

            with col2:
                feat_row = row[FEATURE_COLS + ["candidate_id"]].copy()
                exp      = engine.explain_decision(feat_row, pred)
                method   = exp.get("method", "")
                st.markdown(f"**Decision Explanation** *(method: {method})*")
                st.markdown(
                    f'<div class="explanation-box">{exp["plain_english"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )

                # Waterfall / bar chart of factors
                factors = [f for f in exp["factors"] if f["contribution"] != 0][:6]
                if factors:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    labels  = [f["label"] for f in factors]
                    values  = [f["contribution"] for f in factors]
                    colors  = ["#28a745" if v > 0 else "#dc3545" for v in values]
                    ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_title("Decision Factor Contributions", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            st.info(
                "ℹ️ Under UK GDPR Article 22, this candidate has the right to request "
                "a human review of this automated decision."
            )

    # ── TAB 5: Report ─────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<p class="section-header">Audit Report</p>', unsafe_allow_html=True)

        if st.button("📄 Generate Report"):
            gen = AuditReportGenerator(company_name=cn, job_role=jr)
            os.makedirs("reports", exist_ok=True)
            path   = f"reports/audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report = gen.generate(
                model_info=model_info,
                data_audit_results=data_results,
                bias_results=bias_results,
                explainability_summary=global_exp,
                output_path=path,
            )
            st.success(f"Report generated: `{path}`")
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=report,
                file_name=os.path.basename(path),
                mime="text/markdown",
            )
            with st.expander("Preview Report"):
                st.markdown(report)

else:
    # Landing page
    st.markdown("""
### Welcome to the XAI Audit Toolkit

This toolkit helps UK SMEs comply with AI fairness and transparency regulations
when using automated tools in recruitment.

**To get started:**
1. Configure your company details in the sidebar
2. Choose a dataset (or upload your own)
3. Click **🚀 Run Audit**

---
**What the toolkit checks:**
- 📊 **Bias Detection** — Disparate impact across gender, ethnicity, age, disability
- 🔍 **Data Audit** — Proxy features, representation gaps, label bias
- 💡 **Explainability** — Plain-English explanations for each hiring decision
- 📄 **Report** — Downloadable compliance report aligned with the Equality Act 2010
    """)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Protected Characteristics", "4")
    col2.metric("Fairness Metrics", "3+")
    col3.metric("Regulatory References", "6")
    col4.metric("Data Source", "ONS 2023")
