"""
Optional Dependency Checker

Central module that detects which optional libraries are installed.
All other modules import from here so the check happens in one place.

Optional libraries and what they unlock:
  - shap      → More accurate per-candidate SHAP explanations
  - fairlearn → Additional fairness metrics (demographic parity, equalised odds)
  - aif360    → IBM AI Fairness 360 extended bias metrics
  - sdv       → Synthetic Data Vault — more realistic synthetic data generation
  - torch     → PyTorch neural network model option

If any of these are missing, the toolkit falls back to built-in implementations
and notifies the user how to install the missing library.
"""


def check_optional_dependencies() -> dict:
    """
    Checks which optional libraries are available.
    Returns a dict of {library_name: bool}
    """
    deps = {}

    try:
        import shap  # noqa: F401
        deps["shap"] = True
    except ImportError:
        deps["shap"] = False

    try:
        import fairlearn  # noqa: F401
        deps["fairlearn"] = True
    except ImportError:
        deps["fairlearn"] = False

    try:
        import aif360  # noqa: F401
        deps["aif360"] = True
    except ImportError:
        deps["aif360"] = False

    try:
        import sdv  # noqa: F401
        deps["sdv"] = True
    except ImportError:
        deps["sdv"] = False

    try:
        import torch  # noqa: F401
        deps["torch"] = True
    except ImportError:
        deps["torch"] = False

    return deps


def print_dependency_status():
    """Prints a formatted status table of optional dependencies to the console."""
    deps = check_optional_dependencies()

    print("Optional Dependencies")
    descriptions = {
        "shap":      "SHAP explanations (more accurate per-candidate explainability)",
        "fairlearn": "Fairlearn metrics (demographic parity, equalised odds)",
        "aif360":    "IBM AI Fairness 360 (extended bias metrics)",
        "sdv":       "Synthetic Data Vault (more realistic data generation)",
        "torch":     "PyTorch neural network model",
    }
    for lib, available in deps.items():
        status = "✅ Installed" if available else "❌ Not installed — fallback active"
        desc   = descriptions.get(lib, "")
        print(f"  {lib:<12} {status}")
        print(f"               {desc}")
    print()

    missing = [lib for lib, available in deps.items() if not available]
    if missing:
        print("  To install missing libraries:")
        print(f"  pip install {' '.join(missing)}")
    else:
        print("  All optional libraries installed.")
    print("")


def get_dependency_status_list() -> list:
    """
    Returns a list of dicts with library status info.
    Used by the Streamlit dashboard to render status badges.
    """
    deps = check_optional_dependencies()
    hints = {
        "shap":      "pip install shap",
        "fairlearn": "pip install fairlearn",
        "aif360":    "pip install aif360",
        "sdv":       "pip install sdv",
        "torch":     "pip install torch",
    }
    descriptions = {
        "shap":      "SHAP explanations",
        "fairlearn": "Fairlearn metrics",
        "aif360":    "IBM AI Fairness 360",
        "sdv":       "Synthetic Data Vault",
        "torch":     "PyTorch neural network",
    }
    result = []
    for lib, available in deps.items():
        result.append({
            "library":     lib,
            "available":   available,
            "description": descriptions.get(lib, ""),
            "install":     hints.get(lib, f"pip install {lib}"),
        })
    return result


# ── Convenience booleans ──────────────────────────────────────────────────────
# Import these directly in other modules instead of repeating try/except blocks
_deps = check_optional_dependencies()
SHAP_AVAILABLE      = _deps["shap"]
FAIRLEARN_AVAILABLE = _deps["fairlearn"]
AIF360_AVAILABLE    = _deps["aif360"]
SDV_AVAILABLE       = _deps["sdv"]
TORCH_AVAILABLE     = _deps["torch"]
