#!/usr/bin/env python3
"""
Pro classification pipeline (Spotify-style) — NaN-safe + rare-class safe + interview-ready

Highlights
- Normalizes column names
- Target: default 'label' with aliases {'label','genre'} -> 'top_genre'
- If TEST lacks target, uses Stratified 80/20 split on TRAIN (falls back to random split if needed)
- Leakage-safe preprocessing:
    * Numeric: SimpleImputer(median) + StandardScaler
    * Categorical: SimpleImputer(most_frequent) + OneHotEncoder (works on old/new scikit-learn)
- Handles class imbalance / tiny classes:
    * Collapses rare classes to 'other' before splitting (configurable)
- Models + RandomizedSearchCV:
    * RandomForestClassifier
    * SVM (rbf)
    * XGBoost (optional, if installed)
    * LightGBM (optional, if installed)
- Outputs (in --results_dir):
    * classification_cv_leaderboard.csv     (CV Accuracy leaderboard)
    * classification_best_model_name.json
    * classification_best_params.json
    * classification_metrics.json/.csv      (Holdout Accuracy, F1-weighted)
    * classification_accuracy_comparison.png (CV bar chart)
    * classification_confusion.png          (best model, labeled)
    * classification_report.txt
"""

import argparse, json, os, random, sys, warnings
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from packaging.version import Version
from sklearn import __version__ as sk_version
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------- Config ----------------
SEED_DEFAULT = 42
np.random.seed(SEED_DEFAULT)
random.seed(SEED_DEFAULT)

SK_VER = Version(sk_version)
HAS_SPARSE_OUTPUT = SK_VER >= Version("1.2.0")  # OneHotEncoder.sparse_output added in 1.2

# Rare-class handling (you can tweak these)
MIN_CLASS_COUNT = 3       # any class with < 3 samples will be collapsed to OTHER_LABEL
TOP_K_CLASSES   = None    # or set an int to keep only the top-K frequent classes, rest -> OTHER_LABEL
OTHER_LABEL     = "other"

# ------------- Optional model imports -------------
def try_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def try_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

# ---------------- JSON helpers ----------------
def _to_py(o):
    """Convert numpy/pandas scalars/containers to pure Python for json.dump."""
    import numpy as _np
    if isinstance(o, (_np.generic,)):
        return o.item()
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    return o

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="label")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n_iter", type=int, default=40)
    p.add_argument("--random_state", type=int, default=SEED_DEFAULT)
    return p.parse_args()

# ---------------- Utils ----------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def ensure_target(df: pd.DataFrame, target: str) -> str:
    """Map friendly aliases to real target; ensure present in df."""
    t = target.strip().lower().replace(" ", "_")
    aliases = {"label": "top_genre", "genre": "top_genre"}
    t = aliases.get(t, t)
    if t in df.columns:
        return t
    raise AssertionError(f"Target '{target}' not found. Available: {sorted(df.columns)}")

def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    assert target in df.columns, f"Target '{target}' not in columns."
    df = df.dropna(subset=[target])  # drop rows with NaN target
    return df.drop(columns=[target]), df[target]

def collapse_rare_classes(y: pd.Series,
                          min_count: int = MIN_CLASS_COUNT,
                          top_k: int = TOP_K_CLASSES,
                          other_label: str = OTHER_LABEL) -> pd.Series:
    counts = y.value_counts(dropna=False)
    keep = set(counts[counts >= min_count].index)
    if top_k is not None and top_k > 0:
        keep |= set(counts.sort_values(ascending=False).head(top_k).index)
    y2 = y.where(y.isin(keep), other_label)
    # Ensure at least 2 classes remain; if not, relax to >=2 rule
    if y2.nunique() < 2:
        keep2 = set(counts[counts >= 2].index)
        y2 = y.where(y.isin(keep2), other_label)
    return y2

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_proc = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    if cat_cols:
        if HAS_SPARSE_OUTPUT:
            cat_proc = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
        else:
            cat_proc = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))
            ])
    else:
        cat_proc = "drop"

    pre = ColumnTransformer(
        [("num", num_proc, num_cols),
         ("cat", cat_proc, cat_cols)],
        remainder="drop"
    )
    return pre, num_cols, cat_cols

# ---------------- Main ----------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Tame noisy (harmless) warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Read + normalize
    train = norm_cols(pd.read_csv(args.train))
    test  = norm_cols(pd.read_csv(args.test))

    # Target on TRAIN (authoritative)
    target = ensure_target(train, args.target)

    # Split & collapse rare classes BEFORE any splitting
    X_train, y_train_raw = split_xy(train, target)
    y_train_raw = collapse_rare_classes(y_train_raw, MIN_CLASS_COUNT, TOP_K_CLASSES, OTHER_LABEL)

    # If TEST has labels, use them; else stratified split (fallback to random split if needed)
    if target in test.columns:
        X_hold, y_hold_raw = split_xy(test, target)
        y_hold_raw = collapse_rare_classes(y_hold_raw, MIN_CLASS_COUNT, TOP_K_CLASSES, OTHER_LABEL)
        test_supplied = True
    else:
        print(f"⚠️ Target '{target}' not in TEST — using Stratified 80/20 split on TRAIN.")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
        try:
            idx_train, idx_hold = next(sss.split(X_train, y_train_raw))
            X_train, X_hold = X_train.iloc[idx_train], X_train.iloc[idx_hold]
            y_train_raw, y_hold_raw = y_train_raw.iloc[idx_train], y_train_raw.iloc[idx_hold]
        except ValueError:
            print("⚠️ Not enough members per class for stratified split — "
                  "falling back to random split without stratification.")
            X_train, X_hold, y_train_raw, y_hold_raw = train_test_split(
                X_train, y_train_raw, test_size=0.2, random_state=args.random_state
            )
        test_supplied = False

    # Label encode
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_hold  = le.transform(y_hold_raw)

    # Preprocessor + scorer
    pre, num_cols, cat_cols = build_preprocessor(X_train)
    scorer = make_scorer(accuracy_score)

    # Model search spaces
    rng = np.random.RandomState(args.random_state)
    model_spaces = []

    model_spaces.append((
        "RandomForest",
        RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
        {
            "model__n_estimators": rng.randint(300, 901, size=20),
            "model__max_depth": [None] + list(range(4, 31)),
            "model__max_features": ["auto", "sqrt", 0.5, 0.7, 0.9],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    ))

    model_spaces.append((
        "SVM_rbf",
        SVC(probability=True),
        {
            "model__kernel": ["rbf"],
            "model__C": np.logspace(-1, 2, 20),
            "model__gamma": ["scale", "auto"],
        }
    ))

    xgb = try_xgb()
    if xgb is not None:
        model_spaces.append((
            "XGBoost",
            xgb.XGBClassifier(
                random_state=args.random_state, n_jobs=-1, tree_method="hist",
                eval_metric="mlogloss"
            ),
            {
                "model__n_estimators": rng.randint(300, 901, size=20),
                "model__max_depth": rng.randint(3, 11, size=6),
                "model__learning_rate": np.linspace(0.01, 0.2, 20),
                "model__subsample": np.linspace(0.6, 1.0, 5),
                "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
                "model__reg_lambda": np.linspace(0.0, 1.0, 6),
            }
        ))

    lgb = try_lgbm()
    if lgb is not None:
        model_spaces.append((
            "LightGBM",
            lgb.LGBMClassifier(random_state=args.random_state, n_jobs=-1),
            {
                "model__n_estimators": rng.randint(300, 901, size=20),
                "model__num_leaves": rng.randint(31, 256, size=8),
                "model__learning_rate": np.linspace(0.01, 0.2, 20),
                "model__subsample": np.linspace(0.6, 1.0, 5),
                "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
            }
        ))

    # CV + randomized search
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    cv_results = []
    best_model = None
    best_name = None
    best_params = {}
    best_cv_acc = -np.inf

    for name, est, space in model_spaces:
        pipe = Pipeline([("pre", pre), ("model", est)])
        rs = RandomizedSearchCV(
            pipe, param_distributions=space, n_iter=args.n_iter,
            scoring=scorer, cv=skf, n_jobs=-1, random_state=args.random_state, verbose=0
        )
        rs.fit(X_train, y_train)
        acc = float(rs.best_score_)
        cv_results.append({"model": name, "CV_Accuracy": acc})
        if acc > best_cv_acc:
            best_cv_acc = acc
            best_model = rs.best_estimator_
            best_name = name
            best_params = rs.best_params_

    # Save CV leaderboard
    cv_tbl = pd.DataFrame(cv_results).sort_values("CV_Accuracy", ascending=False)
    cv_csv = os.path.join(args.results_dir, "classification_cv_leaderboard.csv")
    cv_tbl.to_csv(cv_csv, index=False)

    # Holdout evaluation
    y_pred = best_model.predict(X_hold)
    acc = float(accuracy_score(y_hold, y_pred))
    f1w = float(f1_score(y_hold, y_pred, average="weighted"))
    metrics = {
        "best_model": best_name,
        "CV_Accuracy_best": float(best_cv_acc),
        "Holdout_Accuracy": acc,
        "Holdout_F1_weighted": f1w,
        "classes": le.classes_.tolist(),
        "test_has_labels": test_supplied
    }

    # Save artifacts
    with open(os.path.join(args.results_dir, "classification_best_model_name.json"), "w") as f:
        json.dump({"best_model": best_name}, f, indent=2)
    with open(os.path.join(args.results_dir, "classification_best_params.json"), "w") as f:
        json.dump(_to_py(best_params), f, indent=2)
    with open(os.path.join(args.results_dir, "classification_metrics.json"), "w") as f:
        json.dump(_to_py(metrics), f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.results_dir, "classification_metrics.csv"), index=False)

    # Plots: CV bar
    plt.figure(figsize=(7,4))
    plt.bar(cv_tbl["model"], cv_tbl["CV_Accuracy"])
    plt.title("Cross-Validated Accuracy by Model")
    plt.ylabel("CV Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "classification_accuracy_comparison.png"), dpi=150)

    # Confusion matrix
    cm = confusion_matrix(y_hold, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "classification_confusion.png"), dpi=150)

    # Classification report
    report = classification_report(y_hold, y_pred, target_names=le.classes_)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print("\n✅ Done.")
    print("Best model:", best_name)
    print(cv_tbl.to_string(index=False))
    print("\nHoldout metrics:", metrics)

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n❌ {e}")
        print("\nHint: check headers with:\n  head -n 1 data/ClassificationTrain.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
