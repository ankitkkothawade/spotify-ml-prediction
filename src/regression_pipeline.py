#!/usr/bin/env python3
"""
Pro regression pipeline (Spotify-style, interview-ready)

- Column normalization + target alias ('popularity' -> 'pop')
- If TEST lacks target, falls back to 80/20 split on TRAIN
- ColumnTransformer with StandardScaler + OneHotEncoder (backward compatible)
- Model zoo + RandomizedSearchCV:
    * LinearRegression (baseline)
    * SVR (rbf)
    * RandomForestRegressor
    * GradientBoostingRegressor
    * XGBRegressor (optional)
    * LGBMRegressor (optional)
- Outputs: cv_leaderboard, best_model_name/params, holdout metrics, plots
- JSON-safe serialization for numpy types
"""

import argparse, json, os, random, sys, warnings
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn compat + version checks
from packaging.version import Version
from sklearn import __version__ as sk_version
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

SEED_DEFAULT = 42
np.random.seed(SEED_DEFAULT)
random.seed(SEED_DEFAULT)

SK_VER = Version(sk_version)
HAS_SPARSE_OUTPUT = SK_VER >= Version("1.2.0")  # OneHotEncoder.sparse_output added in 1.2

# ---------- optional imports ----------
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

# ---------- json helpers ----------
def _to_py(o):
    """Convert numpy / pandas scalars to pure Python for json.dump."""
    import numpy as _np
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    return o

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--target", default="pop")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    p.add_argument("--n_iter", type=int, default=40, help="RandomizedSearch iterations per model (default 40)")
    p.add_argument("--random_state", type=int, default=SEED_DEFAULT)
    return p.parse_args()

# -------------------- utils --------------------
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def resolve_target(train_df: pd.DataFrame, requested: str) -> str:
    cols = set(train_df.columns)
    t = requested.strip().lower().replace(" ", "_")
    aliases = {"popularity": "pop"}
    t = aliases.get(t, t)
    if t in cols: return t
    for cand in ["pop", "popularity"]:
        if cand in cols: return cand
    raise AssertionError(f"Target '{requested}' not found. Available: {sorted(train_df.columns)}")

def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    assert target in df.columns, f"Target '{target}' not found in columns."
    return df.drop(columns=[target]), df[target]

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_proc = Pipeline([("scaler", StandardScaler())])
    if cat_cols:
        if HAS_SPARSE_OUTPUT:
            cat_proc = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        else:
            cat_proc = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    else:
        cat_proc = "drop"
    pre = ColumnTransformer([("num", num_proc, num_cols), ("cat", cat_proc, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def holdout_metrics(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    return {"R2": float(r2_score(y_test, y_pred)),
            "RMSE": rmse(y_test, y_pred),
            "MAE": float(mean_absolute_error(y_test, y_pred))}

# -------------------- main --------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Calm down noisy warnings (harmless numeric overflow / ill-conditioning etc.)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Read + normalize
    train = norm_cols(pd.read_csv(args.train))
    test  = norm_cols(pd.read_csv(args.test))

    # Resolve target on TRAIN
    target = resolve_target(train, args.target)

    # Train/holdout split
    if target not in test.columns:
        print(f"⚠️ Target '{target}' not in TEST — using 80/20 split on TRAIN for evaluation.")
        X_all, y_all = split_xy(train, target)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_all, y_all, test_size=0.2, random_state=args.random_state
        )
    else:
        X_train, y_train = split_xy(train, target)
        X_holdout, y_holdout = split_xy(test, target)

    # Preprocessor + scorer
    pre, num_cols, cat_cols = build_preprocessor(X_train)
    scorer = make_scorer(r2_score)

    # Model spaces
    rng = np.random.RandomState(args.random_state)
    model_spaces = []

    model_spaces.append(("LinearRegression", LinearRegression(), {}))
    model_spaces.append(("SVR_rbf", SVR(), {
        "model__kernel": ["rbf"],
        "model__C": np.logspace(-1, 2, 20),
        "model__gamma": ["scale", "auto"],
    }))
    model_spaces.append(("RandomForest",
        RandomForestRegressor(random_state=args.random_state, n_jobs=-1),
        {
            "model__n_estimators": rng.randint(200, 601, size=20),
            "model__max_depth": [None] + list(range(4, 21)),
            "model__max_features": ["auto", "sqrt", 0.5, 0.7, 0.9],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    ))
    model_spaces.append(("GradientBoosting",
        GradientBoostingRegressor(random_state=args.random_state),
        {
            "model__n_estimators": rng.randint(200, 601, size=20),
            "model__learning_rate": np.linspace(0.01, 0.2, 20),
            "model__max_depth": rng.randint(2, 7, size=5),
            "model__subsample": np.linspace(0.6, 1.0, 5),
        }
    ))

    xgb = try_xgb()
    if xgb is not None:
        model_spaces.append(("XGBoost", xgb.XGBRegressor(
            random_state=args.random_state, n_jobs=-1, tree_method="hist"
        ), {
            "model__n_estimators": rng.randint(300, 801, size=20),
            "model__max_depth": rng.randint(3, 9, size=6),
            "model__learning_rate": np.linspace(0.01, 0.2, 20),
            "model__subsample": np.linspace(0.6, 1.0, 5),
            "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
            "model__reg_lambda": np.linspace(0.0, 1.0, 6),
        }))

    lgb = try_lgbm()
    if lgb is not None:
        model_spaces.append(("LightGBM", lgb.LGBMRegressor(
            random_state=args.random_state, n_jobs=-1
        ), {
            "model__n_estimators": rng.randint(300, 801, size=20),
            "model__num_leaves": rng.randint(31, 256, size=8),
            "model__learning_rate": np.linspace(0.01, 0.2, 20),
            "model__subsample": np.linspace(0.6, 1.0, 5),
            "model__colsample_bytree": np.linspace(0.6, 1.0, 5),
        }))

    # CV + search
    kf = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    cv_results = []
    best_model = None
    best_name = None
    best_params = {}
    best_cv_score = -np.inf

    for name, est, space in model_spaces:
        pipe = Pipeline([("pre", pre), ("model", est)])
        if space:
            rs = RandomizedSearchCV(
                pipe, param_distributions=space, n_iter=args.n_iter,
                scoring=scorer, cv=kf, n_jobs=-1, random_state=args.random_state, verbose=0
            )
            rs.fit(X_train, y_train)
            score = float(rs.best_score_)
            fitted = rs.best_estimator_
            params = rs.best_params_
        else:
            # LR baseline with oof
            oof = cross_val_predict(pipe, X_train, y_train, cv=kf, n_jobs=-1)
            score = float(r2_score(y_train, oof))
            fitted = pipe.fit(X_train, y_train)
            params = {}

        cv_results.append({"model": name, "CV_R2": score})
        if score > best_cv_score:
            best_cv_score = score
            best_model = fitted
            best_name = name
            best_params = params

    # Save CV leaderboard
    cv_tbl = pd.DataFrame(cv_results).sort_values("CV_R2", ascending=False)
    cv_csv = os.path.join(args.results_dir, "cv_leaderboard.csv")
    cv_tbl.to_csv(cv_csv, index=False)

    # Holdout metrics
    hold = holdout_metrics(best_model, X_holdout, y_holdout)
    metrics = {
        "best_model": best_name,
        "CV_R2_best": float(best_cv_score),
        "Holdout_R2": hold["R2"],
        "Holdout_RMSE": hold["RMSE"],
        "Holdout_MAE": hold["MAE"],
    }

    # JSON-safe dumps
    with open(os.path.join(args.results_dir, "best_model_name.json"), "w") as f:
        json.dump({"best_model": best_name}, f, indent=2)
    with open(os.path.join(args.results_dir, "best_params.json"), "w") as f:
        json.dump(_to_py(best_params), f, indent=2)
    with open(os.path.join(args.results_dir, "regression_metrics.json"), "w") as f:
        json.dump(_to_py(metrics), f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.results_dir, "regression_metrics.csv"), index=False)

    # Plots
    plt.figure(figsize=(7,4))
    plt.bar(cv_tbl["model"], cv_tbl["CV_R2"])
    plt.title("Cross-Validated R² by Model")
    plt.ylabel("CV R²")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "regression_r2_comparison.png"), dpi=150)

    y_pred_hold = best_model.predict(X_holdout)
    plt.figure(figsize=(5,5))
    plt.scatter(y_holdout, y_pred_hold, alpha=0.6)
    plt.xlabel("True (holdout)")
    plt.ylabel(f"Predicted ({best_name})")
    plt.title("Predicted vs True — Holdout")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "regression_pred_vs_true.png"), dpi=150)

    print("\n✅ Done.")
    print("Best model:", best_name)
    print(cv_tbl.to_string(index=False))
    print("\nHoldout metrics:", metrics)

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n❌ {e}")
        print("\nHint: check headers with:")
        print("  head -n 1 data/RegressionTrain.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
