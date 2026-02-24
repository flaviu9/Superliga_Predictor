"""
Train Logistic Regression, Random Forest and XGBoost to predict
whether a team will finish in the top 6 at the end of the season.

Evaluation: Leave-One-Season-Out cross-validation (5 folds).
Each fold trains on 4 seasons and validates on the held-out season —
the most realistic split since seasons are the natural independent unit.

Output
------
- Per-fold and mean metrics (AUC, F1, Precision, Recall) for each model
- Feature importance ranking (RF + XGB)
- Top-6 probabilities for the current season at round 27
- Saved models under models/
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_PATH  = "data/processed/all_seasons.csv"
TEST_PATH   = "data/processed/test_2025_2026.csv"
MODELS_DIR  = "models"
LAST_ROUND  = 28   # last completed round in the test season

# ── Feature columns ────────────────────────────────────────────────────────
# 'round' is included so the model knows how far into the season we are.
# 'season', 'team', 'final_position', 'is_top6' are never features.
FEATURES = [
    "round",
    "gp", "wins", "draws", "losses",
    "points", "gf", "ga", "gd", "ppg",
    "home_wins", "home_draws", "home_losses", "home_gf", "home_ga", "home_gd",
    "away_wins", "away_draws", "away_losses", "away_gf", "away_ga", "away_gd",
    "form_pts_last5", "form_gd_last5", "form_wins_last5",
    "position", "points_from_6th",
]
TARGET = "is_top6"


# ── Model definitions ──────────────────────────────────────────────────────
def make_models():
    """Return a dict of {name: estimator} ready for fit/predict."""

    # Class ratio: 50 non-top6 / 30 top6 ≈ 1.67  →  mild imbalance
    neg_pos_ratio = 50 / 30

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}


# ── Cross-validation ───────────────────────────────────────────────────────
def evaluate_logo(model, X, y, groups, model_name):
    """
    Leave-One-Group-Out CV where groups = season labels.
    Returns per-fold metrics and prints a summary.
    """
    logo   = LeaveOneGroupOut()
    season_names = groups.unique()

    fold_results = []
    print(f"\n{'─'*56}")
    print(f"  {model_name}")
    print(f"{'─'*56}")

    for train_idx, val_idx in logo.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        val_season  = groups.iloc[val_idx].iloc[0]

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        auc  = roc_auc_score(y_val, proba)
        f1   = f1_score(y_val, pred, zero_division=0)
        prec = precision_score(y_val, pred, zero_division=0)
        rec  = recall_score(y_val, pred, zero_division=0)

        fold_results.append({"season": val_season, "auc": auc, "f1": f1,
                              "precision": prec, "recall": rec})
        print(f"  {val_season}  AUC={auc:.3f}  F1={f1:.3f}  "
              f"P={prec:.3f}  R={rec:.3f}")

    res = pd.DataFrame(fold_results)
    print(f"  {'MEAN':11}  AUC={res['auc'].mean():.3f}±{res['auc'].std():.3f}  "
          f"F1={res['f1'].mean():.3f}±{res['f1'].std():.3f}")
    return res


# ── Feature importance ─────────────────────────────────────────────────────
def print_feature_importance(model, model_name, top_n=10):
    """Print top-N feature importances for tree-based models."""
    if model_name == "Random Forest":
        imp = model.feature_importances_
    elif model_name == "XGBoost":
        imp = model.feature_importances_
    else:
        # Logistic Regression: use absolute coefficient magnitude
        imp = np.abs(model.named_steps["clf"].coef_[0])

    order = np.argsort(imp)[::-1][:top_n]
    print(f"\n  Top {top_n} features ({model_name}):")
    for rank, idx in enumerate(order, 1):
        print(f"    {rank:2}. {FEATURES[idx]:<22} {imp[idx]:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load training data ─────────────────────────────────────────────────
    train = pd.read_csv(TRAIN_PATH)
    X     = train[FEATURES]
    y     = train[TARGET]
    groups = train["season"]   # grouping variable for LOGO-CV

    print("=" * 56)
    print("  TRAINING DATA")
    print("=" * 56)
    print(f"  Rows: {len(train):,}  |  Features: {len(FEATURES)}")
    print(f"  Seasons: {sorted(groups.unique())}")
    print(f"  Label: top6={y.sum()} ({y.mean()*100:.1f}%)  "
          f"non-top6={(~y.astype(bool)).sum()}")

    models = make_models()

    # ── Leave-One-Season-Out evaluation ────────────────────────────────────
    print("\n" + "=" * 56)
    print("  LEAVE-ONE-SEASON-OUT CROSS-VALIDATION")
    print("=" * 56)

    all_cv_results = {}
    for name, model in models.items():
        cv_res = evaluate_logo(model, X, y, groups, name)
        all_cv_results[name] = cv_res

    # ── Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  SUMMARY")
    print("=" * 56)
    print(f"  {'Model':<22} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for name, res in all_cv_results.items():
        print(f"  {name:<22} {res['auc'].mean():>8.3f} {res['f1'].mean():>8.3f} "
              f"{res['precision'].mean():>10.3f} {res['recall'].mean():>8.3f}")

    # ── Train final models on ALL training data ────────────────────────────
    print("\n" + "=" * 56)
    print("  FINAL MODELS (trained on all 5 seasons)")
    print("=" * 56)

    final_models = make_models()
    for name, model in final_models.items():
        model.fit(X, y)
        path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, path)
        print(f"  Saved: {path}")
        print_feature_importance(model, name)

    # ── Predictions on current season at round 27 ──────────────────────────
    print("\n" + "=" * 56)
    print(f"  PREDICTIONS — 2025/26 season at round {LAST_ROUND}")
    print("=" * 56)

    test = pd.read_csv(TEST_PATH)
    snap = test[test["round"] == LAST_ROUND].copy()

    X_test = snap[FEATURES]

    for name, model in final_models.items():
        snap[f"p_top6_{name.split()[0].lower()}"] = model.predict_proba(X_test)[:, 1]

    prob_cols = [c for c in snap.columns if c.startswith("p_top6_")]
    snap["p_top6_mean"] = snap[prob_cols].mean(axis=1)

    display = (
        snap[["team", "position", "points", "gd", "points_from_6th"]
             + prob_cols + ["p_top6_mean"]]
        .sort_values("p_top6_mean", ascending=False)
        .reset_index(drop=True)
    )
    display.index += 1

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(display.to_string())

    # Save predictions
    pred_path = os.path.join(MODELS_DIR, "predictions_round28.csv")
    display.to_csv(pred_path, index=True)
    print(f"\n  Predictions saved → {pred_path}")

    # ── Detailed report on the last CV fold (most recent season as proxy) ──
    print("\n" + "=" * 56)
    print("  CLASSIFICATION REPORT — best model on 2024_2025 fold")
    print("=" * 56)
    best_name = max(all_cv_results, key=lambda n: all_cv_results[n]["auc"].mean())
    print(f"  Best model by AUC: {best_name}")

    # Re-run the 2024_2025 fold for the best model
    best_model = make_models()[best_name]
    mask_test  = groups == "2024_2025"
    best_model.fit(X[~mask_test], y[~mask_test])
    proba_last = best_model.predict_proba(X[mask_test])[:, 1]
    pred_last  = (proba_last >= 0.5).astype(int)
    print(classification_report(y[mask_test], pred_last,
                                target_names=["not top6", "top6"]))


if __name__ == "__main__":
    main()
