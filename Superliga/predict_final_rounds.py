"""
Show model predictions at each of the last 3 completed rounds (25, 26, 27)
of the 2025/26 season, and highlight how probabilities shifted round-by-round.

Rounds 28-30 have no results yet — predictions are based on current stats.
"""

import joblib
import numpy as np
import pandas as pd

TEST_PATH  = "data/processed/test_2025_2026.csv"
MODELS_DIR = "models"
ROUNDS     = [26, 27, 28]

FEATURES = [
    "round",
    "gp", "wins", "draws", "losses",
    "points", "gf", "ga", "gd", "ppg",
    "home_wins", "home_draws", "home_losses", "home_gf", "home_ga", "home_gd",
    "away_wins", "away_draws", "away_losses", "away_gf", "away_ga", "away_gd",
    "form_pts_last5", "form_gd_last5", "form_wins_last5",
    "position", "points_from_6th",
]

# ── Load models ──────────────────────────────────────────────────────────────
lr  = joblib.load(f"{MODELS_DIR}/logistic_regression.pkl")
rf  = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
xgb = joblib.load(f"{MODELS_DIR}/xgboost.pkl")

def ensemble_proba(X):
    p_lr  = lr.predict_proba(X)[:, 1]
    p_rf  = rf.predict_proba(X)[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1]
    return p_lr, p_rf, p_xgb, (p_lr + p_rf + p_xgb) / 3

# ── Load test data ───────────────────────────────────────────────────────────
test = pd.read_csv(TEST_PATH)

# ── Predict at each round ────────────────────────────────────────────────────
round_frames = {}
for rnd in ROUNDS:
    snap = test[test["round"] == rnd].copy().reset_index(drop=True)
    p_lr, p_rf, p_xgb, p_mean = ensemble_proba(snap[FEATURES])
    snap["p_lr"]   = p_lr
    snap["p_rf"]   = p_rf
    snap["p_xgb"]  = p_xgb
    snap["p_mean"] = p_mean
    round_frames[rnd] = snap

# ── Print per-round tables ───────────────────────────────────────────────────
sep = "=" * 80
for rnd in ROUNDS:
    df = round_frames[rnd].sort_values("p_mean", ascending=False).reset_index(drop=True)
    df.index += 1

    print(f"\n{sep}")
    print(f"  ROUND {rnd}  —  Predicted P(top 6)")
    print(sep)
    print(f"  {'#':>2}  {'Team':<30} {'Pos':>4} {'Pts':>4} {'GD':>4} "
          f"{'Gap':>5} {'LR':>6} {'RF':>6} {'XGB':>6} {'Mean':>6}")
    print(f"  {'-'*2}  {'-'*30} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for rank, row in df.iterrows():
        marker = " ◀" if rank == 6 else ("   " if rank < 6 else "")
        print(f"  {rank:>2}  {row['team']:<30} {int(row['position']):>4} "
              f"{int(row['points']):>4} {int(row['gd']):>4} "
              f"{int(row['points_from_6th']):>+5} "
              f"{row['p_lr']:>6.3f} {row['p_rf']:>6.3f} "
              f"{row['p_xgb']:>6.3f} {row['p_mean']:>6.3f}{marker}")

# ── Movement table — how probabilities shifted across rounds ─────────────────
print(f"\n{sep}")
print("  PROBABILITY MOVEMENT  (rounds 26 → 27 → 28)")
print(sep)

# Anchor on round-27 order
anchor = round_frames[28].set_index("team")["p_mean"].sort_values(ascending=False)
teams  = anchor.index.tolist()

print(f"  {'Team':<30} {'R26':>7} {'R27':>7} {'R28':>7} "
      f"{'Δ26→27':>8} {'Δ27→28':>8} {'Δ26→28':>8}")
print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

for team in teams:
    p = {}
    for rnd in ROUNDS:
        row = round_frames[rnd][round_frames[rnd]["team"] == team]
        p[rnd] = row["p_mean"].values[0] if len(row) else float("nan")

    d1 = p[27] - p[26]
    d2 = p[28] - p[27]
    d3 = p[28] - p[26]

    flag = ""
    if abs(d3) >= 0.10:
        flag = "  ↑" if d3 > 0 else "  ↓"

    print(f"  {team:<30} {p[26]:>7.3f} {p[27]:>7.3f} {p[28]:>7.3f} "
          f"{d1:>+8.3f} {d2:>+8.3f} {d3:>+8.3f}{flag}")

# ── Final verdict: who is in / out / on the bubble ───────────────────────────
print(f"\n{sep}")
print("  VERDICT at round 28  (ensemble mean probability)")
print(sep)

final = round_frames[28].sort_values("p_mean", ascending=False).reset_index(drop=True)
final.index += 1

safe    = final[final["p_mean"] >= 0.85]
bubble  = final[(final["p_mean"] >= 0.20) & (final["p_mean"] < 0.85)]
out     = final[final["p_mean"] < 0.20]

print(f"\n  SAFE (≥85%) ── {len(safe)} teams")
for _, r in safe.iterrows():
    print(f"    {r['team']:<30}  {r['p_mean']:.1%}")

print(f"\n  ON THE BUBBLE (20–85%) ── {len(bubble)} teams")
for _, r in bubble.iterrows():
    print(f"    {r['team']:<30}  {r['p_mean']:.1%}  (pos={int(r['position'])}, "
          f"pts={int(r['points'])}, gap={int(r['points_from_6th']):+d})")

print(f"\n  OUT (<20%) ── {len(out)} teams")
for _, r in out.iterrows():
    print(f"    {r['team']:<30}  {r['p_mean']:.1%}")
