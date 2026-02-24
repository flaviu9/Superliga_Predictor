# Romanian Superliga — Top-6 Prediction System

A machine learning + Monte Carlo simulation system that predicts which teams will finish in the top 6 of the Romanian Superliga regular season, qualifying for the Championship Play-off.

## Overview

The regular season ends with the top 6 clubs advancing to the Championship Play-off and the bottom 10 entering the Relegation Play-off. This system estimates, at any point mid-season, the probability that each team will finish in the top 6.

**Three-stage pipeline:**

1. **Feature engineering** — transform raw match results into cumulative team-round statistics
2. **Machine learning** — train classifiers on 5 completed seasons to learn top-6 patterns
3. **Monte Carlo simulation** — simulate remaining fixtures (200,000 runs) to produce final top-6 probabilities

## Results (2025/26, Round 28)

| # | Team | P(top 6) | Gap to 6th |
|---|---|---|---|
| 1 | Universitatea Craiova | **100.0%** | +10 pts |
| 2 | Dinamo București | **100.0%** | +6 pts |
| 3 | Rapid București | **100.0%** | +6 pts |
| 4 | Universitatea Cluj | **97.4%** | +2 pts |
| 5 | CFR Cluj | **84.9%** | +1 pt |
| 6 | Argeș Pitești | **78.9%** | 0 (6th) |
| 7 | FCSB | **17.4%** | −3 pts |
| 8 | Botoșani | **12.9%** | −4 pts |
| 9 | UTA Arad | **4.9%** | −4 pts |
| 10 | Oțelul Galați | **3.5%** | −5 pts |
| 11 | Farul Constanța | **0.0%** | −9 pts |

See [REPORT.md](REPORT.md) for the full technical analysis.

## Models

| Model | AUC (LOGO-CV) | F1 |
|---|---|---|
| **Random Forest** | **0.906** | **0.763** |
| Logistic Regression | 0.896 | 0.738 |
| XGBoost | 0.884 | 0.708 |

Evaluation uses **Leave-One-Season-Out cross-validation** — each fold trains on 4 seasons and validates on the held-out 5th, which is the most realistic split for time-structured data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib
```

## Usage

Run the pipeline in order:

```bash
# 1. Build features from raw CSVs
python build_dataset.py

# 2. Train models and evaluate (Leave-One-Season-Out CV)
python train_models.py

# 3. Generate ML predictions at rounds 26, 27, 28
python predict_final_rounds.py

# 4. Run Monte Carlo simulation for final-rounds probabilities
python simulate_season.py
```

## Project Structure

```
Superliga/
├── data/
│   ├── seasons/          ← raw match CSVs per season (2020/21 – 2025/26)
│   └── processed/
│       ├── all_seasons.csv         ← training set (2,320 rows)
│       └── test_2025_2026.csv      ← current season test set (no labels)
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── predictions_round28.csv              ← ML snapshot predictions
│   ├── simulation_top6_proba_round27.csv    ← Monte Carlo probs after round 27
│   └── simulation_top6_proba_round28.csv    ← Monte Carlo probs after round 28
├── build_dataset.py        ← feature engineering pipeline
├── train_models.py         ← ML training + LOGO-CV evaluation
├── predict_final_rounds.py ← ML predictions at rounds 26/27/28
├── simulate_season.py      ← Monte Carlo simulation
└── REPORT.md               ← full technical report
```

## Key Features

- **Cumulative statistics** per team per round (points, GD, form, home/away splits)
- **`points_from_6th`** — signed gap to the 6th-place team; the strongest single predictor
- **Bayesian shrinkage** on attack/defense ratings in the simulation (K=6 prior games)
- **Form multiplier** scales Poisson goal rates by recent form (last 5 games)
- **Vectorised simulation** — 200,000 season samples drawn in a single NumPy operation

## Data

One CSV per season covering 2020/21 through 2025/26. Each file has one row per match:

| Column | Description |
|---|---|
| `Round` | Round label (Round 1 – Round 30) |
| `Home Team` / `Away Team` | Team names |
| `Home Score` / `Away Score` | Goals scored |

16 teams, double round-robin, 30 rounds, 240 matches per season.
