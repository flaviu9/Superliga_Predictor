# Romanian Superliga — Top-6 Prediction System
## Technical Report

---

## 1. Project Overview

The Romanian Superliga regular season ends with the top 6 clubs advancing to the Championship Play-off and the bottom 10 entering the Relegation Play-off. The goal of this project is to predict, at any point during the regular season, **which teams will finish in the top 6**, and to produce reliable probability estimates for the current (2025/26) incomplete season.

The system is built in three stages:

1. **Feature engineering** — transform raw match results into cumulative team-round statistics
2. **Machine learning models** — train classifiers on 5 completed seasons to learn what top-6 teams look like mid-season
3. **Monte Carlo simulation** — simulate the remaining fixtures to produce final top-6 probabilities grounded in fixture difficulty and current form

---

## 2. Data

### 2.1 Raw data

One CSV file per season (`regulat<YY>_<YY>.csv`) covering seasons 2020/21 through 2025/26.
Each file contains one row per match with the following columns:

| Column | Description |
|---|---|
| `idEvent` | Unique match identifier |
| `dateEvent` | Match date (YYYY-MM-DD) |
| `Round` | Round label ("Round 1" … "Round 30") |
| `Home Team` | Home team name |
| `Home Score` | Home goals scored |
| `Away Team` | Away team name |
| `Away Score` | Away goals scored |

Each season has 16 teams playing a double round-robin (30 rounds, 8 matches per round = 240 matches total).

### 2.2 Data quality issues found and fixed

| Issue | Season | Fix applied |
|---|---|---|
| `"Dinamo Bucuresti "` (trailing space) counted as a 17th team | 2021/22 | `.strip()` applied to all team name strings |
| Some matches played on different dates than their logical round (postponements) | 2025/26 and others | `round_num` parsed from the `Round` column, not from `dateEvent` |
| Rounds 29–30 of 2025/26 have no scores yet | 2025/26 | Rows without scores filtered out; only completed matches used |

---

## 3. Feature Engineering (`build_dataset.py`)

For each team at each round, all features are **cumulative** — they describe the team's season-to-date performance up to and including that round. This mirrors the information a model would have access to when making a prediction mid-season.

### 3.1 Feature groups

| Group | Features | Description |
|---|---|---|
| **Identity** | `season`, `team`, `round` | Not used as model features |
| **Cumulative overall** | `gp`, `wins`, `draws`, `losses`, `points`, `gf`, `ga`, `gd`, `ppg` | Season totals up to this round |
| **Home split** | `home_wins/draws/losses/gf/ga/gd` | Performance in home matches only |
| **Away split** | `away_wins/draws/losses/gf/ga/gd` | Performance in away matches only |
| **Form (last 5)** | `form_pts_last5`, `form_gd_last5`, `form_wins_last5` | Recent form: points, GD, wins in the last 5 games played |
| **Standings context** | `position`, `points_from_6th` | Table position and points gap to 6th place |
| **Labels** | `final_position`, `is_top6` | Never used as features; target variable |

### 3.2 Why `round 30` is excluded from the training set

At round 30 (the final round), a team's current position **is** its final position, so `is_top6` is directly derivable from `position ≤ 6`. Including round 30 rows as training examples would constitute **data leakage** — the model would learn a trivial rule rather than genuine mid-season patterns. Rounds 1–29 are used for training; round 30 statistics are only used to compute the ground-truth `is_top6` label.

### 3.3 `points_from_6th`

This engineered feature computes the signed difference between a team's cumulative points and the points held by the 6th-place team at that same round snapshot. It is the single most direct numerical proxy for whether a team is inside or outside the promotion zone.

```
points_from_6th = team_points − points_of_6th_place_team_at_same_round
```

A positive value means the team is above the cut; negative means below.

### 3.4 Output files

| File | Contents |
|---|---|
| `data/processed/all_seasons.csv` | 2,320 rows (5 seasons × 16 teams × 29 rounds). Training set. |
| `data/processed/test_2025_2026.csv` | 448 rows (1 season × 16 teams × 28 completed rounds). Test set. `is_top6` = NaN. |

---

## 4. Machine Learning Models (`train_models.py`)

### 4.1 Problem framing

This is a **binary classification** problem: given a team's cumulative statistics at a particular round, predict whether the team will finish in the top 6 at the end of the season.

- Each row in the training set is one (team, round) observation
- The label `is_top6` is the same for all 29 rows belonging to the same team-season
- Label distribution across team-seasons: **30 top-6** vs **50 non-top-6** (mild imbalance, 3:5 ratio)

### 4.2 Models

Three classifiers were trained, covering a range of model families:

#### Logistic Regression
- Scaled features with `StandardScaler` (required for L2 regularisation to work correctly)
- `C = 0.1` (moderate regularisation, chosen to avoid overfitting to small dataset)
- `class_weight = 'balanced'` to compensate for the label imbalance

#### Random Forest
- 500 trees, `max_depth = 6`, `min_samples_leaf = 10`
- `class_weight = 'balanced'`
- Ensemble averaging reduces variance from individual decision trees

#### XGBoost
- 300 rounds of boosting, `learning_rate = 0.05`, `max_depth = 4`
- `subsample = 0.8`, `colsample_bytree = 0.8` (stochastic boosting)
- `scale_pos_weight ≈ 1.67` (ratio of negative to positive examples)

### 4.3 Evaluation strategy — Leave-One-Season-Out cross-validation

With only 5 completed seasons, standard k-fold CV is inappropriate because rows from the same team-season would appear in both train and validation, artificially inflating scores. Instead, **Leave-One-Season-Out (LOGO)** CV was used: each fold trains on 4 seasons and validates on the held-out 5th season. This is the most honest evaluation strategy available — the model truly never sees the validation season during training.

### 4.4 Results

| Model | Mean AUC | Mean F1 | Mean Precision | Mean Recall |
|---|---|---|---|---|
| **Random Forest** | **0.906** | **0.763** | 0.764 | 0.766 |
| Logistic Regression | 0.896 | 0.738 | 0.708 | 0.777 |
| XGBoost | 0.884 | 0.708 | 0.706 | 0.714 |

Random Forest is the strongest model overall. Logistic Regression is competitive and more interpretable. XGBoost underperforms slightly — likely because boosted trees need more data to regularise well, and 5 seasons is a limited training set.

**Per-season AUC breakdown:**

| Season (held out) | LR | RF | XGB |
|---|---|---|---|
| 2020/21 | 0.911 | 0.934 | 0.880 |
| 2021/22 | 0.869 | 0.861 | 0.853 |
| 2022/23 | 0.927 | 0.931 | 0.926 |
| 2023/24 | 0.906 | 0.905 | 0.903 |
| 2024/25 | 0.869 | 0.899 | 0.857 |

2021/22 is consistently the hardest season to predict — it had a particularly competitive mid-table, reflected in lower AUC for all models.

### 4.5 Feature importances (consistent across models)

| Rank | Feature | Why it matters |
|---|---|---|
| 1 | `gd` — goal difference | Strongest overall quality signal; separates dominant teams from the pack |
| 2 | `ppg` — points per game | Normalises for rounds played; captures consistency |
| 3 | `position` — current table position | Direct standing in the league |
| 4 | `points_from_6th` | Most direct signal: how far from the cut |
| 5 | `home_gd` — home goal difference | Strong teams dominate at home; home form is more stable |
| 6 | `form_gd_last5` | Recent trajectory |
| 7 | `away_gd` | Away GD separates the truly good from home-only performers |

Notably, raw `points` is less important than `gd` and `ppg` — this makes sense because by mid-season many teams have similar point totals, but goal difference reveals the quality gap more clearly.

---

## 5. Monte Carlo Season Simulation (`simulate_season.py`)

The ML models produce a static prediction based on round 28 statistics. For the current incomplete season, a more informative approach is to explicitly model what will happen in the remaining 2 rounds and estimate final standings probabilistically.

### 5.1 Match outcome model — Independent Poisson goals

For each remaining fixture, the number of goals scored by each side is modelled as an independent Poisson random variable:

```
goals_home ~ Poisson(λ_home)
goals_away ~ Poisson(λ_away)
```

The Poisson rates are computed from each team's season-to-date home/away statistics:

```
λ_home = h_att[home] × (a_def[away] / μ_away_def) × form_mult[home]
λ_away = a_att[away] × (h_def[home]  / μ_home_def) × form_mult[away]
```

Where:
- `h_att[team]` = team's home goals scored per game (after shrinkage)
- `a_def[team]` = team's away goals conceded per game (after shrinkage)
- `μ_away_def` = league average away goals conceded per game
- `form_mult[team]` = form-based scaling factor (see §5.2)

### 5.2 Bayesian shrinkage

With ~14 home and ~14 away games per team at round 28, individual attack/defense rates can be noisy. A **Bayesian shrinkage** toward the league mean is applied:

```
shrunk_rate = (K × league_mean + N × team_rate) / (K + N)
```

`K = 6` (equivalent to 6 games of prior data). This prevents a team with one exceptional run of results from being projected to score 4 goals per game for the rest of the season.

### 5.3 Form multiplier

Recent form is incorporated as a scaling factor on expected goals:

```
form_mult = 1.0 + 0.15 × clip(z_score(form_pts_last5), −2, 2) / 2
```

- A team on maximum form (15/15 points, like CFR Cluj at round 28) gets a ~+7.5% boost on attack
- A cold team (0/15 points, like Metaloglobus) gets a ~−7.5% penalty
- The cap at ±2 standard deviations prevents extreme outliers from distorting predictions

### 5.4 Simulation procedure

1. **Pre-compute** Poisson λ values for all 16 remaining fixtures (8 per round × 2 rounds)
2. **Vectorised sampling**: draw `(200,000 × 16)` Poisson samples in a single NumPy operation — all simulations run simultaneously without Python loops
3. **Accumulate** goals and points to the round 28 standings for each simulation
4. **Rank** all 16 teams in each simulation by the standard tiebreak (pts → GD → GF)
5. **Count** how many simulations each team finishes in the top 6

### 5.5 Final simulation results (round 28 → end of season)

| # | Team | P(top 6) | Exp. final pts | Current gap |
|---|---|---|---|---|
| 1 | Universitatea Craiova | **100.0%** | 60.1 ± 1.5 | +10 pts |
| 2 | Dinamo București | **100.0%** | 55.1 ± 1.8 | +6 pts |
| 3 | Rapid București | **100.0%** | 55.2 ± 1.8 | +6 pts |
| 4 | Universitatea Cluj | **97.4%** | 51.1 ± 1.8 | +2 pts |
| 5 | CFR Cluj | **84.9%** | 49.7 ± 1.9 | +1 pt, form 15/15 |
| 6 | Argeș Pitești | **78.9%** | 49.0 ± 1.7 | 0 pts (6th) |
| 7 | FCSB | **17.4%** | 45.8 ± 1.8 | −3 pts |
| 8 | Botoșani | **12.9%** | 45.5 ± 1.8 | −4 pts |
| 9 | UTA Arad | **4.9%** | 45.0 ± 1.8 | −4 pts, negative GD |
| 10 | Oțelul Galați | **3.5%** | 44.2 ± 1.7 | −5 pts |
| 11 | Farul Constanța | **0.0%** | 39.6 ± 1.8 | −9 pts |
| 12–16 | (remaining teams) | **< 0.1%** | — | Eliminated |

**Key observations:**
- **Spots 1–4** are effectively decided with 2 rounds left — Craiova, Dinamo, Rapid at 100% and Cluj at 97.4%
- **Spots 5–6** remain contested: CFR Cluj (84.9%, perfect form 15/15) and Argeș Pitești (78.9%, currently 6th) are both likely qualifiers but neither is mathematically safe
- **CFR Cluj's 15/15 form** continues to be the most impactful single factor — they face Farul away in R29 before a tough R30 vs Dinamo
- **FCSB (17.4%) and Botoșani (12.9%)** still have slim chances: both need to win out while hoping CFR and Argeș both drop points
- **UTA Arad** is punished by their negative GD (−3): despite equal points with Botoșani, tiebreaks frequently go against them

---

## 6. File Structure

```
Superliga/
├── data/
│   ├── seasons/
│   │   ├── 2020_2021/regulat20_21.csv
│   │   ├── 2021_2022/regulat21_22.csv
│   │   ├── 2022_2023/regulat22_23.csv
│   │   ├── 2023_2024/regulat23_24.csv
│   │   ├── 2024_2025/regulat24_25.csv
│   │   └── 2025_2026/regulat25_26.csv      ← current season (test)
│   └── processed/
│       ├── all_seasons.csv                  ← training set (2,320 rows)
│       └── test_2025_2026.csv               ← test set (448 rows, no labels)
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── predictions_round28.csv              ← ML model snapshot predictions
│   ├── simulation_top6_proba_round27.csv    ← Monte Carlo probs after round 27
│   └── simulation_top6_proba_round28.csv    ← Monte Carlo probs after round 28
├── build_dataset.py                         ← feature engineering pipeline
├── train_models.py                          ← ML training + LOGO-CV evaluation
├── predict_final_rounds.py                  ← ML predictions at rounds 25/26/27
├── simulate_season.py                       ← Monte Carlo simulation
└── REPORT.md                                ← this file
```

---

## 7. Limitations

### 7.1 Small training set
Five seasons is a very limited dataset. Each team-season produces only 29 training rows (rounds 1–29), but **the label is identical for all 29 rows of the same team-season** — meaning the effective number of independent observations is only 80 (16 teams × 5 seasons). This severely constrains how complex a model can be before overfitting.

### 7.2 No team identity features
The model treats every team as anonymous — it sees statistics, not names. A promoted team in their first Superliga season and a historic champion with the same 15-game stats look identical to the model. In reality, squad depth, budget, and managerial experience all influence how well a team sustains performance, especially under pressure.

### 7.3 Independent Poisson assumption
The Poisson goal model assumes goals by each team are independent. In reality, the scoreline influences both teams' behaviour (a team winning 2–0 may defend more conservatively). The **Dixon-Coles correction** for low-scoring scorelines (0–0, 1–0, 0–1, 1–1) partially addresses this but was not implemented here.

### 7.4 No injury / suspension data
The model has no access to squad availability. A key player's absence in a crucial match can completely change the expected outcome.

### 7.5 Fixture congestion and schedule effects
Some matches in the dataset were postponed and played weeks after their original round window. The model treats `round_num` as the logical unit of time and assigns postponed matches to their correct round, but it does not account for the fatigue effects of congested fixtures.

---

## 8. Future Improvements

### 8.1 More data
Acquiring additional seasons (ideally 10+) would substantially improve model reliability and enable more robust cross-validation. Incorporating data from similar leagues (Bulgarian First League, Slovak Super Liga) could expand the training set further through transfer learning.

### 8.2 Dixon-Coles match model
Replace the independent Poisson model with the **Dixon-Coles (1997)** correction, which adjusts probabilities for low-scoring draws. This is the industry standard for football match prediction and typically improves calibration by 2–5%.

### 8.3 ELO or TrueSkill team ratings
Replace per-season attack/defense ratings with a continuous rating system (ELO or TrueSkill) that carries information across seasons. A team that finished 1st last season starts the new season with a higher prior — this would be especially valuable at round 1 when no in-season data exists yet.

### 8.4 Time-weighted features
Current cumulative features weight all rounds equally. Exponentially decaying weights (more recent games count more) would better reflect team form and squad changes mid-season.

### 8.5 Remaining schedule difficulty as a feature
Currently, `points_from_6th` and `position` describe where a team is. Adding a pre-computed "remaining schedule difficulty" feature (average opponent strength over the last N rounds) into the ML models would allow them to anticipate fixture congestion — rather than relying on the simulation stage alone.

### 8.6 Neural network / sequence model
The team's trajectory through the season (a sequence of 29 round-level vectors) is naturally modelled by a **recurrent network (LSTM/GRU)** or a **Transformer**. These architectures can capture patterns like "a team that started slowly but accelerated in the second half" — which a flat feature vector cannot express. This requires more data to train well.

### 8.7 Live pipeline
The current workflow is batch-run manually. A production pipeline would:
- Automatically fetch new results after each matchday
- Re-run `build_dataset.py` and `simulate_season.py`
- Publish updated probabilities via a dashboard or API

### 8.8 Calibration
The model probabilities are not explicitly calibrated. **Platt scaling** or **isotonic regression** applied to the classifier outputs would ensure that a predicted 70% probability truly corresponds to ~70% empirical frequency across historical seasons.

---

## 9. Summary

| Stage | Tool | Key output |
|---|---|---|
| Feature engineering | `build_dataset.py` | 2,320-row training set + 432-row test set |
| Model training & evaluation | `train_models.py` | Random Forest: AUC 0.906, F1 0.763 (LOGO-CV) |
| Mid-season ML snapshot | `predict_final_rounds.py` | Per-team P(top 6) at rounds 26, 27, 28 |
| Final-rounds simulation | `simulate_season.py` | Per-team P(top 6) from 200,000 simulated seasons |

The simulation gives the most reliable final estimate because it explicitly propagates uncertainty through the remaining fixtures rather than extrapolating from a static snapshot. At round 28, with 2 rounds remaining, spots 1–4 are effectively decided. The real contest is for 5th and 6th, where CFR Cluj (84.9%) and Argeș Pitești (78.9%) hold the advantage, with FCSB (17.4%) and Botoșani (12.9%) still mathematically alive.
