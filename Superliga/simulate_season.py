"""
Monte Carlo simulation of the remaining 3 rounds (28, 29, 30) to estimate
each team's probability of finishing in the top 6.

Match outcome model (Poisson goals):
  - Each team has a home-attack, home-defense, away-attack, away-defense
    rating derived from the season so far, shrunk toward the league mean
    to avoid overreacting to small samples.
  - A separate "form multiplier" (last 5 games) shifts expected goals up/down.
  - Goals for each side are sampled independently from Poisson distributions.
  - The resulting scoreline determines points (W=3, D=1, L=0).

Simulation:
  - 200,000 independent runs of rounds 28-30.
  - Each run starts from current round 27 standings and adds simulated points.
  - Final standings (pts → gd → gf tiebreak) decide who is top 6.

Output:
  - P(top 6) from simulation
  - Expected final points ± std
  - Per-team fixture difficulty breakdown
  - Last-5 form summary
"""

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
TEST_PATH     = "data/processed/test_2025_2026.csv"
FIXTURES_PATH = "data/seasons/2025_2026/regulat25_26.csv"
N_SIM         = 200_000
SNAPSHOT_RND  = 28
SHRINK_K      = 6      # equivalent games of shrinkage toward league mean
FORM_WEIGHT   = 0.15   # max ±15% scaling of expected goals from form
RNG           = np.random.default_rng(42)

# ── Load current standings (round 27) ───────────────────────────────────────
test  = pd.read_csv(TEST_PATH)
stats = test[test["round"] == SNAPSHOT_RND].copy().reset_index(drop=True)
stats["team"] = stats["team"].str.strip()
stats = stats.set_index("team")

teams = list(stats.index)
n_teams = len(teams)

# ── Current points / gd / gf ────────────────────────────────────────────────
cur_pts = stats["points"].to_dict()
cur_gd  = stats["gd"].to_dict()
cur_gf  = stats["gf"].to_dict()

# ── Compute Poisson ratings ──────────────────────────────────────────────────
# Derive home/away games played from win+draw+loss columns
stats["home_gp"] = stats["home_wins"] + stats["home_draws"] + stats["home_losses"]
stats["away_gp"] = stats["away_wins"] + stats["away_draws"] + stats["away_losses"]

# Raw rates (goals per game), protecting against 0 games played
h_gp  = stats["home_gp"].clip(lower=1)
a_gp  = stats["away_gp"].clip(lower=1)

h_att_raw = stats["home_gf"] / h_gp   # home goals scored per game
h_def_raw = stats["home_ga"] / h_gp   # home goals conceded per game
a_att_raw = stats["away_gf"] / a_gp   # away goals scored per game
a_def_raw = stats["away_ga"] / a_gp   # away goals conceded per game

# League averages (used as Bayesian prior)
mu_h_att = h_att_raw.mean()
mu_h_def = h_def_raw.mean()
mu_a_att = a_att_raw.mean()
mu_a_def = a_def_raw.mean()

# Shrinkage toward league mean: blend team rate with prior
def shrink(raw, prior, k, n):
    return (k * prior + n * raw) / (k + n)

h_att = shrink(h_att_raw, mu_h_att, SHRINK_K, h_gp)
h_def = shrink(h_def_raw, mu_h_def, SHRINK_K, h_gp)
a_att = shrink(a_att_raw, mu_a_att, SHRINK_K, a_gp)
a_def = shrink(a_def_raw, mu_a_def, SHRINK_K, a_gp)

# ── Form multiplier ─────────────────────────────────────────────────────────
# Converts form_pts_last5 (0-15) to a ±FORM_WEIGHT scaling factor.
# A team on maximum form (15 pts) gets a +FORM_WEIGHT boost on attack
# and a -FORM_WEIGHT boost on defense (i.e., concedes less).
form_pts = stats["form_pts_last5"]
form_norm = (form_pts - form_pts.mean()) / (form_pts.std() + 1e-9)  # z-score
form_mult = 1.0 + FORM_WEIGHT * form_norm.clip(-2, 2) / 2  # soft cap ±FORM_WEIGHT

# ── Expected goals for a match ───────────────────────────────────────────────
def expected_goals(home: str, away: str):
    """
    Return (lambda_home, lambda_away): Poisson rates for each side.
    Home team attacks against away team's away-defense, and vice versa.
    """
    # Base rates
    lam_h = h_att[home] * (a_def[away] / mu_a_def) * form_mult[home]
    lam_a = a_att[away] * (h_def[home] / mu_h_def) * form_mult[away]
    # Clip to sensible range
    return max(lam_h, 0.2), max(lam_a, 0.2)

# ── Load remaining fixtures ──────────────────────────────────────────────────
raw = pd.read_csv(FIXTURES_PATH)
raw.columns = [c.strip() for c in raw.columns]
raw["Home Team"] = raw["Home Team"].str.strip()
raw["Away Team"]  = raw["Away Team"].str.strip()
raw["round_num"]  = raw["Round"].str.replace("Round", "").str.strip().astype(int)

future = raw[raw["Home Score"].isna() | (raw["Home Score"].astype(str).str.strip() == "")]
future = future[future["round_num"] >= 29][["round_num", "Home Team", "Away Team"]].reset_index(drop=True)

# Pre-compute Poisson lambdas for every fixture
fixtures = []
for _, row in future.iterrows():
    h, a = row["Home Team"], row["Away Team"]
    if h in teams and a in teams:
        lh, la = expected_goals(h, a)
        fixtures.append({"rnd": row["round_num"], "home": h, "away": a,
                         "lam_h": lh, "lam_a": la})
fixtures = pd.DataFrame(fixtures)

# ── Monte Carlo simulation ───────────────────────────────────────────────────
# Pre-sample all goals at once for speed: shape (N_SIM, n_fixtures, 2)
lam_h_arr = fixtures["lam_h"].values   # (n_fixtures,)
lam_a_arr = fixtures["lam_a"].values

goals_h = RNG.poisson(lam_h_arr, size=(N_SIM, len(fixtures)))  # (N_SIM, F)
goals_a = RNG.poisson(lam_a_arr, size=(N_SIM, len(fixtures)))  # (N_SIM, F)

# Map team names to integer indices
team_idx = {t: i for i, t in enumerate(teams)}
home_idx = [team_idx[h] for h in fixtures["home"]]
away_idx = [team_idx[a] for a in fixtures["away"]]

# Starting state for each simulation
start_pts = np.array([cur_pts[t] for t in teams], dtype=np.int32)   # (n_teams,)
start_gd  = np.array([cur_gd[t]  for t in teams], dtype=np.int32)
start_gf  = np.array([cur_gf[t]  for t in teams], dtype=np.int32)

sim_pts = np.tile(start_pts, (N_SIM, 1))  # (N_SIM, n_teams)
sim_gd  = np.tile(start_gd,  (N_SIM, 1))
sim_gf  = np.tile(start_gf,  (N_SIM, 1))

# Apply each fixture result to all simulations at once
for f in range(len(fixtures)):
    hi = home_idx[f]
    ai = away_idx[f]
    gh = goals_h[:, f]  # (N_SIM,)
    ga = goals_a[:, f]

    gd_h = gh - ga
    gd_a = ga - gh

    # Points
    home_win  = gh > ga
    away_win  = ga > gh
    draw      = gh == ga

    pts_h = home_win * 3 + draw * 1
    pts_a = away_win * 3 + draw * 1

    sim_pts[:, hi] += pts_h
    sim_pts[:, ai] += pts_a
    sim_gd[:, hi]  += gd_h
    sim_gd[:, ai]  += gd_a
    sim_gf[:, hi]  += gh
    sim_gf[:, ai]  += ga

# Rank each team in each simulation using (pts desc, gd desc, gf desc)
# Use a combined sort key: pts×10^8 + gd×10^4 + gf (works for typical ranges)
sort_key = sim_pts * 100_000 + sim_gd * 100 + (sim_gf // 100)  # rough tiebreak
ranks = np.argsort(-sort_key, axis=1)           # (N_SIM, n_teams) indices sorted best→worst
positions = np.empty_like(ranks)
positions[np.arange(N_SIM)[:, None], ranks] = np.arange(1, n_teams + 1)

top6_mask = positions <= 6   # (N_SIM, n_teams) bool

# ── Aggregate results ────────────────────────────────────────────────────────
p_top6       = top6_mask.mean(axis=0)
mean_pts     = sim_pts.mean(axis=0)
std_pts      = sim_pts.std(axis=0)
mean_pos     = positions.mean(axis=0)

results = pd.DataFrame({
    "team":         teams,
    "cur_pts":      start_pts,
    "cur_gd":       start_gd,
    "cur_pos":      [stats.loc[t, "position"] for t in teams],
    "points_from_6th": [stats.loc[t, "points_from_6th"] for t in teams],
    "form_pts_last5": [int(stats.loc[t, "form_pts_last5"]) for t in teams],
    "p_top6":       p_top6,
    "exp_final_pts": mean_pts,
    "std_final_pts": std_pts,
    "exp_final_pos": mean_pos,
}).sort_values("p_top6", ascending=False).reset_index(drop=True)
results.index += 1

# ── Per-team fixture list with difficulty ────────────────────────────────────
def fixture_difficulty(home: str, away: str, perspective: str):
    """Return 'easy/medium/hard' based on opponent's current points."""
    opp = away if perspective == home else home
    opp_pts = cur_pts.get(opp, 0)
    if opp_pts >= 44:    return "HARD  "
    elif opp_pts >= 38:  return "MEDIUM"
    else:                return "EASY  "

print("=" * 74)
print("  SEASON SIMULATION — 200,000 Monte Carlo runs (rounds 29-30)")
print("=" * 74)
print(f"\n  {'#':>2}  {'Team':<30} {'Pos':>4} {'Pts':>4} {'GD':>4} {'Gap':>5} "
      f"{'Form/15':>7} {'P(Top6)':>8} {'Exp.Pts':>8}")
print(f"  {'─'*2}  {'─'*30} {'─'*4} {'─'*4} {'─'*4} {'─'*5} "
      f"{'─'*7} {'─'*8} {'─'*8}")

for rank, row in results.iterrows():
    gap_str = f"{int(row['points_from_6th']):+d}"
    bar_len = int(row["p_top6"] * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    form_str = f"{int(row['form_pts_last5'])}/15"
    print(f"  {rank:>2}  {row['team']:<30} {int(row['cur_pos']):>4} "
          f"{int(row['cur_pts']):>4} {int(row['cur_gd']):>4} {gap_str:>5} "
          f"{form_str:>7}  {row['p_top6']:>7.1%}  {row['exp_final_pts']:>6.1f}±{row['std_final_pts']:.1f}")

# ── Fixture breakdown per team ────────────────────────────────────────────────
print("\n" + "=" * 74)
print("  REMAINING FIXTURES (rounds 29 → 30)")
print("=" * 74)

for rank, row in results.iterrows():
    team = row["team"]
    team_fixtures = []
    for _, fx in future.iterrows():
        if fx["Home Team"] == team or fx["Away Team"] == team:
            is_home = fx["Home Team"] == team
            opp = fx["Away Team"] if is_home else fx["Home Team"]
            venue = "H" if is_home else "A"
            opp_pts = cur_pts.get(opp, 0)
            opp_pos  = int(stats.loc[opp, "position"]) if opp in stats.index else "?"
            diff = fixture_difficulty(fx["Home Team"], fx["Away Team"], team)
            lh, la = expected_goals(fx["Home Team"], fx["Away Team"])
            if is_home:
                exp_str = f"xG {lh:.2f}-{la:.2f}"
            else:
                exp_str = f"xG {la:.2f}-{lh:.2f}"
            team_fixtures.append((int(fx["round_num"]), venue, opp, opp_pos, opp_pts, diff, exp_str))

    form_lst5 = int(row["form_pts_last5"])
    form_bar  = "W" * (form_lst5 // 3) + ("D" if form_lst5 % 3 == 1 else "")
    print(f"\n  {rank:>2}. {team}  (cur: {int(row['cur_pts'])}pts, pos={int(row['cur_pos'])})  "
          f"Form[last5]: {form_lst5}/15")
    for rnd, venue, opp, opp_pos, opp_pts, diff, xg in team_fixtures:
        print(f"      R{rnd} [{venue}] vs {opp:<30} (pos={opp_pos:>2}, {opp_pts}pts) "
              f"[{diff}]  {xg}")

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("  VERDICT")
print("=" * 74)

safe   = results[results["p_top6"] >= 0.85]
bubble = results[(results["p_top6"] >= 0.10) & (results["p_top6"] < 0.85)]
out    = results[results["p_top6"] < 0.10]

print(f"\n  EFFECTIVELY QUALIFIED (≥85%) ── {len(safe)} teams")
for _, r in safe.iterrows():
    print(f"    {r['team']:<32} {r['p_top6']:.1%}  "
          f"(exp final pos: {r['exp_final_pos']:.1f})")

print(f"\n  ON THE BUBBLE (10–85%) ── {len(bubble)} teams")
for _, r in bubble.iterrows():
    pts_needed = int(r["cur_pts"])
    opp_pts_6th = results.iloc[5]["cur_pts"] if len(results) >= 6 else 0
    print(f"    {r['team']:<32} {r['p_top6']:.1%}  "
          f"gap to 6th: {int(r['points_from_6th']):+d}pts  "
          f"exp final pos: {r['exp_final_pos']:.1f}")

print(f"\n  EFFECTIVELY OUT (<10%) ── {len(out)} teams")
for _, r in out.iterrows():
    print(f"    {r['team']:<32} {r['p_top6']:.1%}")

# Save
out_path = f"models/simulation_top6_proba_round{SNAPSHOT_RND}.csv"
results.to_csv(out_path, index=True)
print(f"\n  Saved → {out_path}")
