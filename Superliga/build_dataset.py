"""
Build a team-round dataset with cumulative standings for all seasons.
Each row = (season, team, round) with cumulative stats + top-6 label.

Notes:
  - Team names are stripped of whitespace (fixes "Dinamo Bucuresti " duplicate).
  - Round 30 is excluded: at round 30 the current position IS the final position,
    making is_top6 trivially derivable (data leakage).
  - Output: one combined CSV at data/processed/all_seasons.csv
"""

import os
import pandas as pd

SEASON_FILES = {
    "2020_2021": "data/seasons/2020_2021/regulat20_21.csv",
    "2021_2022": "data/seasons/2021_2022/regulat21_22.csv",
    "2022_2023": "data/seasons/2022_2023/regulat22_23.csv",
    "2023_2024": "data/seasons/2023_2024/regulat23_24.csv",
    "2024_2025": "data/seasons/2024_2025/regulat24_25.csv",
}

OUTPUT_PATH      = "data/processed/all_seasons.csv"
TEST_OUTPUT_PATH = "data/processed/test_2025_2026.csv"
TEST_FILE        = "data/seasons/2025_2026/regulat25_26.csv"
MAX_ROUND        = 30   # final round — excluded from features, used only for labels


def parse_round(round_str):
    try:
        return int(str(round_str).replace("Round", "").strip())
    except ValueError:
        return None


def compute_match_record(row, team_col, score_col, opp_score_col, venue):
    gs = row[score_col]
    ga = row[opp_score_col]
    if gs > ga:
        result, pts = "W", 3
    elif gs == ga:
        result, pts = "D", 1
    else:
        result, pts = "L", 0
    return {
        "team":     row[team_col].strip(),   # strip to fix trailing-space duplicates
        "venue":    venue,
        "gf":       gs,
        "ga":       ga,
        "result":   result,
        "points":   pts,
        "round_num": row["round_num"],
    }


def final_standings(matches):
    """Return a dict team → (final_position, is_top6) based on all 30 rounds."""
    cum = matches.copy()
    agg = cum.groupby("team").agg(
        points=("points", "sum"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    ).reset_index()
    agg["gd"] = agg["gf"] - agg["ga"]
    agg = agg.sort_values(["points", "gd", "gf"], ascending=[False, False, False]).reset_index(drop=True)
    agg["final_position"] = agg.index + 1
    agg["is_top6"] = (agg["final_position"] <= 6).astype(int)
    return agg.set_index("team")[["final_position", "is_top6"]].to_dict("index")


def process_season(season_label, filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    df["round_num"] = df["Round"].apply(parse_round)
    df = df.dropna(subset=["round_num", "Home Score", "Away Score"])
    df["Home Score"] = df["Home Score"].astype(int)
    df["Away Score"] = df["Away Score"].astype(int)
    df = df.sort_values(["round_num", "dateEvent"]).reset_index(drop=True)

    # Flatten to one record per team per match
    records = []
    for _, row in df.iterrows():
        records.append(compute_match_record(row, "Home Team", "Home Score", "Away Score", "home"))
        records.append(compute_match_record(row, "Away Team", "Away Score", "Home Score", "away"))

    matches = pd.DataFrame(records)

    # Compute final standings (using ALL 30 rounds) for the label
    final_info = final_standings(matches)

    # Build feature rows for rounds 1-29 only (round 30 excluded — data leakage)
    all_teams  = sorted(matches["team"].unique())
    build_rounds = range(1, MAX_ROUND)   # 1..29

    rows = []
    for team in all_teams:
        tm   = matches[matches["team"] == team].sort_values("round_num").reset_index(drop=True)
        hm   = tm[tm["venue"] == "home"]
        am   = tm[tm["venue"] == "away"]

        for rnd in build_rounds:
            cum  = tm[tm["round_num"] <= rnd]
            if cum.empty:
                continue

            hcum = cum[cum["venue"] == "home"]
            acum = cum[cum["venue"] == "away"]

            gp     = len(cum)
            wins   = (cum["result"] == "W").sum()
            draws  = (cum["result"] == "D").sum()
            losses = (cum["result"] == "L").sum()
            pts    = int(cum["points"].sum())
            gf     = int(cum["gf"].sum())
            ga     = int(cum["ga"].sum())
            gd     = gf - ga
            ppg    = round(pts / gp, 4)

            h_wins   = int((hcum["result"] == "W").sum())
            h_draws  = int((hcum["result"] == "D").sum())
            h_losses = int((hcum["result"] == "L").sum())
            h_gf     = int(hcum["gf"].sum())
            h_ga     = int(hcum["ga"].sum())
            h_gd     = h_gf - h_ga

            a_wins   = int((acum["result"] == "W").sum())
            a_draws  = int((acum["result"] == "D").sum())
            a_losses = int((acum["result"] == "L").sum())
            a_gf     = int(acum["gf"].sum())
            a_ga     = int(acum["ga"].sum())
            a_gd     = a_gf - a_ga

            last5       = cum.tail(5)
            form_pts    = int(last5["points"].sum())
            form_gd     = int((last5["gf"] - last5["ga"]).sum())
            form_wins   = int((last5["result"] == "W").sum())

            rows.append({
                "season":        season_label,
                "team":          team,
                "round":         rnd,
                "gp":            gp,
                "wins":          wins,
                "draws":         draws,
                "losses":        losses,
                "points":        pts,
                "gf":            gf,
                "ga":            ga,
                "gd":            gd,
                "ppg":           ppg,
                "home_wins":     h_wins,
                "home_draws":    h_draws,
                "home_losses":   h_losses,
                "home_gf":       h_gf,
                "home_ga":       h_ga,
                "home_gd":       h_gd,
                "away_wins":     a_wins,
                "away_draws":    a_draws,
                "away_losses":   a_losses,
                "away_gf":       a_gf,
                "away_ga":       a_ga,
                "away_gd":       a_gd,
                "form_pts_last5":  form_pts,
                "form_gd_last5":   form_gd,
                "form_wins_last5": form_wins,
            })

    season_df = pd.DataFrame(rows)

    # --- Position at each round ---
    season_df = season_df.sort_values(
        ["round", "points", "gd", "gf"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    season_df["position"] = season_df.groupby("round").cumcount() + 1

    # --- Points gap to 6th place at each round ---
    sixth_pts = season_df.groupby("round").apply(
        lambda g: g.nlargest(6, "points")["points"].min()
    ).rename("sixth_pts")
    season_df = season_df.merge(sixth_pts, on="round")
    season_df["points_from_6th"] = season_df["points"] - season_df["sixth_pts"]
    season_df.drop(columns="sixth_pts", inplace=True)

    # --- Labels from full-season final standings ---
    season_df["final_position"] = season_df["team"].map(
        lambda t: final_info[t]["final_position"]
    )
    season_df["is_top6"] = season_df["team"].map(
        lambda t: final_info[t]["is_top6"]
    )

    col_order = [
        "season", "team", "round",
        # Cumulative overall
        "gp", "wins", "draws", "losses",
        "points", "gf", "ga", "gd", "ppg",
        # Home split
        "home_wins", "home_draws", "home_losses", "home_gf", "home_ga", "home_gd",
        # Away split
        "away_wins", "away_draws", "away_losses", "away_gf", "away_ga", "away_gd",
        # Form (last 5 matches)
        "form_pts_last5", "form_gd_last5", "form_wins_last5",
        # Standings context
        "position", "points_from_6th",
        # Labels (never use as features)
        "final_position", "is_top6",
    ]
    return season_df[col_order]


def build_test_set(season_label, filepath):
    """
    Process an incomplete season for use as a test set.
    Only rows with actual scores are used (completed matches).
    is_top6 and final_position are left as NaN — the season is not over.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    df["round_num"] = df["Round"].apply(parse_round)

    # Keep only completed matches (non-empty scores)
    df = df.dropna(subset=["Home Score", "Away Score"])
    df = df[df["Home Score"].astype(str).str.strip() != ""]
    df["Home Score"] = df["Home Score"].astype(int)
    df["Away Score"] = df["Away Score"].astype(int)
    df = df.dropna(subset=["round_num"])
    df = df.sort_values(["round_num", "dateEvent"]).reset_index(drop=True)

    last_complete_round = int(df["round_num"].max())

    # Flatten to one record per team per match
    records = []
    for _, row in df.iterrows():
        records.append(compute_match_record(row, "Home Team", "Home Score", "Away Score", "home"))
        records.append(compute_match_record(row, "Away Team", "Away Score", "Home Score", "away"))

    matches = pd.DataFrame(records)
    all_teams    = sorted(matches["team"].unique())
    build_rounds = range(1, last_complete_round + 1)

    rows = []
    for team in all_teams:
        tm   = matches[matches["team"] == team].sort_values("round_num").reset_index(drop=True)

        for rnd in build_rounds:
            cum  = tm[tm["round_num"] <= rnd]
            if cum.empty:
                continue

            hcum = cum[cum["venue"] == "home"]
            acum = cum[cum["venue"] == "away"]

            gp     = len(cum)
            wins   = (cum["result"] == "W").sum()
            draws  = (cum["result"] == "D").sum()
            losses = (cum["result"] == "L").sum()
            pts    = int(cum["points"].sum())
            gf     = int(cum["gf"].sum())
            ga     = int(cum["ga"].sum())
            gd     = gf - ga
            ppg    = round(pts / gp, 4)

            h_wins   = int((hcum["result"] == "W").sum())
            h_draws  = int((hcum["result"] == "D").sum())
            h_losses = int((hcum["result"] == "L").sum())
            h_gf     = int(hcum["gf"].sum())
            h_ga     = int(hcum["ga"].sum())
            h_gd     = h_gf - h_ga

            a_wins   = int((acum["result"] == "W").sum())
            a_draws  = int((acum["result"] == "D").sum())
            a_losses = int((acum["result"] == "L").sum())
            a_gf     = int(acum["gf"].sum())
            a_ga     = int(acum["ga"].sum())
            a_gd     = a_gf - a_ga

            last5       = cum.tail(5)
            form_pts    = int(last5["points"].sum())
            form_gd     = int((last5["gf"] - last5["ga"]).sum())
            form_wins   = int((last5["result"] == "W").sum())

            rows.append({
                "season":          season_label,
                "team":            team,
                "round":           rnd,
                "gp":              gp,
                "wins":            wins,
                "draws":           draws,
                "losses":          losses,
                "points":          pts,
                "gf":              gf,
                "ga":              ga,
                "gd":              gd,
                "ppg":             ppg,
                "home_wins":       h_wins,
                "home_draws":      h_draws,
                "home_losses":     h_losses,
                "home_gf":         h_gf,
                "home_ga":         h_ga,
                "home_gd":         h_gd,
                "away_wins":       a_wins,
                "away_draws":      a_draws,
                "away_losses":     a_losses,
                "away_gf":         a_gf,
                "away_ga":         a_ga,
                "away_gd":         a_gd,
                "form_pts_last5":  form_pts,
                "form_gd_last5":   form_gd,
                "form_wins_last5": form_wins,
            })

    test_df = pd.DataFrame(rows)

    # Position at each round
    test_df = test_df.sort_values(
        ["round", "points", "gd", "gf"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    test_df["position"] = test_df.groupby("round").cumcount() + 1

    # Points gap to 6th place at each round
    sixth_pts = test_df.groupby("round").apply(
        lambda g: g.nlargest(6, "points")["points"].min()
    ).rename("sixth_pts")
    test_df = test_df.merge(sixth_pts, on="round")
    test_df["points_from_6th"] = test_df["points"] - test_df["sixth_pts"]
    test_df.drop(columns="sixth_pts", inplace=True)

    # Labels are unknown — season not finished
    test_df["final_position"] = pd.NA
    test_df["is_top6"]        = pd.NA

    col_order = [
        "season", "team", "round",
        "gp", "wins", "draws", "losses",
        "points", "gf", "ga", "gd", "ppg",
        "home_wins", "home_draws", "home_losses", "home_gf", "home_ga", "home_gd",
        "away_wins", "away_draws", "away_losses", "away_gf", "away_ga", "away_gd",
        "form_pts_last5", "form_gd_last5", "form_wins_last5",
        "position", "points_from_6th",
        "final_position", "is_top6",
    ]
    return test_df[col_order], last_complete_round


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # --- Training set ---
    all_dfs = []
    for season_label, filepath in SEASON_FILES.items():
        print(f"Processing {season_label} ...")
        df = process_season(season_label, filepath)
        all_dfs.append(df)

        teams  = df["team"].nunique()
        rounds = df["round"].max()
        top6   = df[["team", "is_top6"]].drop_duplicates()["is_top6"].sum()
        print(f"  {teams} teams · rounds 1-{rounds} · {len(df)} rows  |  top-6 teams: {top6}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTraining set: {len(combined)} rows · {combined['team'].nunique()} unique teams")
    print(f"Saved → {OUTPUT_PATH}")
    print(f"\nLabel balance:\n{combined[['season','team','is_top6']].drop_duplicates().groupby('is_top6').size()}")

    # --- Test set (current incomplete season) ---
    print(f"\nBuilding test set from {TEST_FILE} ...")
    test_df, last_round = build_test_set("2025_2026", TEST_FILE)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print(f"  {test_df['team'].nunique()} teams · rounds 1-{last_round} · {len(test_df)} rows")
    print(f"  Saved → {TEST_OUTPUT_PATH}")
    print(f"\nCurrent standings at round {last_round}:")
    snap = test_df[test_df["round"] == last_round].sort_values("position")
    print(snap[["team","round","gp","points","gd","position","points_from_6th"]].to_string(index=False))


if __name__ == "__main__":
    main()
