"""
NBA team-season analysis (2000–2023).

What this script does
- Loads team-season stats, engineers efficiency features, and evaluates Ridge models.
- Optionally merges playoffs (2000–2021) and payroll (1990–2023; matched 2000–2021).
- Exports one enriched dataset for downstream viz (Tableau) and saves a summary figure.

Outputs
- outputs/nba_team_seasons_enriched.csv
- outputs/figures/league_trends_and_models.png

How to run
    python nba_analysis.py

Config
- Toggle USE_PLAYOFFS / USE_PAYROLL
- QUIET_MODE=True for a clean portfolio run; set SHOW_PLOTS=True for interactive exploration.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Data file paths (expected in data/raw/ unless you change these)
CSV_PATH = "data/raw/nba_team_stats_00_to_23.csv"  # Regular season team statistics
PLAYOFFS_CSV_PATH = "data/raw/nba_team_stats_playoffs_00_to_21.csv"  # Playoff outcomes
PAYROLL_CSV_PATH = "data/raw/nba_team_payroll_1990_to_2023.csv"  # Team payroll data

# Feature flags
USE_PLAYOFFS = True
USE_PAYROLL = True

# Output / logging behavior
QUIET_MODE = True   # When True, only prints a small summary
VERBOSE = False     # Extra diagnostics (overrides QUIET_MODE for debug prints)

# Outputs
SAVE_ENRICHED_CSV = True  # Writes one merged/enriched CSV to outputs/
ENRICHED_CSV_NAME = "nba_team_seasons_enriched.csv"
SAVE_PLOTS = True         # Save plots to outputs/figures/
SHOW_PLOTS = False        # Keep False for portfolio scripts; set True when exploring locally


# HELPER FUNCTIONS


def season_start_year(season_str: str) -> int:
    """Return the start year from a season label like '2000-01'."""
    return int(str(season_str).split("-")[0])


# DATA HYGIENE FUNCTION

def validate_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data hygiene: coerce numeric cols, check ranges, drop invalid rows."""
    df = df.copy()

    # Normalize identifiers
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).str.strip()

    # Coerce key numeric columns (bad strings -> NaN)
    numeric_cols = [
        "wins", "games_played", "plus_minus",
        "field_goals_attempted", "field_goals_made",
        "three_pointers_attempted", "three_pointers_made",
        "free_throw_attempted",
        "turnovers", "assists", "rebounds", "offensive_rebounds",
        "steals", "blocks", "personal_fouls",
        "win_percentage",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Flag rows that are structurally impossible (these are true data errors, not "missing" values).
    before = df.shape[0]
    invalid = pd.Series(False, index=df.index)

    if "games_played" in df.columns:
        invalid |= df["games_played"].isna() | (df["games_played"] <= 0)

    if "wins" in df.columns and "games_played" in df.columns:
        invalid |= df["wins"].isna() | (df["wins"] < 0) | (df["wins"] > df["games_played"])

    if invalid.any():
        dropped = int(invalid.sum())
        df = df.loc[~invalid].copy()
        if (not QUIET_MODE) or VERBOSE:
            print(f"[DATA] Dropped {dropped} invalid rows (bad games_played/wins).")

    # Warn about out-of-range (don’t drop)
    if "win_percentage" in df.columns and df["win_percentage"].notna().any():
        wmin = float(df["win_percentage"].min())
        wmax = float(df["win_percentage"].max())
        if wmin < 0 or wmax > 1.0:
            if (not QUIET_MODE) or VERBOSE:
                print(f"[DATA] Warning: win_percentage outside [0,1] (min={wmin:.3f}, max={wmax:.3f}).")

    # Lockout/short seasons are OK; just note if present
    if "games_played" in df.columns:
        short = df[df["games_played"] < 70]
        if short.shape[0] > 0:
            years = []
            if "season_start" in df.columns and short["season_start"].notna().any():
                years = sorted(short["season_start"].astype(int).unique().tolist())
            if (not QUIET_MODE) or VERBOSE:
                print(f"[DATA] Note: {short.shape[0]} rows have games_played < 70 (lockout/short seasons possible).")
            if years:
                if (not QUIET_MODE) or VERBOSE:
                    print(f"[DATA] Short-season years (sample): {years[:10]}")

    after = df.shape[0]
    if before != after:
        if (not QUIET_MODE) or VERBOSE:
            print(f"[DATA] Rows: {before} -> {after}")

    return df


# DATA MERGING FUNCTIONS

def attach_payroll(df: pd.DataFrame, payroll_path: str) -> pd.DataFrame:
    """Merge team payroll (by team-season) onto the main dataframe.

    Notes:
      - Handles common team-name variants and a few franchise relocations.
      - Computes payroll_z_by_season (within-season z-score) for era-adjusted spending.
    """
    # Try to load payroll file - handle missing file gracefully
    try:
        pay = pd.read_csv(payroll_path)
    except FileNotFoundError:
        # If file not found, try adding .csv extension
        if not payroll_path.lower().endswith(".csv"):
            try:
                pay = pd.read_csv(payroll_path + ".csv")
            except FileNotFoundError:
                return df  # Return original dataframe if file can't be found
        else:
            return df

    cols_lower = {c.lower(): c for c in pay.columns}
    pick_col = lambda candidates: next((cols_lower.get(c.lower()) for c in candidates if c.lower() in cols_lower), None)

    # Find team, season, and payroll columns (try multiple common names)
    team_col = pick_col(["team", "franchise", "tm", "Team"])
    season_col = pick_col(["seasonstartyear", "season_start", "season", "year", "Season", "Year"])
    payroll_col = next((c for c in pay.columns if "payroll" in c.lower() or "salary" in c.lower()), None)

    if not all([team_col, season_col, payroll_col]):
        return df

    pay = pay.rename(columns={team_col: "Team", season_col: "season_raw", payroll_col: "payroll"})

    # Normalize payroll team labels to match the main dataset.
    team_map = {
        "Cleveland": "Cleveland Cavaliers", "New York": "New York Knicks", "Detroit": "Detroit Pistons",
        "LA Lakers": "Los Angeles Lakers", "Atlanta": "Atlanta Hawks", "Dallas": "Dallas Mavericks",
        "Philadelphia": "Philadelphia 76ers", "Milwaukee": "Milwaukee Bucks", "Phoenix": "Phoenix Suns",
        "Brooklyn": "Brooklyn Nets", "Boston": "Boston Celtics", "Portland": "Portland Trail Blazers",
        "Golden State": "Golden State Warriors", "San Antonio": "San Antonio Spurs", "Indiana": "Indiana Pacers",
        "Utah": "Utah Jazz", "Oklahoma City": "Oklahoma City Thunder", "Houston": "Houston Rockets",
        "Charlotte": "Charlotte Hornets", "Denver": "Denver Nuggets", "LA Clippers": "Los Angeles Clippers",
        "Chicago": "Chicago Bulls", "Washington": "Washington Wizards", "Sacramento": "Sacramento Kings",
        "Miami": "Miami Heat", "Minnesota": "Minnesota Timberwolves", "Orlando": "Orlando Magic",
        "Memphis": "Memphis Grizzlies", "Toronto": "Toronto Raptors", "New Orleans": "New Orleans Pelicans",
    }
    pay["Team"] = pay["Team"].replace(team_map)

    # Parse season and clean payroll
    pay["season_start"] = pay["season_raw"].astype(str).apply(season_start_year)
    pay["payroll"] = pay["payroll"].astype(str).str.replace("$", "", regex=False).str.replace(",", "",
                                                                                              regex=False).astype(float)

    pay_small = pay[["Team", "season_start", "payroll"]].dropna(subset=["Team", "season_start"])
    pay_small = pay_small.groupby(["Team", "season_start"], as_index=False)["payroll"].mean()

    # Normalize franchise names for merge
    franchise_map = {
        "LA Clippers": "Los Angeles Clippers", "New Jersey Nets": "Brooklyn Nets",
        "Charlotte Bobcats": "Charlotte Hornets", "New Orleans Hornets": "New Orleans Pelicans",
        "New Orleans/Oklahoma City Hornets": "New Orleans Pelicans",
        "Seattle SuperSonics": "Oklahoma City Thunder", "Vancouver Grizzlies": "Memphis Grizzlies",
    }

    df_merge = df.copy()
    df_merge["Team_norm"] = df_merge["Team"].replace(franchise_map)
    pay_merge = pay_small.copy()
    pay_merge["Team_norm"] = pay_merge["Team"].replace(franchise_map)

    # Merge payroll data with main dataset using normalized team names
    out = df_merge.merge(pay_merge[["Team_norm", "season_start", "payroll"]], on=["Team_norm", "season_start"],
                         how="left")
    out = out.drop(columns=["Team_norm"])  # Remove temporary normalization column
    out["payroll_available"] = out["payroll"].notna()  # Flag for which rows have payroll data

    # Create derived metrics from payroll
    mask = out["payroll_available"] & out["wins"].notna() & (out["wins"] > 0)
    out["payroll_per_win"] = pd.NA
    out.loc[mask, "payroll_per_win"] = out.loc[mask, "payroll"] / out.loc[mask, "wins"].astype(float)

    # payroll_z_by_season: z-score normalized within each season
    m = out["payroll_available"]
    out["payroll_z_by_season"] = pd.NA
    if m.any():
        grp = out.loc[m].groupby("season_start")["payroll"]
        out.loc[m, "payroll_z_by_season"] = (out.loc[m, "payroll"] - grp.transform("mean")) / grp.transform("std")

    return out



# FEATURE ENGINEERING


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer pace/era-robust efficiency features (eFG%, per-possession rates, shares)."""
    df = df.copy()
    fga = df["field_goals_attempted"].replace(0, pd.NA)
    reb = df["rebounds"].replace(0, pd.NA)
    df["efg"] = (df["field_goals_made"] + 0.5 * df["three_pointers_made"]) / fga
    df["threepa_rate"] = df["three_pointers_attempted"] / fga
    df["fta_rate"] = df["free_throw_attempted"] / fga
    df["possessions"] = 0.96 * (df["field_goals_attempted"] + df["turnovers"] + 0.44 * df["free_throw_attempted"] - df[
        "offensive_rebounds"])
    poss = df["possessions"].replace(0, pd.NA)
    df["tov_rate"] = df["turnovers"] / poss  # Turnover rate (lower is better)
    df["ast_rate"] = df["assists"] / poss
    df["oreb_share"] = df["offensive_rebounds"] / reb
    df["stl_rate"] = df["steals"] / poss
    df["blk_rate"] = df["blocks"] / poss
    df["pf_rate"] = df["personal_fouls"] / poss
    return df


def attach_playoffs(df: pd.DataFrame, playoffs_path: str) -> pd.DataFrame:
    """Merge playoff outcomes onto regular-season rows (made_playoffs, champion, po_wins)."""
    try:
        po = pd.read_csv(playoffs_path)
    except FileNotFoundError:
        return df  # Return unchanged if file missing

    # Handle case variations in column names
    if "Team" not in po.columns and "team" in po.columns:
        po = po.rename(columns={"team": "Team"})

    # Check required columns exist
    required = {"Team", "season", "games_played", "wins"}
    if not required.issubset(set(po.columns)):
        return df

    # Parse season and prepare for merge
    po["season_start"] = po["season"].astype(str).apply(season_start_year)
    po_max_season = int(po["season_start"].max())  # Track coverage for later

    # Rename columns to avoid conflicts with regular season columns
    po_small = po[["Team", "season_start", "games_played", "wins"]].rename(
        columns={"games_played": "po_games", "wins": "po_wins"})

    # Merge playoff data with regular season data
    out = df.merge(po_small, on=["Team", "season_start"], how="left")
    out["po_data_available"] = out["season_start"] <= po_max_season  # Flag for data coverage

    # Handle missing values intelligently:
    # - If playoff data exists for that season but team has NA, they missed playoffs (fill with 0)
    # - If playoff data doesn't exist for that season, keep as NA (unknown)
    mask = out["po_data_available"]
    out.loc[mask, "po_games"] = out.loc[mask, "po_games"].fillna(0)
    out.loc[mask, "po_wins"] = out.loc[mask, "po_wins"].fillna(0)
    out.loc[~mask, ["po_games", "po_wins"]] = pd.NA  # Keep as NA for seasons without data

    # Create binary flags
    out["made_playoffs"] = pd.NA
    out.loc[mask, "made_playoffs"] = (out.loc[mask, "po_games"].astype(float) > 0).astype(int)

    # Champion = team with most playoff wins in that season
    out["champion"] = pd.NA
    season_max = out.loc[mask].groupby("season_start")["po_wins"].transform("max")
    out.loc[mask, "champion"] = ((out.loc[mask, "po_wins"] == season_max) & (out.loc[mask, "po_wins"] > 0)).astype(int)

    return out



# MODELING FUNCTIONS


def fit_model(X_train, y_train, X_test, y_test, feature_names):
    """Fit a standardized Ridge model and return (MAE, coefficients).

    Ridge (L2) is used because team stats are correlated; standardization makes coefficients comparable.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Fill missing values
        ("scaler", StandardScaler()),  # Normalize features
        ("model", Ridge(alpha=1.0)),  # Ridge regression with L2 penalty
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Extract and sort coefficients by absolute value (most important features first)
    coefs = pd.Series(pipe.named_steps["model"].coef_, index=feature_names).sort_values(key=lambda s: abs(s),
                                                                                        ascending=False)
    return mae, coefs



# MAIN ANALYSIS

def main():
    # Data loading
    # Load main dataset (regular season team statistics)
    df = pd.read_csv(CSV_PATH)

    # Output folder for tables/metrics (keeps results reproducible for the repo)
    out_dir = "outputs"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    def log(msg: str, force: bool = False):
        """Print only when not in quiet mode, unless forced."""
        if force or (not QUIET_MODE) or VERBOSE:
            print(msg)

    # Create point differential per game (more stable than win% for prediction)
    # Plus-minus is season total; divide by games to get per-game average
    df["plus_minus_per_game"] = df["plus_minus"] / df["games_played"].replace(0, pd.NA)

    if (not QUIET_MODE) or VERBOSE:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
    log(f"\nDataset: {df.shape[0]} team-seasons, {df.shape[1]} columns", force=True)

    # Normalize percentage columns - some datasets use 0-100, others use 0-1
    # If max value > 1.5, assume it's in 0-100 format and convert to 0-1
    pct_cols = ["field_goal_percentage", "three_point_percentage", "free_throw_percentage"]
    for col in pct_cols:
        if col in df.columns and df[col].notna().any():
            # If max value > 1.5, assume it's in 0–100 format and convert to 0–1.
            if df[col].max() > 1.5:
                df[col] = df[col] / 100.0

    # Extract season start year for time-based analysis
    df["season_start"] = df["season"].astype(str).apply(season_start_year)

    # Data hygiene: types + basic validity checks
    df = validate_and_standardize(df)

    # Engineer rate/efficiency features (eFG%, turnover rate, etc.)
    df = add_features(df)

    # Merge optional datasets
    if USE_PLAYOFFS:
        df = attach_playoffs(df, PLAYOFFS_CSV_PATH)
        if "po_data_available" in df.columns:
            po_cov = df[df["po_data_available"]]
            if po_cov.shape[0] > 0:
                log(
                    f"[COVERAGE] Playoffs available: {po_cov['season_start'].min():.0f}–{po_cov['season_start'].max():.0f} "
                    f"({po_cov.shape[0]} team-seasons).",
                    force=True,
                )
    # Track payroll coverage for later (payroll data may not cover all seasons)
    payroll_max_season = None
    if USE_PAYROLL:
        df = attach_payroll(df, PAYROLL_CSV_PATH)
        if "payroll_available" in df.columns and df["payroll_available"].any():
            payroll_max_season = int(df.loc[df["payroll_available"], "season_start"].max())
            if VERBOSE:
                print(
                    f"\nPayroll data: {int(df['payroll_available'].sum())} team-seasons matched (seasons up to {payroll_max_season})"
                )
        else:
            if VERBOSE:
                print("\nPayroll data: No payroll data available")

    if USE_PAYROLL and "payroll_available" in df.columns and df["payroll_available"].any():
        pay_cov = df[df["payroll_available"]]
        log(
            f"[COVERAGE] Payroll available: {pay_cov['season_start'].min():.0f}–{pay_cov['season_start'].max():.0f} "
            f"({int(pay_cov.shape[0])} team-seasons).",
            force=True,
        )

    # Export one enriched dataset
    if SAVE_ENRICHED_CSV:
        enriched_path = os.path.join(out_dir, ENRICHED_CSV_NAME)
        df.to_csv(enriched_path, index=False)
        log(f"[OUTPUT] Wrote enriched dataset: {enriched_path} ({df.shape[0]} rows, {df.shape[1]} cols)", force=True)

    # Data validation - quick sanity checks
    log(f"\nWin% range: {df['win_percentage'].min():.1%} to {df['win_percentage'].max():.1%}")
    log(f"eFG% range: {df['efg'].min():.1%} to {df['efg'].max():.1%}")
    log(f"Duplicate team-seasons: {df.duplicated(subset=['Team', 'season']).sum()}")

    # Train/test split: time-based to mimic forecasting (no future seasons in training).
    train = df[df["season_start"] <= 2017].copy()
    test = df[df["season_start"] >= 2018].copy()
    # Feature definition
    # Define feature groups for model comparison
    offense_cols = ["efg", "tov_rate", "threepa_rate", "fta_rate", "oreb_share", "ast_rate"]

    # Defense features: defensive event rates (steals, blocks, fouls)
    # Note: These are proxies - true defense requires opponent stats
    defense_cols = ["stl_rate", "blk_rate", "pf_rate"]

    # Combined: all features together
    combined_cols = offense_cols + defense_cols

    # Payroll feature: z-score normalized by season (adjusts for inflation)
    payroll_feat = ["payroll_z_by_season"] if "payroll_z_by_season" in df.columns else []

    # Modeling
    log("\n--- Win% prediction ---")
    # Target variable: win percentage (0–100). Reporting MAE in percentage points.
    y_train = (train["win_percentage"].astype(float) * 100.0)
    y_test = (test["win_percentage"].astype(float) * 100.0)

    # Compare three model types: offense-only, defense-only, and combined
    # This answers: "Which type of stats better predicts wins?"
    results = {}
    for name, cols in [("Offense", offense_cols), ("Defense", defense_cols), ("Combined", combined_cols)]:
        mae, coefs = fit_model(train[cols], y_train, test[cols], y_test, cols)
        results[name] = mae
        log(f"{name:15s} -> MAE: {mae:.2f} percentage points")

    # Display models sorted by performance (best first)
    log("\nModel comparison (lower MAE is better):")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        log(f"  {name:15s}: {mae:.2f} pp")


    # Payroll experiment: only evaluate seasons where payroll exists (otherwise we'd be imputing whole years).
    payroll_results = {}
    if payroll_feat and payroll_max_season:
        train_pay = train[train["season_start"] <= payroll_max_season].copy()
        test_pay = test[test["season_start"] <= payroll_max_season].copy()
        if train_pay.shape[0] > 0 and test_pay.shape[0] > 0:
            # Check correlation: Does spending more lead to more wins?
            if "payroll_z_by_season" in df.columns:
                corr_df = df[["win_percentage", "payroll_z_by_season"]].apply(pd.to_numeric, errors="coerce").dropna()
                if corr_df.shape[0] >= 3:
                    corr_pay = corr_df.corr().iloc[0, 1]
                    log(f"\nCorrelation: win% vs payroll_z_by_season = {corr_pay:.3f}")

            # Add payroll to offense and combined models
            for name, cols in [("Offense+Payroll", offense_cols + payroll_feat),
                               ("Combined+Payroll", combined_cols + payroll_feat)]:
                mae, _ = fit_model(
                    train_pay[cols], train_pay["win_percentage"].astype(float) * 100.0,
                    test_pay[cols], test_pay["win_percentage"].astype(float) * 100.0,
                    cols,
                )
                payroll_results[name] = mae
                log(f"{name:20s} (restricted) -> MAE: {mae:.2f} percentage points")

            # Highlight best payroll model
            if payroll_results:
                best_payroll = min(payroll_results.items(), key=lambda x: x[1])
                log(f"\nBest model with payroll: {best_payroll[0]} (MAE: {best_payroll[1]:.2f} pp)")

    # Payroll value analysis: over/under-performance after accounting for spending AND team stats.
    # Residuals here are closer to "front office value" than a payroll-only baseline.
    if USE_PAYROLL and "payroll_available" in df.columns and df["payroll_available"].any() and "payroll_z_by_season" in df.columns:
        pay_df = df[df["payroll_available"]].copy()
        pay_df = pay_df.dropna(subset=["win_percentage", "payroll_z_by_season"]).copy()

        # Model: win% (pp) ~ payroll_z + on-court features (combined)
        value_cols = combined_cols + ["payroll_z_by_season"]
        value_cols = [c for c in value_cols if c in pay_df.columns]

        if pay_df.shape[0] >= 10 and len(value_cols) >= 2:
            pay_df["win_pct_pp"] = pay_df["win_percentage"].astype(float) * 100.0
            X = pay_df[value_cols].astype(float)
            y = pay_df["win_pct_pp"].astype(float)

            value_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ])
            value_pipe.fit(X, y)
            pay_df["win_pct_expected"] = value_pipe.predict(X)
            pay_df["win_pct_residual"] = pay_df["win_pct_pp"] - pay_df["win_pct_expected"]

            log("\n--- Payroll value (net of team stats) ---")
            log("Residual = actual win% (pp) − expected win% (pp) given payroll + efficiency stats")

            cols_show = ["Team", "season_start", "win_pct_pp", "payroll_z_by_season", "win_pct_residual"]

            top_value = pay_df.sort_values("win_pct_residual", ascending=False).head(10)[cols_show]
            low_value = pay_df.sort_values("win_pct_residual", ascending=True).head(10)[cols_show]

            log("\nTop 10 value team-seasons (won more than payroll+stats predicts):")
            log(top_value.to_string(index=False, formatters={
                "win_pct_pp": "{:.1f}".format,
                "payroll_z_by_season": "{:.2f}".format,
                "win_pct_residual": "{:+.1f}".format,
            }))

            log("\nBottom 10 value team-seasons (won less than payroll+stats predicts):")
            log(low_value.to_string(index=False, formatters={
                "win_pct_pp": "{:.1f}".format,
                "payroll_z_by_season": "{:.2f}".format,
                "win_pct_residual": "{:+.1f}".format,
            }))

            # Most recent season with payroll coverage
            if payroll_max_season is not None:
                recent = pay_df[pay_df["season_start"] == payroll_max_season].copy()
                if recent.shape[0] > 0:
                    log(f"\nMost recent payroll season ({payroll_max_season}) value leaders:")
                    recent_top = recent.sort_values("win_pct_residual", ascending=False).head(5)[cols_show]
                    recent_low = recent.sort_values("win_pct_residual", ascending=True).head(5)[cols_show]

                    log("\nBest value (top 5):")
                    log(recent_top.to_string(index=False, formatters={
                        "win_pct_pp": "{:.1f}".format,
                        "payroll_z_by_season": "{:.2f}".format,
                        "win_pct_residual": "{:+.1f}".format,
                    }))

                    log("\nWorst value (bottom 5):")
                    log(recent_low.to_string(index=False, formatters={
                        "win_pct_pp": "{:.1f}".format,
                        "payroll_z_by_season": "{:.2f}".format,
                        "win_pct_residual": "{:+.1f}".format,
                    }))

    # Era split: compare model performance and coefficients across periods.
    # Two eras: pre-3pt boom (<=2014) vs modern (>=2015).
    log("\n--- Era split (model stability) ---")

    def _fit_and_report(label, df_train, df_test, cols, use_pp=True):
        if df_train.shape[0] < 10 or df_test.shape[0] < 10:
            log(f"{label}: not enough data")
            return None, None
        if use_pp:
            y_tr = df_train["win_percentage"].astype(float) * 100.0
            y_te = df_test["win_percentage"].astype(float) * 100.0
        else:
            y_tr = df_train["win_percentage"].astype(float)
            y_te = df_test["win_percentage"].astype(float)
        mae, coefs = fit_model(df_train[cols], y_tr, df_test[cols], y_te, cols)
        log(f"{label} -> MAE: {mae:.2f} pp")
        log("  Top coefficients:")
        for k, v in coefs.head(4).items():
            log(f"    {k:12s} {v:+.3f}")
        return mae, coefs

    # Pre-2015: train 2000–2010, test 2011–2014
    pre = df[df["season_start"] <= 2014].copy()
    pre_train = pre[pre["season_start"] <= 2010].copy()
    pre_test = pre[(pre["season_start"] >= 2011) & (pre["season_start"] <= 2014)].copy()
    pre_mae, pre_coefs = _fit_and_report("Pre-2015 (2000–2014)", pre_train, pre_test, combined_cols)

    # Modern: train 2015–2019, test 2020–2023
    modern = df[df["season_start"] >= 2015].copy()
    modern_train = modern[(modern["season_start"] >= 2015) & (modern["season_start"] <= 2019)].copy()
    modern_test = modern[(modern["season_start"] >= 2020) & (modern["season_start"] <= 2023)].copy()
    modern_mae, modern_coefs = _fit_and_report("Modern (2015–2023)", modern_train, modern_test, combined_cols)

    # Which features changed most across eras?
    if pre_coefs is not None and modern_coefs is not None:
        delta = (modern_coefs - pre_coefs).sort_values(key=lambda s: abs(s), ascending=False)
        delta_df = pd.DataFrame({
            "feature": delta.index,
            "coef_pre": pre_coefs.reindex(delta.index).values,
            "coef_modern": modern_coefs.reindex(delta.index).values,
            "delta_modern_minus_pre": delta.values,
        })

        log("\nBiggest coefficient shifts (modern − pre):")
        log(delta_df.head(8).to_string(index=False, formatters={
            "coef_pre": "{:+.3f}".format,
            "coef_modern": "{:+.3f}".format,
            "delta_modern_minus_pre": "{:+.3f}".format,
        }))

    # If payroll exists, compare modern era with payroll on the payroll-covered window.
    if payroll_feat and payroll_max_season is not None:
        modern_pay = df[(df["season_start"] >= 2015) & (df["season_start"] <= payroll_max_season) & (df["payroll_available"])].copy()
        pay_train = modern_pay[(modern_pay["season_start"] >= 2015) & (modern_pay["season_start"] <= min(2018, payroll_max_season))].copy()
        pay_test = modern_pay[(modern_pay["season_start"] >= max(2019, 2015)) & (modern_pay["season_start"] <= payroll_max_season)].copy()
        if pay_train.shape[0] >= 10 and pay_test.shape[0] >= 10:
            _fit_and_report("Modern w/ payroll (restricted)", pay_train, pay_test, combined_cols + payroll_feat)

    # Point differential prediction
    log("\n--- Point differential prediction ---")
    # Target: point differential per game (can be negative or positive)
    y_train_pm = train["plus_minus_per_game"].astype(float)
    y_test_pm = test["plus_minus_per_game"].astype(float)

    # Baseline: predict the mean (simple benchmark)
    baseline_mae = float((y_test_pm - y_test_pm.mean()).abs().mean())
    log(f"Baseline MAE (predict mean): {baseline_mae:.3f}")

    # Test same models on point differential
    pm_results = {}
    for name, cols in [("Offense", offense_cols), ("Defense", defense_cols), ("Combined", combined_cols)]:
        mae, _ = fit_model(train[cols], y_train_pm, test[cols], y_test_pm, cols)
        pm_results[name] = mae
        log(f"{name:15s} -> MAE: {mae:.3f} points/game")

    # Compare to baseline
    best_pm = min(pm_results.items(), key=lambda x: x[1])
    improvement = ((baseline_mae - best_pm[1]) / baseline_mae) * 100
    log(f"\nBest model: {best_pm[0]} (MAE: {best_pm[1]:.3f}) - {improvement:.1f}% better than baseline")

    # Test payroll models on point differential
    if payroll_feat and payroll_max_season:
        train_pay = train[train["season_start"] <= payroll_max_season].copy()
        test_pay = test[test["season_start"] <= payroll_max_season].copy()
        if train_pay.shape[0] > 0 and test_pay.shape[0] > 0:
            for name, cols in [("Offense+Payroll", offense_cols + payroll_feat),
                               ("Combined+Payroll", combined_cols + payroll_feat)]:
                mae, _ = fit_model(
                    train_pay[cols], train_pay["plus_minus_per_game"].astype(float),
                    test_pay[cols], test_pay["plus_minus_per_game"].astype(float), cols
                )
                log(f"{name} (restricted) -> MAE: {mae:.3f}")

    # Playoff analysis
    if "po_data_available" in df.columns:
        log("\n--- Playoff analysis ---")

        # Filter to seasons with playoff data
        df_po = df[df["po_data_available"]].copy()
        log(f"Playoff data available for {df_po.shape[0]} team-seasons")
        log(f"Made playoffs: {int(df_po['made_playoffs'].sum())} team-seasons")
        log(f"Champions: {int(df_po['champion'].sum())} team-seasons")

        # Compare regular-season stats: playoff teams vs non-playoff teams
        # This shows what separates successful teams
        cols = offense_cols + ["win_percentage"]
        log("\nRegular-season averages:")
        playoff_comp = df_po.groupby("made_playoffs")[cols].mean().round(3)
        playoff_comp.index = ["Non-playoff", "Playoff"]
        log(playoff_comp)

        # Compare champions vs non-champions
        # What makes a championship team different?
        log("\nChampion comparison:")
        champ_comp = df_po.groupby("champion")[cols].mean().round(3)
        champ_comp.index = ["Non-champion", "Champion"]
        log(champ_comp)

        # Check if regular-season success predicts playoff success
        corr_po = df_po[["win_percentage", "po_wins"]].corr(numeric_only=True).iloc[0, 1]
        log(f"\nCorrelation: regular-season win% vs playoff wins = {corr_po:.3f}")

        # Can we predict playoff wins from regular-season stats?
        # Split at 2013/2014 for time-based validation
        df_po_model = df_po.dropna(subset=["po_wins"]).copy()
        po_train = df_po_model[df_po_model["season_start"] <= 2013].copy()
        po_test = df_po_model[df_po_model["season_start"] >= 2014].copy()

        if po_train.shape[0] > 0 and po_test.shape[0] > 0:
            po_features = offense_cols + ["win_percentage"]
            mae, coefs = fit_model(
                po_train[po_features], po_train["po_wins"].astype(float),
                po_test[po_features], po_test["po_wins"].astype(float), po_features
            )
            log(f"\nPlayoff wins prediction MAE: {mae:.2f}")

    # League trends & exploratory analysis
    log("\n--- League trends ---")

    # Calculate league-wide averages by season
    # Shows how the game has evolved (e.g., more 3-pointers over time)
    league = df.groupby("season_start", as_index=False).agg(
        efg_mean=("efg", "mean"), threepa_rate_mean=("threepa_rate", "mean"),
        tov_rate_mean=("tov_rate", "mean"), ast_rate_mean=("ast_rate", "mean"),
        fta_rate_mean=("fta_rate", "mean")
    ).sort_values("season_start")

    log("\nLast 5 seasons:")
    log(league.tail(5)[["season_start", "efg_mean", "threepa_rate_mean"]].to_string(index=False))

    # Feature importance: Which stats correlate most with winning?
    # Higher correlation = stronger relationship with win percentage
    feature_cols = offense_cols + ["win_percentage"]
    corr = df[feature_cols].corr(numeric_only=True)["win_percentage"].sort_values(ascending=False)
    log("\nFeature correlations with win%:")
    for feat, val in corr.items():
        if feat != "win_percentage":
            log(f"  {feat:15s}: {val:6.3f}")

    # Visualizations (kept lightweight; no custom styling beyond a safe default)
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("default")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. 3PA Rate Over Time
    axes[0, 0].plot(league["season_start"], league["threepa_rate_mean"], linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title("League 3-Point Attempt Rate Over Time", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Season Start Year")
    axes[0, 0].set_ylabel("3PA / FGA")
    axes[0, 0].grid(True, alpha=0.3)
    # 2. eFG% Over Time
    axes[0, 1].plot(league["season_start"], league["efg_mean"], linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_title("League Effective Field Goal % Over Time", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Season Start Year")
    axes[0, 1].set_ylabel("eFG%")
    axes[0, 1].grid(True, alpha=0.3)
    # 3. Feature Correlations
    corr_data = corr[corr.index != "win_percentage"].sort_values(ascending=True)
    axes[1, 0].barh(corr_data.index, corr_data.values, alpha=0.7)
    axes[1, 0].set_title("Feature Correlations with Win%", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Correlation Coefficient")
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    # 4. Model Performance Comparison
    model_names = list(results.keys())
    model_maes = [results[m] for m in model_names]
    axes[1, 1].bar(model_names, model_maes, alpha=0.7)
    axes[1, 1].set_title("Model Performance (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Mean Absolute Error")
    axes[1, 1].set_ylim(0, max(model_maes) * 1.2)
    for i, (name, mae) in enumerate(zip(model_names, model_maes)):
        axes[1, 1].text(i, mae + max(model_maes) * 0.02, f'{mae:.4f}', ha='center', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if SAVE_PLOTS:
        fig_path = os.path.join(fig_dir, "league_trends_and_models.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        log(f"[OUTPUT] Saved figure: {fig_path}", force=True)

    if SHOW_PLOTS and ((not QUIET_MODE) or VERBOSE):
        plt.show()
    else:
        plt.close(fig)

    # Key findings summary
    log("\n--- Key findings ---")
    best_feature = corr[corr.index != "win_percentage"].idxmax()
    best_corr = corr[corr.index != "win_percentage"].max()
    log(f"1. Best predictor of win%: {best_feature} (r={best_corr:.3f})")
    best_model_name = min(results.items(), key=lambda x: x[1])[0]
    best_model_mae = min(results.values())
    log(f"2. Best model for win%: {best_model_name} (MAE: {best_model_mae:.2f} pp)")

    # Save a compact metrics table for the repo
    # (Removed CSV output)

    if payroll_results:
        best_payroll = min(payroll_results.items(), key=lambda x: x[1])
        improvement = ((results["Combined"] - best_payroll[1]) / results["Combined"] * 100.0)
        log(f"3. Payroll improves prediction: {best_payroll[0]} reduces MAE by {improvement:.1f}%")
    if "po_data_available" in df.columns:
        df_po = df[df["po_data_available"]].copy()
        corr_po = df_po[["win_percentage", "po_wins"]].corr(numeric_only=True).iloc[0, 1]
        log(f"4. Regular-season win% predicts playoff success: r={corr_po:.3f}")
    log(
        f"5. League trends: 3PA rate increased from {league['threepa_rate_mean'].iloc[0]:.3f} to {league['threepa_rate_mean'].iloc[-1]:.3f}")
    # (trailing divider removed)


if __name__ == "__main__":
    main()
