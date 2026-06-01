"""
NBA team stats project script.
I use this to clean the CSVs, build a few rate stats, and run quick models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# config
CSV_PATH = "data/raw/nba_team_stats_00_to_23.csv"
PLAYOFFS_CSV_PATH = "data/raw/nba_team_stats_playoffs_00_to_21.csv"
PAYROLL_CSV_PATH = "data/raw/nba_team_payroll_1990_to_2023.csv"
USE_PLAYOFFS = True
USE_PAYROLL = True
QUIET_MODE = True
VERBOSE = True
SAVE_ENRICHED_CSV = True
ENRICHED_CSV_NAME = "nba_team_seasons_enriched.csv"
SAVE_PLOTS = True
SHOW_PLOTS = False


# helpers


def season_start_year(s):
    return int(str(s).split("-")[0])


# data cleanup

def validate_and_standardize(df):
    df = df.copy()

    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).str.strip()

    cols = [
        "wins",
        "games_played",
        "plus_minus",
        "field_goals_attempted",
        "field_goals_made",
        "three_pointers_attempted",
        "three_pointers_made",
        "free_throw_attempted",
        "turnovers",
        "assists",
        "rebounds",
        "offensive_rebounds",
        "steals",
        "blocks",
        "personal_fouls",
        "win_percentage",
    ]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    before = df.shape[0]
    invalid = (df["wins"] < 0) | (df["wins"] > df["games_played"])
    if invalid.any():
        df = df[~invalid].copy()
    after = df.shape[0]
    if before > after:
        if (not QUIET_MODE) or VERBOSE:
            print(f"[DATA] Dropped {before - after} bad rows")

    if "win_percentage" in df.columns and df["win_percentage"].max() > 1:
        wmax = float(df["win_percentage"].max())
        if (not QUIET_MODE) or VERBOSE:
            print(f"[DATA] Warning: win_percentage max is {wmax}")

    return df


# optional merges (payroll + playoffs)

def attach_payroll(df, payroll_path):
    try:
        pay = pd.read_csv(payroll_path, index_col=0)
    except Exception:
        return df
    team_col = season_col = payroll_col = None
    for col in pay.columns:
        if col.lower() in ['team', 'franchise']:
            team_col = col
        elif "season" in col.lower() or col.lower() == "year":
            season_col = col
        elif 'payroll' in col.lower() or 'salary' in col.lower():
            payroll_col = col
    if not (team_col and season_col and payroll_col):
        return df
    pay = pay[[team_col, season_col, payroll_col]].copy()
    pay.columns = ["Team", "season_start", "payroll"]
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
    pay["season_start"] = pay["season_start"].astype(str).apply(season_start_year)
    pay["payroll"] = (
        pay["payroll"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    pay_small = pay[["Team", "season_start", "payroll"]].dropna(subset=["Team", "season_start"])
    pay_small = pay_small.groupby(["Team", "season_start"], as_index=False)["payroll"].mean()
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
    out = df_merge.merge(pay_merge[["Team_norm", "season_start", "payroll"]], on=["Team_norm", "season_start"], how="left")
    out = out.drop(columns=["Team_norm"])
    out["payroll_available"] = out["payroll"].notna()
    out["payroll_z_by_season"] = 0.0
    for season in out["season_start"].unique():
        season_payroll = out[out["season_start"] == season]["payroll"]
        mean = season_payroll.mean()
        std = season_payroll.std()
        if std > 0:
            z_scores = (season_payroll - mean) / std
            out.loc[out["season_start"] == season, "payroll_z_by_season"] = z_scores
    return out



# feature engineering


def add_features(df):
    df = df.copy()
    df["efg"] = (df["field_goals_made"] + 0.5 * df["three_pointers_made"]) / df["field_goals_attempted"]
    df["threepa_rate"] = df["three_pointers_attempted"] / df["field_goals_attempted"]
    df["fta_rate"] = df["free_throw_attempted"] / df["field_goals_attempted"]
    df["possessions"] = 0.96 * (df["field_goals_attempted"] + df["turnovers"] + 0.44 * df["free_throw_attempted"] - df["offensive_rebounds"])
    df["tov_rate"] = df["turnovers"] / df["possessions"]
    df["ast_rate"] = df["assists"] / df["possessions"]
    df["oreb_share"] = df["offensive_rebounds"] / df["rebounds"]
    df["stl_rate"] = df["steals"] / df["possessions"]
    df["blk_rate"] = df["blocks"] / df["possessions"]
    df["pf_rate"] = df["personal_fouls"] / df["possessions"]
    return df


def attach_playoffs(df, playoffs_path):
    try:
        po = pd.read_csv(playoffs_path)
    except Exception:
        return df

    if "Team" not in po.columns:
        po = po.rename(columns={"team": "Team"})
    
    po["season_start"] = po["season"].astype(str).apply(season_start_year)
    po_seasons = set(po["season_start"].unique())
    po = po[["Team", "season_start", "games_played", "wins"]]
    po.columns = ["Team", "season_start", "po_games", "po_wins"]

    out = df.merge(po, on=["Team", "season_start"], how="left")
    # flag seasons the playoff table actually covers, before we fill non-qualifiers with 0
    out["po_data_available"] = out["season_start"].isin(po_seasons)
    out["po_games"] = out["po_games"].fillna(0)
    out["po_wins"] = out["po_wins"].fillna(0)
    out["made_playoffs"] = (out["po_games"] > 0).astype(int)
    
    for season in out["season_start"].unique():
        s_mask = out["season_start"] == season
        champ = out[s_mask]["po_wins"].max()
        out.loc[s_mask & (out["po_wins"] == champ) & (out["po_wins"] > 0), "champion"] = 1
    
    out["champion"] = out["champion"].fillna(0)
    return out



# modeling


def fit_model(X_train, y_train, X_test, y_test, feature_names):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    coefs = pd.Series(pipe.named_steps["model"].coef_, index=feature_names)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    return mae, coefs


# plots

def make_summary_figure(league, corr, results, fig_path, save_plots=True, show_plots=False):
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        # fallback if this style isn't installed
        pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(league["season_start"], league["threepa_rate_mean"], linewidth=2, marker="o")
    ax.set_title("League 3PA Rate")
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(league["season_start"], league["efg_mean"], linewidth=2, marker="o")
    ax.set_title("League eFG%")
    ax.grid(True)

    ax = axes[1, 0]
    c = corr[corr.index != "win_percentage"].sort_values()
    ax.barh(c.index, c.values)
    ax.set_title("Correlations")
    ax.axvline(x=0, color="black", linestyle="--")

    ax = axes[1, 1]
    names = list(results.keys())
    vals = [results[m] for m in names]
    ax.bar(names, vals)
    ax.set_title("Model MAE")

    plt.tight_layout()

    if save_plots:
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)



# main flow

def main():
    df = pd.read_csv(CSV_PATH)
    out_dir = "outputs"
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    def log(msg, force=False):
        if force or VERBOSE or not QUIET_MODE:
            print(msg)

    df["plus_minus_per_game"] = df["plus_minus"] / df["games_played"]

    if (not QUIET_MODE) or VERBOSE:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
    log(f"\nDataset: {df.shape[0]} team-seasons, {df.shape[1]} columns", force=True)
    log(f"Config: USE_PLAYOFFS={USE_PLAYOFFS}, USE_PAYROLL={USE_PAYROLL}, QUIET_MODE={QUIET_MODE}", force=True)

    for col in ["field_goal_percentage", "three_point_percentage", "free_throw_percentage"]:
        if col in df.columns and df[col].max() > 1.5:
            df[col] = df[col] / 100.0

    df["season_start"] = df["season"].astype(str).apply(season_start_year)
    df = validate_and_standardize(df)
    df = add_features(df)
    if USE_PLAYOFFS:
        df = attach_playoffs(df, PLAYOFFS_CSV_PATH)
        if "po_data_available" in df.columns:
            po_cov = df[df["po_data_available"]]
            if po_cov.shape[0] > 0:
                log(
                    f"[COVERAGE] Playoffs available: {po_cov['season_start'].min():.0f}-{po_cov['season_start'].max():.0f} "
                    f"({po_cov.shape[0]} team-seasons).",
                    force=True,
                )
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
            f"[COVERAGE] Payroll available: {pay_cov['season_start'].min():.0f}-{pay_cov['season_start'].max():.0f} "
            f"({int(pay_cov.shape[0])} team-seasons).",
            force=True,
        )

    df["win_pct_pp"] = df["win_percentage"].astype(float) * 100.0
    df["win_pct_expected"] = pd.NA
    df["win_pct_residual"] = pd.NA
    log(f"\nWin% range: {df['win_percentage'].min():.1%} to {df['win_percentage'].max():.1%}")
    log(f"eFG% range: {df['efg'].min():.1%} to {df['efg'].max():.1%}")
    log(f"Duplicate team-seasons: {df.duplicated(subset=['Team', 'season']).sum()}")

    # simple time split so we don't train on future seasons
    train = df[df["season_start"] <= 2017].copy()
    test = df[df["season_start"] >= 2018].copy()
    offense_cols = ["efg", "tov_rate", "threepa_rate", "fta_rate", "oreb_share", "ast_rate"]

    # not true defense (no opponent stats), but still useful signals
    defense_cols = ["stl_rate", "blk_rate", "pf_rate"]
    combined_cols = offense_cols + defense_cols
    payroll_feat = ["payroll_z_by_season"] if "payroll_z_by_season" in df.columns else []

    # win% models
    log("\n--- Win% prediction ---")
    y_train = (train["win_percentage"].astype(float) * 100.0)
    y_test = (test["win_percentage"].astype(float) * 100.0)

    # baseline: predict the training-mean win% for everyone (the bar real models must clear)
    win_baseline_mae = float((y_test - y_train.mean()).abs().mean())
    log(f"Baseline MAE (predict mean win%): {win_baseline_mae:.2f} pp")

    # compare a few feature sets
    results = {}
    for name, cols in [("Offense", offense_cols), ("Defense", defense_cols), ("Combined", combined_cols)]:
        mae, coefs = fit_model(train[cols], y_train, test[cols], y_test, cols)
        results[name] = mae
        lift = (win_baseline_mae - mae) / win_baseline_mae * 100.0
        log(f"{name:15s} -> MAE: {mae:.2f} pp ({lift:+.1f}% vs baseline {win_baseline_mae:.2f})")

    log("\nModel comparison (lower MAE is better):")
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        lift = (win_baseline_mae - mae) / win_baseline_mae * 100.0
        log(f"  {name:15s}: {mae:.2f} pp ({lift:+.1f}% vs baseline)")


    payroll_results = {}
    if payroll_feat and payroll_max_season:
        train_pay = train[train["season_start"] <= payroll_max_season].copy()
        test_pay = test[test["season_start"] <= payroll_max_season].copy()
        if train_pay.shape[0] > 0 and test_pay.shape[0] > 0:
            if "payroll_z_by_season" in df.columns:
                corr_df = df[["win_percentage", "payroll_z_by_season"]].apply(pd.to_numeric, errors="coerce").dropna()
                if corr_df.shape[0] >= 3:
                    corr_pay = corr_df.corr().iloc[0, 1]
                    log(f"\nCorrelation: win% vs payroll_z_by_season = {corr_pay:.3f}")

            for name, cols in [("Offense+Payroll", offense_cols + payroll_feat),
                               ("Combined+Payroll", combined_cols + payroll_feat)]:
                mae, _ = fit_model(
                    train_pay[cols], train_pay["win_percentage"].astype(float) * 100.0,
                    test_pay[cols], test_pay["win_percentage"].astype(float) * 100.0,
                    cols,
                )
                payroll_results[name] = mae
                log(f"{name:20s} (restricted) -> MAE: {mae:.2f} percentage points")

            if payroll_results:
                best_payroll = min(payroll_results.items(), key=lambda x: x[1])
                log(f"\nBest model with payroll: {best_payroll[0]} (MAE: {best_payroll[1]:.2f} pp)")

    if USE_PAYROLL and "payroll_available" in df.columns and df["payroll_available"].any() and "payroll_z_by_season" in df.columns:
        pay_df = df[df["payroll_available"]].copy()
        pay_df = pay_df.dropna(subset=["win_percentage", "payroll_z_by_season"]).copy()

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

            key_cols = ["Team", "season_start"]
            tmp = pay_df[key_cols + ["win_pct_expected", "win_pct_residual"]].copy()
            tmp = tmp.dropna(subset=["win_pct_expected", "win_pct_residual"])

            df_idx = df.set_index(key_cols)
            tmp_idx = tmp.set_index(key_cols)
            df_idx.loc[tmp_idx.index, "win_pct_expected"] = tmp_idx["win_pct_expected"]
            df_idx.loc[tmp_idx.index, "win_pct_residual"] = tmp_idx["win_pct_residual"]
            df = df_idx.reset_index()

            log("\n--- Payroll value view ---")
            log("Residual = actual win% (pp) - expected win% (pp) from payroll + team stats")

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

    log("\n--- Era split check ---")

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

    # pre-2015 split
    pre = df[df["season_start"] <= 2014].copy()
    pre_train = pre[pre["season_start"] <= 2010].copy()
    pre_test = pre[(pre["season_start"] >= 2011) & (pre["season_start"] <= 2014)].copy()
    pre_mae, pre_coefs = _fit_and_report("Pre-2015 (2000-2014)", pre_train, pre_test, combined_cols)

    # modern split
    modern = df[df["season_start"] >= 2015].copy()
    modern_train = modern[(modern["season_start"] >= 2015) & (modern["season_start"] <= 2019)].copy()
    modern_test = modern[(modern["season_start"] >= 2020) & (modern["season_start"] <= 2023)].copy()
    modern_mae, modern_coefs = _fit_and_report("Modern (2015-2023)", modern_train, modern_test, combined_cols)

    if pre_coefs is not None and modern_coefs is not None:
        delta = (modern_coefs - pre_coefs).sort_values(key=lambda s: abs(s), ascending=False)
        delta_df = pd.DataFrame({
            "feature": delta.index,
            "coef_pre": pre_coefs.reindex(delta.index).values,
            "coef_modern": modern_coefs.reindex(delta.index).values,
            "delta_modern_minus_pre": delta.values,
        })

        log("\nBiggest coefficient shifts (modern - pre):")
        log(delta_df.head(8).to_string(index=False, formatters={
            "coef_pre": "{:+.3f}".format,
            "coef_modern": "{:+.3f}".format,
            "delta_modern_minus_pre": "{:+.3f}".format,
        }))

    # if payroll data exists, run modern split with payroll too
    if payroll_feat and payroll_max_season is not None:
        modern_pay = df[(df["season_start"] >= 2015) & (df["season_start"] <= payroll_max_season) & (df["payroll_available"])].copy()
        pay_train = modern_pay[(modern_pay["season_start"] >= 2015) & (modern_pay["season_start"] <= min(2018, payroll_max_season))].copy()
        pay_test = modern_pay[(modern_pay["season_start"] >= max(2019, 2015)) & (modern_pay["season_start"] <= payroll_max_season)].copy()
        if pay_train.shape[0] >= 10 and pay_test.shape[0] >= 10:
            _fit_and_report("Modern w/ payroll (restricted)", pay_train, pay_test, combined_cols + payroll_feat)

    log("\n--- Point differential prediction ---")
    y_train_pm = train["plus_minus_per_game"].astype(float)
    y_test_pm = test["plus_minus_per_game"].astype(float)

    baseline_mae = float((y_test_pm - y_test_pm.mean()).abs().mean())
    log(f"Baseline MAE (predict mean): {baseline_mae:.3f}")

    # same feature sets but target is point differential
    pm_results = {}
    for name, cols in [("Offense", offense_cols), ("Defense", defense_cols), ("Combined", combined_cols)]:
        mae, _ = fit_model(train[cols], y_train_pm, test[cols], y_test_pm, cols)
        pm_results[name] = mae
        log(f"{name:15s} -> MAE: {mae:.3f} points/game")

    best_pm = min(pm_results.items(), key=lambda x: x[1])
    improvement = ((baseline_mae - best_pm[1]) / baseline_mae) * 100
    log(f"\nBest model: {best_pm[0]} (MAE: {best_pm[1]:.3f}) - {improvement:.1f}% better than baseline")

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

    if "po_data_available" in df.columns:
        log("\n--- Playoff analysis ---")

        # only seasons where playoff table has data
        df_po = df[df["po_data_available"]].copy()
        log(f"Playoff data available for {df_po.shape[0]} team-seasons")
        log(f"Made playoffs: {int(df_po['made_playoffs'].sum())} team-seasons")
        log(f"Champions: {int(df_po['champion'].sum())} team-seasons")

        cols = offense_cols + ["win_percentage"]
        log("\nRegular-season averages:")
        playoff_comp = df_po.groupby("made_playoffs")[cols].mean().round(3)
        playoff_comp.index = ["Non-playoff", "Playoff"]
        log(playoff_comp)

       
        log("\nChampion comparison:")
        champ_comp = df_po.groupby("champion")[cols].mean().round(3)
        champ_comp.index = ["Non-champion", "Champion"]
        log(champ_comp)

        corr_po = df_po[["win_percentage", "po_wins"]].corr(numeric_only=True).iloc[0, 1]
        log(f"\nCorrelation: regular-season win% vs playoff wins = {corr_po:.3f}")

        # quick playoff wins model with a time split
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

    log("\n--- League trends ---")


    league = df.groupby("season_start", as_index=False).agg(
        efg_mean=("efg", "mean"), threepa_rate_mean=("threepa_rate", "mean"),
        tov_rate_mean=("tov_rate", "mean"), ast_rate_mean=("ast_rate", "mean"),
        fta_rate_mean=("fta_rate", "mean")
    ).sort_values("season_start")

    log("\nLast 5 seasons:")
    log(league.tail(5)[["season_start", "efg_mean", "threepa_rate_mean"]].to_string(index=False))


    feature_cols = offense_cols + ["win_percentage"]
    corr = df[feature_cols].corr(numeric_only=True)["win_percentage"].sort_values(ascending=False)
    log("\nFeature correlations with win%:")
    for feat, val in corr.items():
        if feat != "win_percentage":
            log(f"  {feat:15s}: {val:6.3f}")

    fig_path = os.path.join(fig_dir, "league_trends_and_models.png")
    make_summary_figure(
        league=league,
        corr=corr,
        results=results,
        fig_path=fig_path,
        save_plots=SAVE_PLOTS,
        show_plots=SHOW_PLOTS,
    )
    if SAVE_PLOTS:
        log(f"[OUTPUT] Saved figure: {fig_path}", force=True)

    if SAVE_ENRICHED_CSV:
        enriched_path = os.path.join(out_dir, ENRICHED_CSV_NAME)
        df.to_csv(enriched_path, index=False)
        log(f"[OUTPUT] Wrote enriched dataset: {enriched_path} ({df.shape[0]} rows, {df.shape[1]} cols)", force=True)

    # quick summary
    log("\n--- Key findings ---")
    best_feature = corr[corr.index != "win_percentage"].idxmax()
    best_corr = corr[corr.index != "win_percentage"].max()
    log(f"1. Best predictor of win%: {best_feature} (r={best_corr:.3f})")
    best_model_name = min(results.items(), key=lambda x: x[1])[0]
    best_model_mae = min(results.values())
    best_model_lift = (win_baseline_mae - best_model_mae) / win_baseline_mae * 100.0
    log(
        f"2. Best model for win%: {best_model_name} (MAE: {best_model_mae:.2f} pp, "
        f"only {best_model_lift:+.1f}% vs predict-the-mean baseline {win_baseline_mae:.2f})"
    )

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


if __name__ == "__main__":
    main()
