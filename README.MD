# NBA Team-Season Analysis (2000-2023)

This is a small NBA data science project. I pulled about two decades of team-season stats and looked at a few things: how the league has changed (mostly the three-point boom), which teams win more or less than their payroll would suggest, and how well you can actually predict win% from basic box-score stats (short answer: not that well).

## Links
- Tableau dashboard (Modern vs Past NBA): [View on Tableau Public](https://public.tableau.com/views/Dashboard1forNBA/Modernvspastnba)
- Tableau dashboard (NBA Value: Wins vs Payroll Residual): [View on Tableau Public](https://public.tableau.com/app/profile/eric.krog/viz/Dashboard1forNBA/NBAValueWinsvsPayrollwithResidual)

## Project goal
I wanted to answer a few straightforward questions:
- How has the league changed over the last two decades?
- Which team stats are most related to winning?
- Which teams overperform or underperform relative to payroll + stats?
- How well can you actually predict win% from basic team stats?

## Quick results
- Coverage:
  - Regular season: 2000-2023 (716 team-seasons)
  - Playoffs: 2000-2020 (626 team-seasons)
  - Payroll: 2000-2021 (656 team-seasons)
- The league changed a lot: 3PA rate roughly doubled (0.170 to 0.395 of all shots) and eFG% went up across 2000-2023. This is the strongest pattern in the data.
- Some teams consistently win more (or less) than their payroll and stats would predict. I dig into this in the "Payroll value lens" section below.
- Regular-season win% lines up pretty well with playoff success (r = 0.696).
- Best single correlation with win%: `efg` (r = 0.442)
- The win% models aren't great on held-out seasons. The best one (Defense features, MAE = 11.50 pp) only beats just guessing the average (11.92 pp) by about 3.5%, and the Offense and Combined sets actually do worse than guessing. So box-score rates on their own don't tell you much about future win%.
  - One caveat: the "Defense is best" result comes from a single 2018+ test split, so I wouldn't lean on it. It doesn't match the plain correlations (where `efg` and `ast_rate` matter most), so I'd treat it as a hint, not a real finding.
- With payroll (restricted seasons): Combined+Payroll MAE = 10.73 percentage points

## What the script does
`nba_analysis.py`:
- Loads team-season regular season CSV data
- Cleans basic data issues and standardizes columns
- Engineers features like `efg`, `threepa_rate`, `tov_rate`, `ast_rate`
- Runs Ridge models using a time split (train on earlier years, test on later years)
- Optionally merges playoff and payroll data
- Exports one enriched CSV for Tableau and saves a summary plot

## Repo structure

```text
README.MD
├── nba_analysis.py
├── requirements.txt
├── .gitignore
├── tableau
│   ├── nba_dashboards.twbx
│   ├── Modern_vs_past_nba.png
│   └── NBA_Value.png
├── repo_assets
│   └── league_trends_and_models.png
├── data
│   ├── README.md
│   └── raw
│       ├── nba_team_stats_00_to_23.csv
│       ├── nba_team_stats_playoffs_00_to_21.csv
│       └── nba_team_payroll_1990_to_2023.csv
└── outputs
    └── (generated artifacts)
```

## Data
Raw CSV files are in `data/raw/` (CC0 / Public Domain). Source notes are in `data/README.md`.

## Output files
Running the script writes:
- `outputs/nba_team_seasons_enriched.csv`
- `outputs/figures/league_trends_and_models.png`

## Modeling setup
- Train/test strategy: time-based split to avoid future-data leakage
- Model family: Ridge regression
- Metric: MAE in percentage points for win%
- Payroll comparison: only on seasons where payroll exists (restricted subset, no imputation)

## Payroll value lens
I also fit a value model and compute:
- `win_pct_expected`: expected win% (pp) based on payroll + team stats
- `win_pct_residual`: actual win% (pp) - expected win% (pp)

Interpretation:
- Positive residual = team did better than expected
- Negative residual = team did worse than expected

### Example value outliers
Top overperformers (relative to payroll+stats):
- Chicago Bulls (2004): +25.8 pp
- Indiana Pacers (2013): +25.8 pp
- Portland Trail Blazers (2013): +24.8 pp
- Chicago Bulls (2011): +23.8 pp
- Portland Trail Blazers (2014): +22.9 pp

Top underperformers:
- Phoenix Suns (2018): -24.8 pp
- Memphis Grizzlies (2006): -23.7 pp
- Philadelphia 76ers (2015): -23.7 pp
- Miami Heat (2007): -23.7 pp
- Golden State Warriors (2009): -23.5 pp

Most recent payroll season in merged data (2021):
- Best: Miami Heat (+14.6), Dallas Mavericks (+13.6), Phoenix Suns (+12.7)
- Worst: Indiana Pacers (-22.2), Houston Rockets (-18.6), Los Angeles Lakers (-18.5)

## Limitations
This is team-season level analysis, so it is correlational, not causal.

Main limitations:
- No direct injury/availability variable
- No opponent-adjusted strength of schedule variable
- No lineup-level or game-level context
- “Defense” features are proxies (not full opponent-based defensive metrics)
- Possessions are estimated with `0.96 * (FGA + TOV + 0.44*FTA - OREB)`
- Payroll and playoff coverage is not complete for all seasons

## Tableau
The workbook is in `tableau/nba_dashboards.twbx` and includes:
- Modern vs past NBA trend dashboard
- NBA value dashboard (payroll vs win% residual)

Interactive links:
- [Modern vs past NBA](https://public.tableau.com/views/Dashboard1forNBA/Modernvspastnba)
- [NBA Value (Wins vs Payroll with Residual)](https://public.tableau.com/app/profile/eric.krog/viz/Dashboard1forNBA/NBAValueWinsvsPayrollwithResidual)

Preview images:
- [Modern vs past NBA image](tableau/Modern_vs_past_nba.png)
- [NBA Value image](tableau/NBA_Value.png)

## Enriched dataset columns (important ones)
- `payroll_z_by_season`: payroll z-score within each season
- `win_pct_pp`: win percentage in 0-100 scale
- `win_pct_expected`: expected win% from payroll+stats model
- `win_pct_residual`: actual - expected win% (pp)
- `plus_minus_per_game`: regular-season point differential per game
- `po_games`, `po_wins`: playoff games and playoff wins
- `po_data_available`, `payroll_available`: data coverage flags

## How to run
1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Run:

```bash
python nba_analysis.py
```
