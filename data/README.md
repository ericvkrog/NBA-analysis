# Data Sources (CC0)

This project includes raw NBA datasets sourced from Kaggle. Both datasets are licensed under **CC0: Public Domain**.

Raw data files live in `data/raw/` and are kept unchanged (no manual edits). Any engineered features / merges are exported by the script to `outputs/nba_team_seasons_enriched.csv`.

---

## Source 1 (Kaggle, CC0)

- Dataset title: NBA Team Stats
- Author: mharvnek
- Link: https://www.kaggle.com/datasets/mharvnek/nba-team-stats-00-to-18
- License: CC0: Public Domain
- Date accessed: 2025-12-28

**Files used**
- `nba_team_stats_00_to_23.csv`  
  Regular-season team stats by season (used as the main modeling dataset).
- `nba_team_stats_playoffs_00_to_21.csv`  
  Playoff outcomes by season/team (merged on `Team` + `season_start`).

---

## Source 2 (Kaggle, CC0)

- Dataset title: NBA Players & Team Data
- Author: loganlauton
- Link: https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data
- License: CC0: Public Domain
- Date accessed: 2025-12-28

**Files used**
- `nba_team_payroll_1990_to_2023.csv`  
  Team payroll by season (merged on `Team` + `season_start`; coverage matches 2000–2021 in this project). (Renamed from the original Kaggle filename for consistency.)

---

## Notes on joins

- Main key: `Team` + `season_start`
- The script standardizes some team naming differences (e.g., franchise moves/renames) before merging.
- Payroll is standardized within each season using `payroll_z_by_season` to make spending comparable across eras.