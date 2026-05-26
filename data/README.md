# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

+ `national_state2020.txt`: Contains the 2020 US state names, abbreviation and corresponding two-digit FIPS. Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html

+ `sc-est2023-agesex-civ.csv`: Contains the 2020-2023 US state population estimates. Downloaded from: https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/state/asrh/sc-est2023-agesex-civ.csv&ved=2ahUKEwjel9Xo4euTAxWFg4kEHQxFIqIQFnoECAwQAQ&usg=AOvVaw0uXd1QJzPHUinJjBpFRA-1

## Interim

### Demography

+ `demography.csv`: Population of US states (2023). Columns: 'name_state', 'fips_state', 'name_state', 'population', 'region_name'.

### Geography

+ `adjacency_matrix.csv`: Adjacency matrix of all US states.

+ `cluster.csv`: Encodes the clusters of US states the training/forecasting loop over. Must start with cluster 0, then 1, 2, 3, etc. The column `all` encodes having no clusters (run model for whole U.S.), while `census_regions` follows the four U.S. census regions. Shares the column `abbrevation_state` from `~/data/interim/demography/demography.csv` as index.

### NHSN-HRD_archive

+ `NHSN-HRD_ending-YYYYWW_gathered-YYYY-MM-DD-HH-MM-SS.parquet`: Hospital incidence for COVID-19, Influenza and RSV in every US state and per epiweek, as obtained from the NHSN HRD dataset. Automatically updated and archived weekly through a Github action: `~/.github/workflows/automated_data_collection.yml`. Columns: 'season', 'year', 'MMWR', 'date', 'fips_state', 'name_state', 'influenza admissions', 'covid-19 admissions', 'rsv admissions'.

### Calibration

To be filled out later.

## Conversion

### Demography

+ `build-demography.py`: Script used to build the US state-level demography.

### Cases

+ `fetch-format_NHSN-HRD-data.py`: Script to collect NHSN HRD data, format it and archive it. Called through a Github action: `~/.github/workflows/fetch-preliminary_NHSN_HRD.yml`.

+ `backfill_preliminary-NHSN-HRD-data.py`: Script used to backfill the preliminary NHSN HRD data. Called through a Github action: `~/.github/workflows/backfill-preliminary_NHSN_HRD.yml`.

### Geography

+ `make-adjacency_matrix.py`: Script to build an adjacency matrix of the US states. Uses a hard-coded dictionary of neighbours.