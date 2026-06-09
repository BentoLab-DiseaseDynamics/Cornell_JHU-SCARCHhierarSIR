"""
This script computes the seasonal average influenza hospital admissions from the 2023-2024 season until the last available season (currently: 2025-2026)..
..then extends the last available season (currently: 2025-2026) until epiweek 36 (end of August) using the seasonal average..
..then makes a hypothetical next season (currently: 2026-2027) using the seasonal average, letting the user implement deviations from the historical average.

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

#############################
## Dependencies & settings ##
#############################

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from SCARCHhierarSIR.data import get_most_recent_filename

# define paths globally
abs_dir = os.path.dirname(__file__)
save_data_path = '../../interim/cases/NHSN-HRD_archive/hypothetical/'

# modifications
start_mmwr = 6                                                                  # modification start week (must be post-January 1, MMWR 1)
factors = [0.05, 0.15, 0.45, 1.35, 1.35, 0.45, 0.45, 0.15, 0.15, 0.05, 0.05]    # multiplicative scaling of seasonal average number of influenza admissions

######################################################################
## Compute seasonal average number of influenza hospital admissions ##
######################################################################

# load most recent dataset and its CDC FluSight reference date
data_folder = Path(abs_dir) / "../../interim/cases/NHSN-HRD_archive/preliminary_backfilled/"
path, reference_date = get_most_recent_filename(data_folder)
df = pd.read_parquet(path)

# filter seasons
start_season = '2023-2024'
df = df[df['season'] >= start_season]

# compute the seasonal average number of influenza admissions
df_avg = (
    df
    .groupby(['fips_state', 'MMWR'], as_index=False)['influenza admissions']
    .mean()
    .rename(columns={'influenza admissions': 'influenza_adm_mean'})
)

##################################
## Impute last available season ##
##################################

# 1. build full grid for last season
states_df = (
    df[['fips_state', 'name_state']]
    .drop_duplicates()
)
weeks_df = pd.DataFrame({'MMWR': range(1, 36)})
full_index = states_df.merge(weeks_df, how='cross')
full_index['season'] = df['season'].max()
full_index['year'] = int(df['season'].max().split('-')[1])
# 2. merge with the original data
last_df = df[df['season'] == df['season'].max()]
merged = full_index.merge(
    last_df,
    on=['season', 'fips_state', 'MMWR'],
    how='left',
    suffixes=('', '_obs')
)
# 3. merge seasonal average
merged = merged.merge(
    df_avg,
    on=['fips_state', 'MMWR'],
    how='left'
)
# 4. fill in influenza admissions where missing
merged['influenza admissions'] = merged['influenza admissions'].fillna(
    merged['influenza_adm_mean']
)
# 5. throw out junk
cols_to_keep = [
    'season', 'year', 'MMWR',
    'fips_state', 'name_state',
    'influenza admissions',
    'date'
]
merged = merged[cols_to_keep]
# 6. fill in the missing dates
merged['date'] = merged['date'].fillna(
    pd.to_datetime(
        merged['year'].astype(int).astype(str)
        + '-W' + merged['MMWR'].astype(int).astype(str)
        + '-6',                     # Saturday of that week
        format='%G-W%V-%u',
        errors='coerce'
    )
)
print(merged['influenza admissions'].isna().sum())
# 7. merge with the original data
df_extended = pd.concat(
    [df[cols_to_keep], merged],
    ignore_index=True
)

##################################
## Make a synthetic next season ##
##################################

# 1. generate the index
weeks = list(range(37, 53)) + list(range(1, 37))
states_df = df_extended[['fips_state', 'name_state']].drop_duplicates()
future = (
    states_df.merge(
        pd.DataFrame({'MMWR': weeks}),
        how='cross'
    )
)
last_season = df_extended['season'].iloc[-1]
start_year, end_year = map(int, last_season.split('-'))
future['season'] = f"{start_year + 1}-{end_year + 1}"
future['year'] = future['MMWR'].apply(
    lambda w: 2026 if w >= 37 else 2027
)

# 2. merge the seasonal average
future = future.merge(
    df_avg,
    on=['fips_state', 'MMWR'],
    how='left'
) 
future = future.rename(columns={'influenza_adm_mean': 'influenza admissions'})

# 3. merge the dates
n_weeks = future['MMWR'].nunique()
start_date = df_extended['date'].max() + pd.Timedelta(weeks=1)
weekly_dates = pd.date_range(
    start=start_date,
    periods=n_weeks,
    freq='W-SAT'
)
week_order = list(range(37, 53)) + list(range(1, 37))
week_to_date = dict(zip(week_order, weekly_dates))
future = future.copy()
future['date'] = future['MMWR'].map(week_to_date)

# 4. make a modification of the future
unique_weeks = np.array(sorted(future['MMWR'].unique()))
mask = np.isin(unique_weeks, range(start_mmwr, start_mmwr + len(factors)))
target_weeks = unique_weeks[mask]
week_multiplier = dict(zip(target_weeks, 1+np.array(factors)))
future['multiplier'] = future['MMWR'].map(week_multiplier).fillna(1.0)
future['influenza admissions'] = future['influenza admissions'] * future['multiplier']
future = future.drop(columns=['multiplier'])

# 5. append to the dataset
df_final = pd.concat([df_extended, future], ignore_index=True)

# 6. visually check the results
fips = 0  # choose your state
df_plot = df_final[df_final['fips_state'] == fips].sort_values('date')
plt.figure()
plt.plot(df_plot['date'], df_plot['influenza admissions'])
plt.title(f'Influenza admissions – FIPS {fips}')
plt.xlabel('Date')
plt.ylabel('Admissions')
#plt.show()
plt.close()

##################
## Save results ##
##################

## Make folder
desired_path = os.path.join(abs_dir, save_data_path)
if not os.path.exists(desired_path):
    os.makedirs(desired_path)
## Dump a copy of the raw data
df_final.to_parquet(os.path.join(desired_path, path.name), compression='gzip', index=False)