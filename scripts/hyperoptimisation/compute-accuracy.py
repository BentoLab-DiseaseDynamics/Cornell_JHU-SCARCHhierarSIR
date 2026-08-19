"""
A script to compute the Weighted Interval Score (WIS) and Mean Absolute Error (MAE) accuracy metric for a hyperoptimisation performed on synthetic data

"""

# packages needed
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# point to simulations
abs_dir = os.path.dirname(__file__)
sim_path = os.path.join(abs_dir, '../../data/interim/calibration/incremental_forecast')
data_path = os.path.join(abs_dir, '../../data/interim/cases/NHSN-HRD_archive/synthetic/early_season_decrease/NHSN-HRD_reference-date-2026-08-15_gathered-2026-08-12-16-21-04.parquet.gzip')
                        

# start of evaluation
eval_start_date = datetime(2026, 10, 15)
eval_end_date = datetime(2027, 5, 1)

# helper functions
def compute_WIS(simout, data):
    """
    Compute the WIS of a simulation in Hubverse format `simout` on groundtruth `data`.

    Input
    -----

    - simout: pd.DataFrame
        - Simulation in Hubverse format.
        - Columns: 'reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id', 'target_end_date', 'value'. 

    - data: pd.Series
        - Groundtruth data. Indexed on 'date'. No location.

    Output
    ------

    - WIS: pd.DataFrame
        - Columns: 'reference_date', 'horizon'
    """

    # get metadata
    reference_dates = simout['reference_date'].unique()
    horizon = simout['horizon'].unique()
    quantiles = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    # pre-allocate output dataframe
    idx = pd.MultiIndex.from_product([reference_dates, horizon], names=['reference_date', 'horizon'])
    WIS = pd.Series(index=idx, name='WIS')
    for reference_date in reference_dates:
        # Loop over horizon
        for n in horizon:
            n = float(n)
            ## get date
            date = reference_date+timedelta(weeks=n)
            ## get data
            try:
                y = data.loc[date]
            except:
                y = np.nan
            ## compute IS
            IS_alpha = []
            for q in quantiles:
                # get quantiles
                try:
                    l = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == q/2))]['value'].values[0]
                    u = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == 1-q/2))]['value'].values[0]
                except:
                    l = np.nan
                    u = np.nan
                # compute IS
                IS = (u - l)
                if y < l:
                    IS += 2/q * (l-y)
                elif y > u:
                    IS += 2/q * (y-u)
                IS_alpha.append(IS)
            IS_alpha = np.array(IS_alpha)
            ## compute WIS & assign
            try:
                m = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == 0.50))]['value'].values[0]
            except:
                m = np.nan
            WIS.loc[reference_date, n] = (1 / (len(quantiles) + 0.5)) * (0.5 * np.abs(y-m) + np.sum(0.5*np.array(quantiles) * IS_alpha))
        return WIS

def get_subfolders(folder_path):
    subfolders = [entry for entry in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, entry))]
    subfolders.sort()
    return subfolders

def list_files_in_directory(directory_path):
    files_list = []
    # Get all entries (files and directories) in the directory
    entries = os.listdir(directory_path)
    for entry in entries:
        # Construct the full path to check if it's a file
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files_list.append(entry)
    return files_list

# list the trainings
training_names = get_subfolders(sim_path)

# retrieve the target data
data = pd.read_parquet(data_path)
data['date'] = pd.to_datetime(data['date'])
data = data[data['date'] > eval_start_date]
data = data.sort_values(by=["date", "fips_state"]).reset_index()[['date', 'fips_state', 'influenza admissions']]
data["fips_state"] = data["fips_state"].astype(str).str.zfill(2)
all_locations = data['fips_state'].unique()
data = data.rename(columns={'fips_state': 'location', 'influenza admissions': 'value'})

# WIS computation loop
WIS_collection = []
print('Starting loop...')
training_acc_collect = []
for training_name in training_names:
    print(f'\tWorking on training: {training_name}')
    filenames = list_files_in_directory(os.path.join(sim_path, f'{training_name}/2026-2027/'))
    filenames = [fn for fn in filenames if fn != '.DS_Store']
    filenames.sort()
    fn_acc_collect = []
    for fn in filenames:
        # get the reference date
        ref_date = datetime.strptime(fn[:10], "%Y-%m-%d")
        # only use if larger than the eval date
        if ((ref_date >= eval_start_date) & (ref_date <= eval_end_date)):
            # get the forecasts
            forecast = pd.read_csv(os.path.join(sim_path, f'{training_name}/2026-2027/', fn), dtype={'location': str}, parse_dates=['reference_date', 'target_end_date'], date_format='%Y-%m-%d')
            # slice right target and metrics
            forecast = forecast[((forecast['target'] == 'wk inc flu hosp') & (forecast['output_type'] == 'quantile'))]
            forecast['output_type_id'] = forecast['output_type_id'].astype(float)
            locations = forecast['location'].unique()
            # loop over locations
            loc_acc_collect = []
            for loc in locations:
                # get the corresponding target data
                d = data[((data['date'].isin(forecast['target_end_date'].unique())) & (data['location'] == loc))][['date', 'value']].set_index('date').squeeze()
                # prevent collapse to float when there is only one value
                if isinstance(d, float):
                    d = pd.Series(index=[ref_date], data=d)
                # slice the right location in forecast
                fc = forecast[forecast['location'] == loc]
                # compute the WIS scores
                acc = compute_WIS(fc, d)
                acc = acc.reset_index()
                acc['location'] = loc
                acc['training_name'] = training_name
                # append the AE
                fc = fc[fc['output_type_id'] == 0.50]
                fc = fc.merge(d.rename("obs"), left_on="target_end_date", right_index=True, how='left')
                acc['MAE'] = np.abs((fc['value'] - fc['obs']).values)
                loc_acc_collect.append(acc)              
            fn_acc_collect.append(pd.concat(loc_acc_collect, axis=0))
    training_acc_collect.append(pd.concat(fn_acc_collect, axis=0))
training_acc = pd.concat(training_acc_collect, axis=0)

# omit horizon -1
training_acc = training_acc[training_acc['horizon'] != -1]

# build a maximalist dataframe
all_training_names = training_acc['training_name'].unique()
all_horizons = training_acc['horizon'].unique()
all_locations = data['location'].unique()
all_reference_dates = training_acc['reference_date'].unique()

index = pd.MultiIndex.from_product([all_training_names, all_reference_dates, all_locations, all_horizons], names=["training_name", "reference_date", "location", "horizon"])
df = pd.DataFrame(index=index, columns=['WIS', 'MAE'])

# join the WIS data
training_acc = training_acc.set_index(["training_name", "reference_date", "location", "horizon"])
df.update(training_acc)

print(df.groupby(by='training_name')['WIS'].mean())
print(df.groupby(by='training_name')['MAE'].mean())

# Save output to a .csv
df.to_csv('accuracy.csv')
