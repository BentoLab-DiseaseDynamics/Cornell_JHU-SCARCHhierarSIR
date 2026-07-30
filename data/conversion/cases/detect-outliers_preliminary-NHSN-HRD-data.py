"""
This script loads the archived preliminary NHSN HRD data and detects and corrects any outliers (f.i. state not reporting data in a given week).
It takes the latest available preliminary dataset in `~/data/interim/cases/NHSN-HRD_archive/preliminary`,
detects outliers and saves it in the `~/data/interim/cases/NHSN-HRD_archive/preliminary_outliers-removed` folder.

Author: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

import os
import glob
import numpy as np
import pandas as pd
from datetime import timedelta
from pygam import LinearGAM, s
import matplotlib.pyplot as plt


# define all paths reletive to this file
abs_dir = os.path.dirname(__file__)

# fit GAM to current date - `rolling_window_length` days
rolling_window_length = 365


###############################################
## Retrieve latest preliminary NHSN HRD data ##
###############################################

parquet_files = sorted(glob.glob(os.path.join(abs_dir, "../../interim/cases/NHSN-HRD_archive/preliminary/*.gzip")))

df = pd.read_parquet(parquet_files[-1])[['date', 'name_state', 'influenza admissions']]

df = df.loc[df['date'] >= max(df['date']) - timedelta(days=rolling_window_length)]


#################################
## Detect outliers using a GAM ##
#################################

states_w_outlier = []

for name_state in df['name_state'].unique():

    dt_train = df.loc[df['name_state'] == name_state, 'date'].iloc[:-1].values
    dt_test = df.loc[df['name_state'] == name_state, 'date'].iloc[-1]
    dt_full = df.loc[df['name_state'] == name_state, 'date'].values

    y_train = np.log1p(df.loc[df['name_state'] == name_state, 'influenza admissions'].iloc[:-1].values)
    y_test =  np.log1p(df.loc[df['name_state'] == name_state, 'influenza admissions'].iloc[-1])

    x_train = np.arange(len(y_train))
    x_test = np.arange(len(y_train) + 1)[-1]
    x_full = np.arange(len(y_train)+1)

    gam = LinearGAM(s(0), lam=0.05, n_splines=int(np.round(len(y_train)/2))).fit(x_train[:, None], y_train)

    trend = np.round(np.expm1(gam.predict(x_full[:, None])))
    confint = np.round(np.expm1(gam.confidence_intervals(x_full[:, None], width=0.9994)))
    y_train = np.expm1(y_train)
    y_test = np.expm1(y_test)

    is_outlier = (y_test < confint[-1, 0]) | (y_test > confint[-1, 1])

    if is_outlier:
        df.loc[((df['name_state'] == name_state) & (df['date'] == max(df['date']))), 'influenza admissions'] = np.expm1(trend[-1])
        states_w_outlier.append(name_state)

    if name_state == 'AL':
        fig,ax=plt.subplots(figsize=(8.7, 11.3/4))
        ax.set_title(name_state)
        ax.scatter(dt_train, y_train, marker='o', color='black', s=10)
        ax.scatter(dt_test, y_test, marker='o', color='black', s=10)
        if is_outlier:
            ax.scatter(dt_test, y_test, marker='x', color='red')
        ax.plot(dt_train, trend[:-1], color='green')
        ax.plot(np.array([dt_train[-1], dt_test]), np.array([trend[-2], trend[-1]]), color='red')
        ax.fill_between(dt_train, confint[:-1,0], confint[:-1,1], color='green', alpha=0.15)
        ax.fill_between(np.array([dt_train[-1], dt_test]), np.array([confint[-2,0], confint[-1, 0]]), np.array([confint[-2,1], confint[-1, 1]]), color='red', alpha=0.15)
        plt.show()
        plt.close()


#################
## Save result ##
#################

os.makedirs(os.path.join(abs_dir, '../../interim/cases/NHSN-HRD_archive/preliminary_outliers-removed/'), exist_ok=True)
df.to_parquet(os.path.join(abs_dir, '../../interim/cases/NHSN-HRD_archive/preliminary_outliers-removed/'+os.path.basename(parquet_files[-1])), compression='gzip', index=False)

