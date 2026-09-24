"""
This script visualises the 4-week ahead forecast of the influenza model starting from the most recent NHSN HSN data
"""

__author__      = "T.W. Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group (JHUBSPH) & Bento Lab (Cornell CVM). All Rights Reserved."

import re
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# hierarchSIR functions
from SCARCHhierarSIR.data import  get_NHSN_HRD_data, get_demography


##############
## Settings ##
##############

plot_start_month = 8
hyperparameters = 'exclude_None-a_garch_0.2-phi_0.375-omega_0.005-targetaccept_0.85'
skip_fips = [29, 18, 21, 54, 1, 45, 11, 33, 24, 10, 25, 9]
start_calibration_month = 9 
plot_on = 'centroid'
move_dict = {'HI': (26.5, 8), 'AK': (24, -12), 'PR': (-6, 6),   # outlying islands
             'NV': (-0.1,1.5), 'TX': (0.7, 0), 'MI': (0, 0.9), 'WI': (0.2, -0.2), 'MN': (0,0.4), 'OH': (-0.2,0),    # Great lakes
             'OK': (-0.8, 0), 'MS': (0.1,-0.1), 'AR': (-0.1, 0.3),  # south
             'NY': (-1.8, 1), 'NJ': (0.8, -0.8), 'ME': (0.9,0.9), 'VT': (0,0.5),
             }


##########################
## Find latest forecast ##
##########################

def extract_timestamp(fname, pattern):
    match = pattern.search(fname.name)
    if match:
        return datetime.strptime(match.group(0)[:10], "%Y-%m-%d")
    return None

from pathlib import Path
forecast_folder = Path(os.path.join(os.path.dirname(__file__), f'../../data/interim/calibration/forecast/{hyperparameters}/'))
pattern = re.compile(r"\d{4}-\d{2}-\d{2}-Cornell_JHU-SCARCHhierarSIR")                                     # regex to capture gathered timestamp
files_with_time = [(f, extract_timestamp(f, pattern)) for f in forecast_folder.glob("*.csv")]          # collect files and their timestamps
files_with_time = [(f, t) for f, t in files_with_time if t is not None]
latest_forecast_file, reference_date = max(files_with_time, key=lambda x: x[1])                        # get the latest file


############################
## Load forecast and data ##
############################

# load state fips
state_fips_index, demography = get_demography()
state_fips_index['population'] = demography

# get the latest data (dummy)
_, data, dt, ts, n_observations = get_NHSN_HRD_data([datetime(2000,1,1),], [datetime(2026,1,1),], 1e4, type = 'preliminary_backfilled', forecast_horizon=4) # (n_season, n_variables, n_observations)
end_date = max(dt[0]).astype('datetime64[us]').astype('O')

# helper function
def get_influenza_season_label(date: datetime) -> str:
    """
    Given a datetime, return the influenza season label in the format 'YYYY-YYYY'.
    Season runs from September 1 to August 31.
    """
    year = date.year
    if date.month >= 9:  # September or later → start of new season
        start_year = year
        end_year = year + 1
    else:  # January–August → still in previous season
        start_year = year - 1
        end_year = year
    return f"{start_year}-{end_year}"
# retrieve latest season
season = get_influenza_season_label(end_date)
# get actual season start date
start_date = datetime(int(season[0:4]), plot_start_month, 1)

# now query the data for real
_, data, dt, ts, n_observations = get_NHSN_HRD_data([start_date,], [datetime(2026,1,1),], 1e4, type = 'preliminary_backfilled', forecast_horizon=0) # (n_season, n_variables, n_observations)

# group it in a pandas dataframe
multi_index = pd.MultiIndex.from_product([state_fips_index['fips_state'], dt[0]], names=['fips_state', 'date'])
data = pd.DataFrame({'value': data[0].reshape(-1)}, index=multi_index).reset_index()

# normalize the data
data = pd.merge(data, state_fips_index[['fips_state', 'population']],  on='fips_state', how='left')
data['value_per_100k'] = (data['value'] / data['population']) * 10E5

# get the previous season data
start_date_prev = datetime(int(season[0:4])-1, plot_start_month, 1)
_, data_prev, dt_prev, ts, n_observations = get_NHSN_HRD_data([start_date_prev,], [datetime(2026,1,1),], 1e4, type = 'preliminary_backfilled', forecast_horizon=0) # (n_season, n_variables, n_observations)

# align it with this year
end_month = (reference_date + timedelta(weeks=5)).month
end_day = (reference_date + timedelta(weeks=5)).day
dt_prev= dt_prev[0]
dt_prev = dt_prev[dt_prev <= start_date_prev + timedelta(weeks=len(dt[0]) + 5)]
data_prev = data_prev[0, :, :len(dt_prev)]

multi_index = pd.MultiIndex.from_product([state_fips_index['fips_state'], dt_prev], names=['fips_state', 'date'])
data_prev = pd.DataFrame({'value': data_prev.reshape(-1)}, index=multi_index).reset_index()
data_prev['date'] = data_prev['date'] + timedelta(days=365)

# normalize it
data_prev = pd.merge(data_prev, state_fips_index[['fips_state', 'population']],  on='fips_state', how='left')
data_prev['value_per_100k'] = (data_prev['value'] / data_prev['population']) * 10E5

# get the latest forecast (For now, assuming there is only )
forecast = pd.read_csv(latest_forecast_file, index_col=0)
forecast = forecast[forecast['target'] ==  'wk inc flu hosp']
forecast['output_type_id'] = pd.to_numeric(forecast['output_type_id'])
fips_state_list =  forecast['location'].unique().tolist()
fips_state_list = [x for x in fips_state_list if x != 'US']
fips_state_list = [int(x) for x in fips_state_list]
fips_state_list = [x for x in fips_state_list if x not in skip_fips]
fips_mappings = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/interim/demography/demography.csv'), dtype={'fips_state': int})
name_state_list = [fips_mappings.loc[fips_mappings['fips_state'] == x]['abbreviation_state'].squeeze() for x in fips_state_list]
forecast["target_end_date"] = pd.to_datetime(forecast["target_end_date"])

# get the shapefiles
gdf = gpd.read_file(os.path.join(os.path.dirname(__file__),f'../../data/raw/geography/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'))
gdf["representative_point"] = gdf.geometry.representative_point()
gdf["centroid"] = gdf.geometry.representative_point()
gdf["STATEFP"] = gdf["STATEFP"].astype(int)


###################
## Build the map ##
###################

# base map
fig, ax = plt.subplots(figsize=(24.1*0.75, 13.9*0.75))
gdf.plot(ax=ax, color="black", alpha=0.15, edgecolor="black")

# plot forecasts per state
for name_state, fips_state in zip(name_state_list, fips_state_list):

    # get x and y of the state's location
    gdf_row = gdf[gdf['STATEFP'] == fips_state]
    cx, cy = gdf_row[plot_on].x.values[0], gdf_row[plot_on].y.values[0]
    
    # correct with move_dict
    try:
        delta_cx, delta_cy = move_dict[name_state]
        cx += delta_cx
        cy += delta_cy
    except:
        pass
    # get state forecast quantiles
    fc = forecast[forecast["location"] == str(fips_state).zfill(2)]

    # normalize to incidence per 100K
    pop = state_fips_index.loc[state_fips_index['fips_state'] == fips_state, 'population'].values[0]
    fc['value'] = fc['value'] / pop * 10E5

    # slice data
    data_slice = data[data['fips_state'] == fips_state]
    data_prev_slice = data_prev[data_prev['fips_state'] == fips_state]

    # inset axes
    iax = inset_axes(ax, width=1, height=0.7, loc="center",
                    bbox_to_anchor=(cx, cy),
                    bbox_transform=ax.transData,
                    borderpad=0)

    # plot forecast intervals
    iax.fill_between(fc["target_end_date"].unique(), fc.loc[fc['output_type_id'] == 0.25, 'value'], fc.loc[fc['output_type_id'] == 0.75, 'value'], color="green", alpha=0.2)
    iax.fill_between(fc["target_end_date"].unique(), fc.loc[fc['output_type_id'] == 0.025, 'value'], fc.loc[fc['output_type_id'] == 0.975, 'value'], color="green", alpha=0.1)
    iax.scatter(data_slice['date'], data_slice['value_per_100k'], color='black', alpha=1, linestyle='None', facecolors='black', s=10, linewidth=1)
    iax.plot(data_prev_slice['date'], data_prev_slice['value_per_100k'], color='red', alpha=1, linewidth=0.5)
    
    # inside your loop, after plotting into iax
    iax.xaxis.set_major_locator(mdates.MonthLocator())
    iax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # %b = abbreviated month name
    iax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    iax.tick_params(axis='x', labelsize=5, rotation=0)
    iax.tick_params(axis='y', labelsize=5)
    iax.set_xlim([start_date, end_date+timedelta(weeks=1)])
    iax.set_ylim([-10,250])

    # put state in
    iax.text(
    0.05, 0.95,                   # position (x,y) in axes fraction coordinates
    f'{name_state}',                   # text string
    transform=iax.transAxes,      # use axes coordinates (0–1)
    fontsize=6,
    va="top", ha="left",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.5)
    )

ax.set_xlim([-128, -67])
ax.set_ylim([22, 52])
ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), f'../../data/interim/calibration/forecast/{hyperparameters}/{reference_date.strftime("%Y-%m-%d")}-Cornell_JHU-hierarchSIR.svg'))
plt.close()