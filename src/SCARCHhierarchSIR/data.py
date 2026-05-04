"""
This script contains all functions related to data prepping

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

import os
import re
import arviz
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, Optional, Tuple, List, Union

# Define relevant global  variables
abs_dir = os.path.dirname(__file__)

################################
## Data fetching and prepping ##
################################

def get_demography(regions: Optional[Iterable[str]]=None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and optionally filter state-level demographic data.

    Parameters
    ----------
    regions : Optional[Iterable[str]], default=None
        Iterable of region names to filter on (matching `region_name` column in `~/data/interim/demography/demography.csv`).
        If None, all regions are included.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        state_fips_index : pd.DataFrame
            DataFrame with columns:
            ['abbreviation_state', 'name_state', 'fips_state'].
        demography : np.ndarray
            Array of population counts corresponding to the returned states.
    """

    demo = pd.read_csv(
        os.path.join(abs_dir, "../../data/interim/demography/demography.csv"),
    )

    if regions is not None:
        demo = demo[demo["region_name"].isin(regions)]

    state_fips_index = demo[
        ["abbreviation_state", "name_state", "fips_state"]
    ]

    demography = demo["population"].to_numpy()

    return state_fips_index, demography



def get_adjacency_matrix(abbreviation_state: Iterable[str]) -> pd.DataFrame:
    """
    Load and subset the state adjacency matrix to the specified states.

    Parameters
    ----------
    abbreviation_state : list
        List containing state abbreviations.
        The ordering of this list determines the ordering of the
        returned adjacency matrix.

    Returns
    -------
    np.ndarray
        Square adjacency matrix indexed and columned by state abbreviations,
    """

    adj = pd.read_csv(
        os.path.join(abs_dir, "../../data/interim/geography/adjacency_matrix.csv"),
        index_col=0,
    )

    return adj.loc[abbreviation_state, abbreviation_state].values



def extract_timestamp(fname: Path, pattern: re.Pattern[str]) -> Optional[datetime]:
    """
    Extract a timestamp from the NHSN HRD data filenames using a regex pattern.

    The filename is expected to contain a single capturing group corresponding
    to a timestamp formatted as "%Y-%m-%d-%H-%M-%S".

    Parameters
    ----------
    fname : Path
        File path whose name will be searched.
    pattern : re.Pattern[str]
        Compiled regex pattern with one capturing group for the timestamp.

    Returns
    -------
    Optional[datetime]
        Parsed datetime if a match is found, otherwise None.
    """
    match = pattern.search(fname.name)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
    return None



def get_most_recent_filename(data_folder: Path) -> Path:
    """
    Retrieve the most recent NHSN HRD parquet file path in a folder based on timestamps embedded in the filenames.

    Also returns the CDC FluSight reference date of these data.

    Filenames are expected to contain the pattern:
    'gathered-YYYY-MM-DD-HH-MM-SS*.parquet.gzip'.

    Parameters
    ----------
    data_folder : Path
        Directory containing parquet files.

    Returns
    -------
    Path
        Path to the most recent file.

    reference_date: str
        CDC FluSight reference date

    Raises
    ------
    ValueError
        If no valid timestamped files are found.
    """

    # find the most recent file
    pattern = re.compile(r"gathered-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")

    files_with_time: List[Tuple[Path, Optional[datetime]]] = [
        (f, extract_timestamp(f, pattern))
        for f in data_folder.glob("*.parquet.gzip")
    ]

    files_with_time = [(f, t) for f, t in files_with_time if t is not None]

    if not files_with_time:
        raise ValueError(f"No valid timestamped files found in {data_folder}")

    latest_file, _ = max(files_with_time, key=lambda x: x[1])

    # return the reference date
    pattern = re.compile(r"reference-date-(\d{4}-\d{2}-\d{2})")

    match = pattern.search(latest_file.name)
    if not match:
        raise ValueError(f"No reference date found in {latest_file}")

    reference_date = datetime.strptime(match.group(1), "%Y-%m-%d")

    return latest_file, reference_date



def get_NHSN_HRD_data(
    start_calibrations: Iterable[pd.Timestamp],
    modifier_reference_dates: Iterable[pd.Timestamp],
    n_observations: int,
    type: str = "preliminary_backfilled",
    forecast_horizon: int = None,
    state_fips: Optional[Iterable[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Format influenza hospitalization data for model input.

    Parameters
    ----------
    start_calibrations : Iterable[pd.Timestamp]
        Start date of calibration period for each season.
    modifier_reference_dates : Iterable[pd.Timestamp]
        Reference dates defining t=0 for each season's time index.
    n_observations : int
        Number of weekly observations used for calibration.
    type : str, default="preliminary_backfilled"
        Data source type. Must be either:
        - "preliminary"
        - "preliminary_backfilled"
    forecast_horizon : int or None
        Number of weeks to extend beyond observed data for forecasting.
        Use 'None' for training (default)
    state_fips : Optional[Iterable[int]], default=None
        List of state FIPS codes to include. If None, all states are used.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        reference_date: str (%Y-%M-%D)
            CDC FluSight reference date (Saturday of week following data end)
        data : np.ndarray
            Shape: (n_seasons, n_states, n_timepoints), weekly admissions (scaled by 1/7).
        dates : np.ndarray
            Shape: (n_seasons, n_timepoints), datetime values.
        timesteps : np.ndarray
            Shape: (n_seasons, n_timepoints), time in days since model's reference date (t=0; October 15).
        true_n_observations: int
            The available number of observations (input `n_observations` can exceed the available data).

    Raises
    ------
    ValueError
        If an invalid data type is provided.
    
    Notes
    -----
    n_seasons should be equal to one for forecasting
    """

    # determine type of data to use
    if type == "preliminary":
        data_folder = Path(abs_dir) / "../../data/interim/cases/NHSN-HRD_archive/preliminary/"
    elif type == "preliminary_backfilled":
        data_folder = Path(abs_dir) / "../../data/interim/cases/NHSN-HRD_archive/preliminary_backfilled/"
    else:
        raise ValueError("`type` must be 'preliminary' or 'preliminary_backfilled'.")
    # determine if training or forecasting
    if forecast_horizon:
        assert len(start_calibrations) == len(modifier_reference_dates) == 1, 'length of `start_calibrations` and `modifier_reference_dates` must be equal to one when using this function for forecasting'

    data: List[np.ndarray] = []
    dates: List[np.ndarray] = []
    timesteps: List[np.ndarray] = []

    for start_calibration, modifier_reference_date in zip(start_calibrations, modifier_reference_dates):

        # load most recent dataset and its CDC FluSight reference date
        path, reference_date = get_most_recent_filename(data_folder)
        df = pd.read_parquet(path)

        # basic cleaning
        ## convert date column to datetime and fips_state to int
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        df["fips_state"] = df["fips_state"].astype(int)
        ## slice out US states of interest
        if state_fips is not None:
            df = df[df["fips_state"].isin(state_fips)]
        ## slice out variables of interest
        df = df[["date", "fips_state", "influenza admissions"]]
        ## trim bottom temporally
        df = df[df["date"] > start_calibration]
        ## Backward fill per state (Happens first week of season 2024-2025 in 3 states)
        df["influenza admissions"] = (df.groupby("fips_state")["influenza admissions"].bfill())

        # determine the data's end date
        user_end_date = start_calibration + timedelta(weeks=n_observations)
        if forecast_horizon:
            # assume 'forecasting' mode, only one season, number of observations can exceed the end of the data
            last_existing_date = df["date"].max()
            if user_end_date <= last_existing_date:
                target_end_date = user_end_date + pd.Timedelta(weeks=forecast_horizon)
            else:
                target_end_date = last_existing_date + pd.Timedelta(weeks=forecast_horizon)
        else:
            # assume 'training' mode, more than one season, number of observations in each season must be identical
            target_end_date = user_end_date

        # generate dataframe encompassing calibration + forecast ranges
        all_dates = pd.date_range(start=df["date"].min(), end=target_end_date, freq="7D")
        all_fips = df["fips_state"].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_fips], names=["date", "fips_state"])
        df = (df.set_index(["date", "fips_state"]).reindex(full_index).reset_index())

        # save the data's time index relative to the forward simulation model's t=0 (per season) + dates
        unique_dates = df["date"].unique()
        dates.append(unique_dates)
        timesteps.append(np.array([(d - modifier_reference_date) / timedelta(days=1) for d in unique_dates]))

        # extract the data as a (n_states x n_observations) numpy array
        data_matrix = (
            df.pivot(index="fips_state", columns="date", values="influenza admissions")
            .sort_index()
            .sort_index(axis=1)
            .to_numpy()
        )
        data.append(data_matrix)
    # stack data to (n_season, n_states, n_observations)
    data_arr = np.stack(data, axis=0)
    dates_arr = np.stack(dates, axis=0)
    timesteps_arr = np.stack(timesteps, axis=0)

    # compute the actual number of observations
    if forecast_horizon:
        n_observations = len(timesteps_arr[0]) - forecast_horizon

    return reference_date, data_arr, dates_arr, timesteps_arr, n_observations

########################################
## Conversion and Hubverse formatting ##
########################################

def simout_to_hubverse(simout: arviz.InferenceData,
                        reference_date: datetime,
                        state_fips_index: dict,
                        target: str,
                        quantiles: bool=False) -> pd.DataFrame:
    """
    Convert pyMC simulation result to CDC FluSight's Hubverse format

    Parameters
    ----------

    - simout: arviz.InferenceData
        - pyMC simulation output. model state "pred". dimensions: (draw, state, horizon).
        - flatten the 'chain' and 'draw' axes 

    - reference_date: datetime
        - when using data until a Saturday `x` to calibrate the model, `reference_date` is the date of the next saturday `x+1`.

    - state_fips_index: dict
        - keys: state abbreviations (e.g., 'MT', type str)
        - values: state fips codes (e.g., '30', type str, with leading zero for fips codes 1-9)

    - target: str
        - simulation target, typically 'wk inc flu hosp'.

    - quantiles: str
        - save quantiles instead of individual trajectories.

    Returns
    -------

    - hubverse_df: pd.Dataframe
        - forecast in hubverse format
        - contains the total hospital admissions in the 'value' column

    Reference
    ---------

    https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md#Forecast-file-format
    """

    # deduce information from simout
    abbreviation_state = simout.coords["state"].values.tolist()
    fips_state = [f"{state_fips_index[s]:02d}" for s in abbreviation_state]
    output_type_id = simout.coords['draw'].values if not quantiles else [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    # fixed metadata
    horizon = simout.coords["horizon"].values.tolist()
    output_type = 'samples' if not quantiles else 'quantile'

    # pre-allocate dataframe
    idx = pd.MultiIndex.from_product([[reference_date,], [target,], horizon, fips_state, [output_type,], output_type_id],
                                        names=['reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id'])
    df = pd.DataFrame(index=idx, columns=['value',])
    # attach target end date
    df = df.reset_index()
    df['target_end_date'] = df.apply(lambda row: row['reference_date'] + timedelta(weeks=row['horizon']), axis=1)

    # fill in dataframe
    for fips,abbrev in zip(fips_state, abbreviation_state):
        if not quantiles:
            for draw in output_type_id:
                df.loc[((df['output_type_id'] == draw) & (df['location'] == fips)), 'value'] = \
                    7*simout.sel(state=abbrev).sel({'draw': draw}).values
        else:
            for q in output_type_id:
                df.loc[((df['output_type_id'] == q) & (df['location'] == fips)), 'value'] = \
                    7*simout.sel(state=abbrev).quantile(q=q, dim='draw').values
    
    # Round to the closest integer and convert to int
    df["value"] = df["value"].round().astype(int)

    return df