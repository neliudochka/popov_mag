import pandas as pd
import numpy as np
from pandas import DataFrame
from pyproj import Geod

from .Data.Data import Data, get_lines_first_indexes
from .frequency import reduce_frequency


def calculate_step_distances(df: DataFrame, lat_col: str, lon_col: str, line_id_col: str) -> pd.Series:
    """
        Calculates geodesic distance between neighbouring GNSS points.
        Returns a pandas Series of distances in meters.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Columns '{lat_col}' or '{lon_col}' not found in DataFrame.")

    if df.empty:
        return pd.Series(dtype=float)

    if df[lat_col].isna().any() or df[lon_col].isna().any():
        raise ValueError("None values in Latitude or/and Longitude columns. Impossible to find step between None values.")

    try:
        geod = Geod(ellps="WGS84")
        _, _, dist = geod.inv(
            df[lon_col].values[:-1],
            df[lat_col].values[:-1],
            df[lon_col].values[1:],
            df[lat_col].values[1:]
        )

        dist_step = np.insert(dist, 0, 0)
        first_indices = get_lines_first_indexes(df, line_id_col)
        dist_step[first_indices] = 0

        return pd.Series(dist_step)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate geodesic distances: {e}")


def get_median_step_distance(df: DataFrame, line_id_col : str, lat_col: str, lon_col: str, step_col: str = "Step [m]") -> float:
    if step_col in df.columns:
        steps = df[step_col]
    else:
        steps = calculate_step_distances(df, lat_col, lon_col, line_id_col)
        df[step_col] = steps # for better computational time

    return steps.drop(index=get_lines_first_indexes(df, line_id_col)).median()


def meters_to_points(offset: float, step_dist: float) -> int:
    """ pass offset in meters,
        step_dist (average distance between 2 neighboring points) in meters"""
    return round(offset / step_dist)


def lag_correction_tfa_col(df: DataFrame, tfa_col: str, line_id_col: str, step_col: str, lat_col: str, lon_col: str, offset: float) -> pd.Series | None:
    """Returns corrected TFA column. Offset is in meters"""
    missing = {tfa_col, line_id_col, lat_col, lon_col} - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")

    if offset == 0:
        return df[tfa_col]

    col = tfa_col
    line = line_id_col

    return df.groupby(line)[col].shift(meters_to_points(offset, get_median_step_distance(df, line_id_col, lat_col, lon_col, step_col))).fillna(0)


def calc_or_gnss_ext_gnss_offest(or_data: Data, ext_gnss_data: Data):
    ts_col_or = or_data.column_names.timestamp_col
    ts_col_ext = ext_gnss_data.column_names.timestamp_col

    or_data.df = reduce_frequency(or_data)

    ext_gnss_data.df = ext_gnss_data.df[
        ext_gnss_data.df[ts_col_ext].isin(or_data.df[ts_col_or])
    ]

    or_data.df = or_data.df[
        or_data.df[ts_col_or].isin(ext_gnss_data.df[ts_col_ext])
    ]


    or_lat = or_data.df[or_data.column_names.lat_col]
    or_lon = or_data.df[or_data.column_names.lon_col]

    ext_lat = ext_gnss_data.df[ext_gnss_data.column_names.lat_col]
    ext_lon = ext_gnss_data.df[ext_gnss_data.column_names.lon_col]

    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(
        or_lon,
        or_lat,
        ext_lon,
        ext_lat
    )

    return dist
