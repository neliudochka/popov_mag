import sys
from dataclasses import fields, asdict

import numpy as np
import pandas as pd
from pandas import DataFrame

from mdd_processing.preprocessing.Data.ColumnNames import ColumnNames
from mdd_processing.preprocessing.Data.Data import Data


def find_nearest(external_ts: float, main_ts_series: pd.Series, max_diff_ms= None):
    if pd.isna(external_ts):
        return None

    diffs = np.abs(main_ts_series.values - external_ts)
    min_idx = diffs.argmin()

    if max_diff_ms is not None and diffs[min_idx] > max_diff_ms:
        return None

    return main_ts_series.index[min_idx]


def periods_overlap(t1_st, t1_end, t2_st, t2_end) -> bool:
    return t1_st < t2_end and t2_st < t1_end


def rename_columns_eg(eg: Data, suffix: str = "_eg"):
    eg.df.rename(
        columns=lambda col: col + suffix,
        inplace=True,
    )

    for field in fields(eg.column_names):
        value = getattr(eg.column_names, field.name)

        if isinstance(value, str):
            setattr(eg.column_names, field.name, value + suffix)


def merge_gnss_into_data(main_data, external_data, max_diff_ms = None):
    main_df = main_data.df
    external_df = external_data.df

    main_ts_col = main_data.column_names.utc_timestamp_col
    external_ts_col = external_data.column_names.utc_timestamp_col

    main_start = main_df[main_ts_col].iloc[0]
    main_end = main_df[main_ts_col].iloc[-1]
    ext_start = external_df[external_ts_col].iloc[0]
    ext_end = external_df[external_ts_col].iloc[-1]

    if not periods_overlap(main_start, main_end,
                           ext_start, ext_end):
        raise ValueError(
            f"Time periods of files {main_data.filepath} and {external_data.filepath} do not overlap: "
            f"main=[{main_start}, {main_end}], external=[{ext_start}, {ext_end}]"
        )

    id_column = "_id"
    time_diff_col = "_time_diff"
    external_df[id_column] = range(len(external_df))

    merged_df = pd.merge_asof(
        main_df.sort_values(main_ts_col),
        external_df.sort_values(external_ts_col),
        left_on=main_ts_col,
        right_on=external_ts_col,
        direction="nearest",
        tolerance=max_diff_ms
    )

    merged_df[time_diff_col] = (merged_df[main_ts_col] - merged_df[external_ts_col]).abs()

    best_indices = (
        merged_df
        .dropna(subset=[id_column])
        .sort_values(time_diff_col)
        .drop_duplicates(id_column, keep="first")
        .index
    )

    duplicate_mask = (
            merged_df[id_column].notna()
            & ~merged_df.index.isin(best_indices)
    )

    for column in external_df.columns:
        merged_df.loc[duplicate_mask, column] = np.nan
    merged_df.drop(columns=[id_column, time_diff_col], inplace=True)
    return merged_df


def replace_own_col_with_eg(merged_df: DataFrame, own_cn: ColumnNames, eg_cn: ColumnNames):
    if merged_df.empty:
        raise ValueError("DataFrame is empty")

    column_replacement = {eg_cn.lat_col:own_cn.lat_col,
                          eg_cn.lon_col:own_cn.lon_col}
    if eg_cn.alt_col in merged_df.columns:
        column_replacement[own_cn.alt_col]=eg_cn.alt_col
    if own_cn.lat_col in merged_df.columns:
        merged_df.drop(columns=column_replacement.values(), inplace=True)
    merged_df.rename(columns=column_replacement, inplace=True)

    columns_to_drop = [val for val in asdict(eg_cn).values() if val is not None and val not in asdict(own_cn).values() and val in merged_df.columns]
    merged_df.drop(columns=columns_to_drop, inplace=True)
    return merged_df


def prepare_ext_gnss_data(data: Data):
    # delete duplicates
    if data.column_names.nmea_col in data.df.columns:
        data.df = data.df.drop_duplicates(subset=[data.column_names.nmea_col], keep='first')
    else:
        data.df = data.df.drop_duplicates(subset=[data.column_names.lat_col, data.column_names.lon_col, data.column_names.gps_time_col], keep='first')
