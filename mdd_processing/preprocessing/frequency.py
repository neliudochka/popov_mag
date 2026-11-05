import pandas as pd
from .Data.Data import Data


def reduce_frequency(data: Data):
    lat_col = data.column_names.lat_col
    lon_col = data.column_names.lon_col

    if lat_col not in data.df.columns or lon_col not in data.df.columns:
        raise KeyError(f"Missing columns: {lat_col}, {lon_col}")

    cleaned_df = data.df.dropna(subset=[lat_col, lon_col])

    if cleaned_df.empty and not data.df.empty:
        raise ValueError("Cleaning resulted in an empty DataFrame. All coordinates are NaN.")

    return cleaned_df.reset_index(drop=True)

def increase_frequency(data_final: Data, data_original: Data):
    conf = data_final.config
    data_original.replace_config(conf)
    fill_data_with_original(data_final, data_original.df)


def fill_data_with_original(data_final, original_df):
    line_col = data_final.column_names.line_id_col
    timestamp_col = data_final.column_names.timestamp_col
    sensor_id_col = data_final.column_names.sensor_id_col
    track_id_col = data_final.column_names.track_id_col
    track_facing_col = data_final.column_names.track_facing_col
    mag1_cols = data_final.column_names.mag_cols1
    mag2_cols = data_final.column_names.mag_cols2

    df = data_final.df
    new_rows = []

    for line_id, df_part in df.groupby(line_col):
        first_ts = df_part[timestamp_col].iloc[0]
        last_ts = df_part[timestamp_col].iloc[-1]

        original_part = original_df.loc[
            original_df[timestamp_col].between(first_ts, last_ts)
        ]

        original_missing = (
            original_part.merge(
                df_part[[timestamp_col]],
                on=timestamp_col,
                how="left",
                indicator=True
            )
            .query("_merge == 'left_only'")
            .drop(columns="_merge")
        )

        if original_missing.empty:
            continue

        if df_part[sensor_id_col].iloc[0] == 2:
            original_missing[mag1_cols[0]] = original_missing[[mag2_cols[0]]]
            original_missing[mag1_cols[1]] = original_missing[[mag2_cols[1]]]
            original_missing[mag1_cols[2]] = original_missing[[mag2_cols[2]]]

        # Assign static fields from df_part
        original_missing[line_col] = line_id
        original_missing[sensor_id_col] = df_part[sensor_id_col].iloc[0]
        original_missing[track_id_col] = df_part[track_id_col].iloc[0]

        if track_facing_col is not None:
            original_missing[track_facing_col] = df_part[track_facing_col].iloc[0]

        new_rows.append(original_missing)

        # Append newly generated rows
    if new_rows:
        new_data = pd.concat(new_rows, ignore_index=True)

        data_final.df = (
            pd.concat([df, new_data], ignore_index=True)
            .sort_values([line_col, timestamp_col])
            .reset_index(drop=True)
        )

        data_final.df.drop(columns=mag2_cols, inplace=True)
