import re
from datetime import datetime, time, timezone, date
from pathlib import Path
from typing import Tuple
import pandas as pd
from .Data.Data import Data
from .Data.DataConfig import TimestampType


def get_datetime_str_by_filepath(filepath_str):
    parts = Path(filepath_str).stem.split("_", 2)
    if len(parts) < 2:
        raise ValueError(
            f"Invalid filepath format '{filepath_str}', expected 'YYYYMMDD_HHMMSS_[...]'"
        )

    return f"{parts[0]}_{parts[1]}"


def get_date_and_time_by_filepath(filepath_str):
    return get_date_and_time_by_str(get_datetime_str_by_filepath(filepath_str))


def get_date_and_time_by_str(datetime_str: str) -> Tuple[date, time]:
    try:
        date_str, time_str = datetime_str.split("_")
    except ValueError:
        raise ValueError(f"Invalid datetime '{datetime_str}', expected 'YYYYMMDD_HHMMSS.mmm'")
    return get_date_by_str(date_str), get_time_by_str(time_str)


def get_date_by_str(date_str: str) -> date:
    try:
        match = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", date_str)
        if not match:
            raise ValueError(f"In '{date_str}' there is no date in format YYYYMMDD")
        date_str = match.group(0)
        return datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{date_str}', expected YYYYMMDD"
        ) from exc


def get_time_by_str(time_str: str) -> time:
    try:
        return datetime.strptime(time_str, "%H%M%S.%f").time()
    except ValueError:
        try:
            return datetime.strptime(time_str, "%H%M%S").time()
        except ValueError as exc:
            raise ValueError(
                f"Invalid time '{time_str}', expected HHMMSS.mmm"
            ) from exc


def round_timestamp_ms(value, step: int = 200):
    if isinstance(value, pd.Series):
        return (value / step).round().mul(step).astype("int64")
    return int(round(value / step) * step)


def seconds_to_utc_ms(tm: float) -> int:
    return int(tm * 1000)


def to_utc_timestamp_ms(date_value: date, t: time | None = None) -> int:
    dt = datetime.combine(date_value, t or time.min)
    dt = dt.replace(tzinfo=timezone.utc)
    return seconds_to_utc_ms(dt.timestamp())


def _resolve_date(filepath, date_specified):
    if date_specified is not None:
        d = get_date_by_str(date_specified)
    else:
        d, _ = get_date_and_time_by_filepath(filepath)
    return d


def add_date(data, dt_specified : str | None = None) -> None:
    data.set_date(_resolve_date(data.filepath, dt_specified))


def compute_utc_ts_from_gps(data: Data):
    gps_time_col = data.column_names.gps_time_col
    timestamp_col = data.column_names.timestamp_col

    valid_mask = data.df[gps_time_col].notna() & (data.df[gps_time_col] != 0)
    if valid_mask.isnull().all():
        raise ValueError("GPS time column has no valid values")
    valid_values = data.df[gps_time_col][valid_mask]
    first_valid_index = valid_values.index[0]
    first_valid_time = str(valid_values.iloc[0])
    first_tmst = round_timestamp_ms(to_utc_timestamp_ms(data.get_date(), get_time_by_str(first_valid_time)), 200)

    anchor_ts = data.df.loc[first_valid_index, timestamp_col]
    timestamps = first_tmst + (data.df[timestamp_col] - anchor_ts)
    return timestamps


def compute_day_ts_from_utc_ts(utc_ts: pd.Series) -> pd.Series:
    FULL_DAY_MS = 86400000
    return utc_ts % FULL_DAY_MS


def compute_utc_tc_from_day_tc(day_tc: pd.Series, day_date) -> pd.Series:
    day_timestamps = day_tc
    return day_timestamps + to_utc_timestamp_ms(day_date)


def compute_utc_ts_from_mission_ts(data: Data) -> pd.Series:
    gps_time_col = data.column_names.gps_time_col

    if gps_time_col not in data.df.columns or data.df[gps_time_col].isnull().all():
        day_timestamps = data.df[data.column_names.timestamp_col]
        _, first_ts = get_date_and_time_by_filepath(data.filepath)
        return day_timestamps + to_utc_timestamp_ms(data.get_date(), first_ts)
    else:
        return compute_utc_ts_from_gps(data)


def time_str_to_utc(dt: pd.Series):
    return to_utc_timestamp_ms(get_time_by_str(f"{dt}"))

def get_utc_timestamps(data: Data, timestamp_type: TimestampType = None) -> pd.Series:
    timestamp_col = data.column_names.timestamp_col
    timestamp_col_type = data.config.columns['timestamp_col'].type
    timestamp_col_type = timestamp_type if timestamp_type else timestamp_col_type

    if timestamp_col_type == TimestampType.UTC:
        return data.df[timestamp_col]
    elif timestamp_col_type == TimestampType.DAY:
        return compute_utc_tc_from_day_tc(data.df[timestamp_col], data.get_date())
    elif timestamp_col_type == TimestampType.MISSION:
        return compute_utc_ts_from_mission_ts(data)
    else:
        raise ValueError(f"Unknown timestamp type: {timestamp_col_type}")


def add_utc_timestamps(data, timestamp_type: TimestampType = None) -> None:
    utc_timestamp_col = data.column_names.utc_timestamp_col
    timestamps = get_utc_timestamps(data, timestamp_type)
    data.add_column(utc_timestamp_col,  timestamps)
    data.df[utc_timestamp_col] = data.df[utc_timestamp_col].apply(
        lambda x: "{:.3f}".format(x)
    )



roman_numbers = {
    "I": "01",
    "II": "02",
    "III": "03",
    "IV" : "04",
    "V": "05",
    "VI": "06",
    "VII": "07",
    "VIII": "08",
    "IX": "09",
    "X": "10",
    "XI": "11",
    "XII": "12"
}

def roman_to_arabic(rom: str) -> int:
    if rom in roman_numbers:
        return roman_numbers[rom]
    else:
        raise ValueError(f"Wrong format of the month {rom}")
