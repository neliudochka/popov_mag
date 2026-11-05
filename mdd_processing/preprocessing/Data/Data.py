import copy
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .DataConfig import DataConfig
from .ColumnNames import ColumnNames, build_column_rename_map
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

VECTOR_COLUMN_TYPE = "vector"
MODULE_DIR = Path(__file__).resolve().parent
default_config = MODULE_DIR / "data_configs" / "default.yaml"

class Data:
    DEFAULT_CONFIG: DataConfig
    column_names = ColumnNames()
    config : DataConfig

    def __init__(self, df: pd.DataFrame, filepath: Optional[str|Path] = None, config: Optional[DataConfig] = None, delimiter: Optional[str] = None):
        self.df = df
        self.filepath = filepath
        self.config = None
        self.apply_config(config or DataConfig.from_yaml(default_config))
        if delimiter:
            self.config.delimiter = delimiter
        self._date = None
        self.set_date()

    def __repr__(self):
        return f'Data\nDf:\n: {self.df}\n{self.config}'

    def __len__(self):
        return len(self.df)

    def apply_config(self, new_config: DataConfig):
        self.config = new_config
        self.column_names = gen_column_names(new_config)

    @classmethod
    def from_file(cls, filepath: str | Path, config: Optional[DataConfig] = None, delimiter: Optional[str] = None, skip_rows: int = None):
        config = config or DataConfig.from_yaml(default_config)
        df = read_df(filepath, config, delimiter, skip_rows)
        return cls(df, filepath=filepath, delimiter=delimiter, config=config)

    @classmethod
    def from_empty(cls, filepath: Optional[str|Path] = None, config: Optional[DataConfig] = None, delimiter: Optional[str] = None):
        if config is None:
            config = DataConfig.from_yaml(default_config)
        if delimiter is None:
            delimiter = config.delimiter
        columns = get_to_use_col(config)
        df = pd.DataFrame(columns=columns)
        return cls(df, filepath=filepath, delimiter=delimiter, config=config)

    @classmethod
    def from_df(cls, df: pd.DataFrame, filepath: Optional[str|Path] = None, config: Optional[DataConfig] = None, delimiter: Optional[str] = None):
        if config is None:
            config = DataConfig.from_yaml(default_config)
        if delimiter is None:
            delimiter = config.delimiter
        df = df[get_existing_use_columns(df, config)]
        return cls(df, filepath=filepath, delimiter=delimiter, config=config)

    def get_date(self):
        if self._date is None:
            raise ValueError("Data's date is not set")
        return self._date

    def has_date(self) -> bool:
        return self._date is not None

    def set_date(self, date: Optional[datetime] = None):
        if date is not None:
            self._date = date
            return

        gps_date_col = self.column_names.gps_date_col
        if gps_date_col in self.df.columns:
            first_valid_idx = self.df[gps_date_col].first_valid_index()
            if first_valid_idx is not None:
                self._date = self.df[gps_date_col].loc[first_valid_idx]
                return

        utc_tmstp_col = self.column_names.utc_timestamp_col
        if utc_tmstp_col in self.df.columns:
            first_valid_idx = self.df[utc_tmstp_col].first_valid_index()
            if first_valid_idx is not None:
                self._date = datetime.fromtimestamp(self.df[utc_tmstp_col].loc[first_valid_idx] / 1000.0).date()
                return

    def add_column(self, col_name: str, values=None, index: Optional[int] = None):
        if values is None:
            self.df.insert(index if index is not None else len(self.df.columns), col_name, np.nan)
        else:
            self.df[col_name] = values

    def save(self, new_filepath: Optional[str] = None, new_delimiter: Optional[str] = None):
        path = new_filepath or self.filepath
        delim = new_delimiter or self.config.delimiter
        to_save_col = get_to_save_col(self.config)
        save_columns(self.df, to_save_col, path, delim)

    def num_tracks(self):
        if self.column_names.track_id_col in self.df.columns:
            return int(self.df[self.column_names.track_id_col].nunique())
        return 0

    def column_freq(self, column_name):
        if column_name not in self.df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")
        if self.df.empty:
            raise ValueError(f"DataFrame is empty. No data to process.")

        df_non_null = self.df.dropna(subset=[column_name])

        if df_non_null.empty:
            raise ValueError(f"Column {column_name} is empty.")

        df = df_non_null.head(1000).reset_index(drop=True)

        ts_part = df[self.column_names.timestamp_col] / 1000 #convert to seconds
        intervals = ts_part.diff().dropna()
        estimated_interval_mode = intervals.median()
        frequency_hz_mode = round(1 / estimated_interval_mode)
        return frequency_hz_mode

    def gen_line_id(self):
        track_id = self.df[self.column_names.track_id_col]
        sensor_id = self.df[self.column_names.sensor_id_col]
        self.df[self.column_names.line_id_col] = track_id * 2 - (sensor_id == 1).astype(int)

    def replace_config(self, new_conf: DataConfig, remove_un_used_columns=True) -> None:
        """Replace current config with a new one. Change all the column names in column_names and in df"""
        src_cn = self.column_names
        self.apply_config(new_conf)
        dst_cn = self.column_names
        rename_map, col_to_drop = build_column_rename_map(src_cn, dst_cn)
        self.df = self.df.rename(columns=rename_map)
        if remove_un_used_columns:
            self.df = drop_columns(self.df, col_to_drop)

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def get_track_ids(self):
        if self.column_names.track_id_col in self.df.columns:
            unique_tr_id = self.df[self.column_names.track_id_col].unique()
            return np.sort(unique_tr_id)
        return None

    def renumber_tracks(self):
        if self.column_names.track_id_col in self.df.columns:
            track_col = self.column_names.track_id_col

            # Get unique track IDs in stable order
            track_ids = self.get_track_ids()

            # Build mapping: old_id -> new_id
            id_map = {old_id: new_id for new_id, old_id in enumerate(track_ids, start=1)}

            # Apply mapping
            self.df[track_col] = self.df[track_col].map(id_map)
        else:
            return None

    def get_frequency(self):
        if self.column_names.line_id_col and self.column_names.line_id_col in self.df.columns:
            diff = self.df.groupby(self.column_names.line_id_col)[
                self.column_names.utc_timestamp_col].diff().dropna().astype("int32")
        elif self.column_names.track_id_col and self.column_names.track_id_col in self.df.columns:
            diff = self.df.groupby(self.column_names.track_id_col)[self.column_names.utc_timestamp_col].diff().dropna().astype("int32")
        else:
            diff = self.df[self.column_names.utc_timestamp_col].diff().dropna().astype("int32")
        return int(1000 / diff.mode()[0])


def get_line_ids(df, line_id_col):
    return df[line_id_col].unique()

def get_lines_first_indexes(df, line_id_col: str):
        return [df[line_id_col].eq(lid).idxmax() for lid in get_line_ids(df, line_id_col)]


def save_columns(df: pd.DataFrame, sc, path, delim):
    existing_to_save = [col for col in sc if col in df.columns]
    if existing_to_save:
        df.to_csv(path, sep=delim, index=False, columns=existing_to_save)
    return df

def get_col_with_bool_prop(config, prop, b: bool):
    c = []
    for name, col in config.columns.items():
        if getattr(col, prop) is b:
            if col.type == VECTOR_COLUMN_TYPE:
                c.extend(col.name)
            else:
                c.append(col.name)
    return c

def get_to_use_col(config):
    return get_col_with_bool_prop(config, "use", True)

def get_unused_col(config):
    return get_col_with_bool_prop(config, "use", False)

def get_to_save_col(config):
    return get_col_with_bool_prop(config, "save", True)

def get_existing_use_columns(df_cols, config: DataConfig) -> list[str]:
    use_col_all = get_to_use_col(config)
    return [c for c in use_col_all if c in df_cols]


def read_df(filepath: str, config: Optional[DataConfig] = None, delimiter: Optional[str] = None, skip_rows: int = None) -> pd.DataFrame:
    """
    Reads only the columns from the CSV that are marked `use=True` in the config and actually exist in the file.
    """
    if config is None:
        raise ValueError("DataConfig must be provided")
    if delimiter is None:
        delimiter = config.delimiter

        # read only the header
    file_header = pd.read_csv(filepath, delimiter=delimiter, nrows=0, encoding="utf-8").columns.tolist()
    use_cols = get_existing_use_columns(file_header, config)

    # read only the needed columns
    df = pd.read_csv(filepath, delimiter=delimiter, usecols=use_cols, skiprows=skip_rows, encoding_errors="ignore")
    return df

def drop_columns(df: pd.DataFrame, cols_to_drop):
    existing_to_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
    return df

def gen_column_names(dt: DataConfig) -> ColumnNames:
    cn = ColumnNames()
    for name, col in dt.columns.items():
        if col.use:
            if hasattr(cn, name):
                setattr(cn, name, col.name)
    return cn

