import datetime
import logging
import os
import pathlib
import sys
from contextlib import contextmanager
import time
from enum import Enum

import numpy as np
import pandas as pd
from numpy import astype
from pyproj import Geod

from .Data.Data import Data
from .Data.DataConfig import DataConfig
from .external_gnss import merge_gnss_into_data, prepare_ext_gnss_data, rename_columns_eg, replace_own_col_with_eg
from .handle_date_time import add_utc_timestamps, add_date, roman_to_arabic, to_utc_timestamp_ms, get_time_by_str
from .interpolation import interpolate_for_lines
from .magnetic_calibration import apply_magnetometer_calibration
from .magnetic_field import add_total_field, add_tfa
from .sensors_offset import add_sensor_positions_by_heading, add_sensor_positions_by_heading_popov
from .trajectory_segmentation import extract_tracks
from .attitude_determination import calculate_yaw, get_facing_tracks, calculate_tilted_yaw, get_yaw_of_track, \
    get_B_reference

INTERNAL_BASE_DIR = pathlib.Path(__file__).resolve().parent

# Fallback to package root '
_data_configs_dir = INTERNAL_BASE_DIR / "Data" / "data_configs"
if not _data_configs_dir.exists():
    import mdd_processing
    _pkg_root = pathlib.Path(mdd_processing.__file__).resolve().parent
    INTERNAL_BASE_DIR = _pkg_root / "preprocessing" 
    _data_configs_dir = INTERNAL_BASE_DIR / "Data" / "data_configs"

DATA_OWN_CONFIG_PATH = INTERNAL_BASE_DIR / "Data" / "data_configs" / "popov_init.yaml"
DATA_FINAL_CONFIG_PATH = INTERNAL_BASE_DIR / "Data" / "data_configs" / "popov.yaml"
DATA_EXTERNAL_GNSS_PATH = INTERNAL_BASE_DIR / "Data" / "data_configs" / "external_gnss_popov.yaml"


class Step(Enum):
    def __new__(cls, description):
        obj = object.__new__(cls)
        value = len(cls.__members__) # index
        obj._value_ = value
        obj.description = description
        return obj

    DOWNLOAD = "Downloading data"
    CALIBRATE = "Calibrating magnetometer sensors"
    UTC_TIMESTAMPS = "Adding date and UTC timestamps"
    ADD_SENSOR2 = "Add 2 sensor data"
    ADD_VARIATIONS = "Add variations"
    EXTERNAL_GNSS = "Applying external GNSS data"
    TRAJECTORY_SEGMENTATION = "Trajectory segmentation. Heading calculation"
    APPLY_OFFSET = "Apply offset (distance from gps to mag sensors)"
    ORIENTATION = "Orientation calculation"
    FACING = "Determining facing direction"
    SIDE_TRACKS = "Calculating the coordinates of side tracks"
    DROP_BAD_MAG = "Drop bad mag data"
    GNSS_INTERPOLATION = "Interpolating GNSS data"
    TOTAL_FIELD = "Calculating total field"
    TOTAL_FIELD_ANOMALY = "Calculating total field anomaly"
    SAVE = "Saving results"

    @classmethod
    def max_value(cls):
        return len(cls) - 1

    @contextmanager
    def run(self, logger, callback=None, kwargs=None):
        logger.info(f"Step {self.value}: {self.description}")
        if kwargs:
            logger.info(f"Arguments {kwargs}")
        if callback:
            callback(self)

        start = time.perf_counter()
        try:
            yield
        except Exception:
            logger.exception(f"Failed at step {self.name}")
            raise
        finally:
            logger.debug(
                f"{self.name} completed in {time.perf_counter() - start:.3f}s"
            )


def get_data_with_config_popov_mag(data_path, config_path, delimiter=None) -> Data:
    dc = DataConfig.from_yaml(config_path)
    if data_path:
        df = pd.read_csv(data_path, delimiter=r"\s+", skiprows=9)
        data = Data.from_df(df, data_path, dc, delimiter=delimiter)
        date_line = open(data_path, 'r').readlines()[2].split()
        if len(date_line) == 7:
            day = date_line[-3]
            mon = roman_to_arabic(date_line[-2])
            year = date_line[-1]
        elif len(date_line) == 6:
            day = date_line[-2]
            mon = roman_to_arabic(date_line[-1][0:3])
            year = date_line[-1][3:-1]
        else:
            raise ValueError(f"Date line has another format {date_line}")
        add_date(data, f"20{year}{mon}{day}")
        data.replace_config(DataConfig.from_yaml(DATA_FINAL_CONFIG_PATH))
        if data.df.empty:
            raise ValueError(f"File {data_path} is empty")
        return data
    return Data.from_empty(data_path, dc, delimiter=delimiter)

def get_data_with_config(data_path, config_path, delimiter=None) -> Data:
    dc = DataConfig.from_yaml(config_path)
    if data_path:
        data = Data.from_file(data_path, dc, delimiter=delimiter)
        if data.df.empty:
            raise ValueError(f"File {data_path} is empty")
        return data
    return Data.from_empty(data_path, dc, delimiter=delimiter)


def step_download(state, *, data_filepath, delim):
    state.data_own = get_data_with_config_popov_mag(
        data_filepath, DATA_OWN_CONFIG_PATH, delimiter=delim
    )
    return state.data_own


def step_timestamps(state, *, datetime_specified):
    if datetime_specified is not None or not state.data_own.has_date():
        add_date(state.data_own, datetime_specified)
    data = state.data_own
    data.df = data.df.dropna(subset=[data.column_names.timestamp_col])
    clean_timestamps = data.df[data.column_names.timestamp_col].astype(str).map(get_time_by_str)
    state.data_own.df[data.column_names.utc_timestamp_col]=clean_timestamps.map(lambda x: to_utc_timestamp_ms(data.get_date(), x))
    return state.data_own


def step_add_sensor2(state):
    state.data_own.df[state.data_own.column_names.s2_tf] = (state.data_own.df[state.data_own.column_names.s1_tf]
                                                            - state.data_own.df[state.data_own.column_names.s2_grad])
    return state.data_own


def step_external_gnss(state, *, external_gnss_filepath):
    external_data = None
    if os.path.basename(external_gnss_filepath).startswith(("Golova", "Rover")):
        df = pd.read_csv(external_gnss_filepath, delimiter=r'\s+',
                         # skiprows=9
                         )
        print(df.columns)

        dt = pd.to_datetime(
            df["%"] + "/" + df["UTC"],
            format="%Y/%m/%d/%H:%M:%S.%f",
            utc=True
        )

        df["Timestamp [ms]"] = dt.astype("int64") // 10 ** 6
        df.rename(columns={"latitude(deg)": "Lat [DD]", "longitude(deg)": "Lon [DD]", "UTC": "GPS Time"}, inplace=True)
        df["Timestamp [ms]"] += 3 * 60 * 60 * 1000

        mission_dir = pathlib.Path(external_gnss_filepath).parent.absolute()
        mission_name = pathlib.Path(external_gnss_filepath).name.split(".")[0]

        external_data = Data.from_df(df, filepath=external_gnss_filepath, config=DataConfig.from_yaml("/media/mila/WinToUSB/Users/ropuh/mila/Code/project/mdd_processing/preprocessing/Data/data_configs/external_gnss.yaml"))
        print(external_data.df)
    else:
        external_data = get_data_with_config(
            external_gnss_filepath, DATA_EXTERNAL_GNSS_PATH
        )
        external_data.df[external_data.column_names.utc_timestamp_col] = (pd.to_datetime(external_data.df[external_data.column_names.gps_time_col], format='%Y-%m-%d %H:%M:%S.%f')).astype('int64') // 10**6
    prepare_ext_gnss_data(external_data)
    rename_columns_eg(external_data)
    merged_df = merge_gnss_into_data(state.data_own, external_data, max_diff_ms=150)
    state.data_own.df = replace_own_col_with_eg(merged_df, state.data_own.column_names, external_data.column_names)
    state.data_own.df.to_csv("smth.csv")
    return state.data_own


def step_variations(state, variations_filepath):
    var_tm_col = "timestamp_var"
    var_var_col = "variations"

    vartn = pd.read_csv(
        variations_filepath,
        delimiter=r"\s+",
        skiprows=6,
        encoding_errors="ignore",
        names=[var_var_col, "time"],
    )


    from datetime import datetime
    vartn[var_tm_col] = vartn["time"].map(lambda x: datetime.strptime(x, "%H:%M:%S.%f").time()).map(lambda x: to_utc_timestamp_ms(state.data_own.get_date(), x))

    dt_tmst = state.data_own.df[state.data_own.column_names.utc_timestamp_col]
    var_tmst = vartn[var_tm_col]

    max_diff_ms = 1000
    id_column = "_id"
    time_diff_col = "_time_diff"
    vartn[id_column] = range(len(vartn))
    dt_utc_col = state.data_own.column_names.utc_timestamp_col

    merged_df = pd.merge_asof(
        state.data_own.df.sort_values(dt_utc_col),
        vartn.sort_values(var_tm_col),
        left_on=dt_utc_col,
        right_on=var_tm_col,
        direction="nearest",
        tolerance=max_diff_ms
    )

    merged_df[time_diff_col] = (merged_df[state.data_own.column_names.utc_timestamp_col] - merged_df[var_tm_col]).abs()
    # best_indices = (
    #     merged_df
    #     .dropna(subset=[id_column])
    #     .sort_values(time_diff_col)
    #     .drop_duplicates(id_column, keep="first")
    #     .index
    # )
    #
    # duplicate_mask = (
    #         merged_df[id_column].notna()
    #         & ~merged_df.index.isin(best_indices)
    # )
    #
    # for column in vartn.columns:
    #     merged_df.loc[duplicate_mask, column] = np.nan
    # merged_df.drop(columns=[id_column, time_diff_col], inplace=True)

    merged_df[var_var_col] = merged_df[var_var_col] - merged_df[var_var_col].dropna().iloc[0]
    state.data_own.df[var_var_col] = merged_df[var_var_col]
    state.data_own.df[var_var_col] = state.data_own.df[var_var_col].fillna(0)
    state.data_own.df[state.data_own.column_names.s1_tf] -= state.data_own.df[var_var_col].to_numpy()
    state.data_own.df[state.data_own.column_names.s2_tf] -= state.data_own.df[var_var_col].to_numpy()

    return state.data_own



def step_trajectory_segmentation(state, max_angle_diff, min_segment_len):
    extract_tracks(state.data_own, max_angle_diff, min_segment_len)
    if not state.data_own.df[state.data_own.column_names.track_id_col].notna().any():
        raise ValueError("No tracks found")
    return state.data_own

def step_apply_offset(state, geom_offset):
    lat_col = state.data_own.column_names.lat_col
    lon_col = state.data_own.column_names.lon_col
    heading_col = state.data_own.column_names.heading_track_col

    df = state.data_own.df

    valid = (
        df[lat_col].notna()
        & df[lon_col].notna()
        & df[heading_col].notna()
    )

    lat = df.loc[valid, lat_col].to_numpy()
    lon = df.loc[valid, lon_col].to_numpy()
    heading = df.loc[valid, heading_col].to_numpy()

    geod = Geod(ellps="WGS84")

    new_lon, new_lat, _ = geod.fwd(
        lon,
        lat,
        heading,
        np.full(len(lat), geom_offset),
    )

    df.loc[valid, lat_col] = new_lat
    df.loc[valid, lon_col] = new_lon

    return state.data_own


def step_orientation(state):
    # check results
    FIRST_FACING = -1
    if FIRST_FACING == 1:
        state.data_own.df[state.data_own.column_names.yaw_track_col] = state.data_own.df[state.data_own.column_names.heading_track_col].iloc[0]
    elif FIRST_FACING == -1:
        df = state.data_own.df[state.data_own.df[state.data_own.column_names.track_id_col] == 2]
        state.data_own.df[state.data_own.column_names.yaw_track_col] = df[state.data_own.column_names.heading_track_col].iloc[0]
    else:
        raise ValueError(f"Wrong value of FIRST_FACING {FIRST_FACING}")

    return state.data_own


def step_facing(state, max_deviation):
    # ft = get_facing_tracks(state.data_own, max_deviation)
    tf_col = state.data_own.column_names.track_facing_col
    # tid_col = state.data_own.column_names.track_id_col
    # state.data_own.add_column(tf_col, ft)
    # ids = state.data_own.df[state.data_own.df[tf_col] == 0][tid_col].unique()
    # if len(ids) > 0:
    #     raise ValueError(f"Impossible to determine the direction of movement in tracks: {ids} (track's facing = 0). Change fc_max_deviation or track_max_angle parameters.")
    state.data_own.df[tf_col] = 1
    state.data_own.df.to_csv("check.csv")
    return state.data_own


def step_side_tracks(state, use_facing, s1_offset, s2_offset):
    state.data_final = get_data_with_config(None, DATA_FINAL_CONFIG_PATH)
    state.data_final = add_sensor_positions_by_heading_popov(state.data_own, state.data_final.config, s1_offset, s2_offset)
    state.data_final.gen_line_id()
    return state.data_final

def step_delete_bad_mag_data(state):
    # Скорочення для зручності читання коду
    df = state.data_final.df
    cols = state.data_final.column_names

    # ПРАВИЛЬНО: заперечення `~` стоїть перед усім виразом у спільних дужках.
    # Ми видаляємо ЛИШЕ ті рядки, де якість == 0 І sensor_id == 1 одночасно.
    df = df[~((df[cols.s1_qual] == 0) & (df[cols.sensor_id_col] == 1))]

    # Повторюємо логіку для другого сенсора
    df = df[~((df[cols.s2_qual] == 0) & (df[cols.sensor_id_col] == 2))]

    # Записуємо результат назад та повертаємо об'єкт, як у зразку
    state.data_final.df = df
    return state.data_final


def step_gnss_interpolation(state, alt):
    state.data_own.df.to_csv("before_interp.csv")

    state.data_final.df[state.data_final.column_names.alt_col] = alt
    state.data_final.df = interpolate_for_lines(state.data_final)
    return state.data_final


def step_total_field(state):
    add_total_field(state.data_final)
    return state.data_final


def step_total_field_anomaly(state, filter_type, filt_window, cutoff_freq):
    add_tfa(state.data_final, state.data_final.column_names.total_field_anomaly_col, filter_type, filt_window, cutoff_freq)
    return state.data_final


def step_save(state, result_filepath, data_filepath, external_gnss_filepath, filter_type, filt_window, cutoff_freq):
    if result_filepath is None:
        result_filepath = str(data_filepath).split(".")[0] + f"_{filter_type}{filt_window}_ctf{cutoff_freq}"
        if external_gnss_filepath:
            result_filepath = str(result_filepath).split(".")[0] + "_eg"  #eg = external_gnss
        result_filepath += ".asc"
    state.data_final.save(new_filepath=result_filepath, new_delimiter='\t')

# 1) зчитати дані +
# 2) перейменувати колонки +
# 3) Обрахувати mag 2 сенсор +
# 4) Синхронізувати з GPS +
# Хедінг (з розворотами, тобто фейсінг 1) +
# Розбити на треки +
# Facing +
# Офсет сенсора
# 6) Інтерполяція
# 7) Обрахувати TFA
# Lag correction
# знову

PIPELINE = [
    {
        "step":  Step.DOWNLOAD,
        "func": step_download,
        "kwargs": lambda ctx: {
            "data_filepath": ctx["data_filepath"],
            "delim": ctx["delim"],
        },
        "condition": lambda ctx: True,
    },
    {
        "step": Step.UTC_TIMESTAMPS,
        "func": step_timestamps,
        "kwargs": lambda ctx: {
            "datetime_specified": ctx["datetime_specified"],
        },
        "condition": lambda ctx: True,
    },
    {
        "step": Step.ADD_SENSOR2,
        "func": step_add_sensor2,
        "kwargs": lambda ctx: {},
        "condition": lambda ctx: True,
    },
    {
        "step": Step.EXTERNAL_GNSS,
        "func": step_external_gnss,
        "kwargs": lambda ctx: {
            "external_gnss_filepath": ctx["external_gnss_filepath"],
        },
        "condition": lambda ctx: ctx["external_gnss_filepath"] is not None,
    },
    {
        "step": Step.ADD_VARIATIONS,
        "func": step_variations,
        "kwargs": lambda ctx: {
            "variations_filepath": ctx["systeminfo_filepath"],
        },
        "condition": lambda ctx: ctx["systeminfo_filepath"] is not None,
    },
    {
        "step": Step.TRAJECTORY_SEGMENTATION,
        "func": step_trajectory_segmentation,
        "kwargs": lambda ctx: {
            "max_angle_diff": ctx["tr_seg_max_angle_diff"],
            "min_segment_len": ctx["tr_seg_min_segment_len"]
        },
        "condition": lambda ctx: True,
    },
    {
        "step": Step.APPLY_OFFSET,
        "func": step_apply_offset,
        "kwargs": lambda ctx: {
            "geom_offset": ctx["geom_offset"],
            # "min_segment_len": ctx["tr_seg_min_segment_len"]
        },
        "condition": lambda ctx: True,
    },
    # {
    #     "step": Step.ORIENTATION,
    #     "func": step_orientation,
    #     "kwargs": lambda ctx: {},
    #     "condition": lambda ctx: True,
    # },
    {
        "step": Step.FACING,
        "func": step_facing,
        "kwargs": lambda ctx: {
            "max_deviation": ctx["fc_max_deviation"]
        },
        "condition": lambda ctx: True,
    },
    {
        "step": Step.SIDE_TRACKS,
        "func": step_side_tracks,
        "kwargs": lambda ctx: {
            "use_facing": ctx["use_facing"],
            "s1_offset": ctx["s1_offset"],
            "s2_offset": ctx["s2_offset"]
        },
        "condition": lambda ctx: True,
    },
    {
        "step": Step.DROP_BAD_MAG,
        "func": step_delete_bad_mag_data,
        "kwargs": lambda ctx: {},
        "condition": lambda ctx: True,
    },
    {
        "step": Step.GNSS_INTERPOLATION,
        "func": step_gnss_interpolation,
        "kwargs": lambda ctx: {
            "alt": ctx["alt"]
        },
        "condition": lambda ctx: True
    },
    {
        "step": Step.TOTAL_FIELD_ANOMALY,
        "func": step_total_field_anomaly,
        "kwargs": lambda ctx: {
            "filter_type": ctx["filter_type"],
            "filt_window" : ctx["filt_window"],
            "cutoff_freq" : ctx["cutoff_freq"]
        },
        "condition": lambda ctx: True
    },
    {
        "step": Step.SAVE,
        "func": step_save,
        "kwargs": lambda ctx: {
            "result_filepath": ctx["result_filepath"],
            "data_filepath": ctx["data_filepath"],
            "external_gnss_filepath": ctx["external_gnss_filepath"],
            "filter_type": ctx["filter_type"],
            "filt_window": ctx["filt_window"],
            "cutoff_freq": ctx["cutoff_freq"]
        },
        "condition": lambda ctx: True
    },
]

class PipelineState:
    def __init__(self):
        self.data_original = None
        self.data_own = None
        self.data_final = None


def run_pipeline(pipeline, state, context, logger, progress_callback):
    for entry in pipeline:
        if not entry["condition"](context):
            continue

        step = entry["step"]
        func = entry["func"]
        kwargs = entry["kwargs"](context)

        with step.run(logger, progress_callback, kwargs):
            res = func(state, **kwargs)
        if res:
            logger.debug(f"\n{res.df}\n")


def preprocess(
    data_filepath,
    delim,
    systeminfo_filepath,
    result_filepath,
    datetime_specified,
    external_gnss_filepath=None,
    track_max_angle=None,
    track_min_len=None,
    fc_max_deviation=None,
    use_facing=None,

    geom_offset=None,
    s1_offset=None,
    s2_offset=None,
    alt=None,

    filter_type=None,
    window_m=None,
    cutoff_freq=None,
    progress_callback=None,
):

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    logger = logging.getLogger(__name__)

    state = PipelineState()

    context = {
        "data_filepath": data_filepath,
        "result_filepath": result_filepath,
        "delim": delim,
        "systeminfo_filepath": systeminfo_filepath,
        "datetime_specified": datetime_specified,
        "external_gnss_filepath": external_gnss_filepath,
        "state": state,

        "tr_seg_max_angle_diff": 20 if track_max_angle is None else track_max_angle,
        "tr_seg_min_segment_len": 10 if track_min_len is None else track_min_len,
        "fc_max_deviation": 90 if fc_max_deviation is None else fc_max_deviation,
        "use_facing": True if use_facing is None else use_facing,

        "geom_offset": 0 if geom_offset is None else geom_offset,
        "s1_offset": 0.5 if s1_offset is None else s1_offset,
        "s2_offset": 0.5 if s2_offset is None else s2_offset,
        "alt": 0 if alt is None else alt,

        "filter_type": "rolling_med" if filter_type is None else filter_type,
        "filt_window": 15 if window_m is None else window_m,
        "cutoff_freq": 5 if cutoff_freq is None else cutoff_freq,
    }

    run_pipeline(
        PIPELINE,
        state,
        context,
        logger,
        progress_callback,
    )

