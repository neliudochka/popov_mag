import logging
import pathlib
from contextlib import contextmanager
import time
from enum import Enum
import pandas as pd

from .Data.Data import Data
from .Data.DataConfig import DataConfig
from .external_gnss import merge_gnss_into_data, prepare_ext_gnss_data, rename_columns_eg, replace_own_col_with_eg
from .handle_date_time import add_utc_timestamps, add_date
from .interpolation import interpolate_for_lines
from .magnetic_calibration import apply_magnetometer_calibration
from .magnetic_field import add_total_field, add_tfa
from .sensors_offset import add_sensor_positions_by_heading
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

DATA_OWN_CONFIG_PATH = INTERNAL_BASE_DIR / "Data" / "data_configs" / "own.yaml"
DATA_FINAL_CONFIG_PATH = INTERNAL_BASE_DIR / "Data" / "data_configs" / "final.yaml"
DATA_EXTERNAL_GNSS_PATH = INTERNAL_BASE_DIR / "Data" / "data_configs" / "external_gnss.yaml"


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
    EXTERNAL_GNSS = "Applying external GNSS data"
    TRAJECTORY_SEGMENTATION = "Trajectory segmentation. Heading calculation"
    ORIENTATION = "Orientation calculation"
    FACING = "Determining facing direction"
    SIDE_TRACKS = "Calculating the coordinates of side tracks"
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


def get_data_with_config(data_path, config_path, delimiter=None) -> Data:
    dc = DataConfig.from_yaml(config_path)
    if data_path:
        data = Data.from_file(data_path, dc, delimiter=delimiter)
        if data.df.empty:
            raise ValueError(f"File {data_path} is empty")
        return data
    return Data.from_empty(data_path, dc, delimiter=delimiter)


def step_download(state, *, data_filepath, delim):
    state.data_own = get_data_with_config(
        data_filepath, DATA_OWN_CONFIG_PATH, delimiter=delim
    )
    return state.data_own


def step_calibrate(state, *, systeminfo_filepath):
    apply_magnetometer_calibration(systeminfo_filepath, state.data_own)
    return state.data_own


def step_timestamps(state, *, datetime_specified):
    if datetime_specified is not None or not state.data_own.has_date():
        add_date(state.data_own, datetime_specified)
    add_utc_timestamps(state.data_own)
    return state.data_own


def step_external_gnss(state, *, external_gnss_filepath):
    external_data = get_data_with_config(
        external_gnss_filepath, DATA_EXTERNAL_GNSS_PATH
    )
    prepare_ext_gnss_data(external_data)
    rename_columns_eg(external_data)
    merged_df = merge_gnss_into_data(state.data_own, external_data, max_diff_ms=5)
    state.data_own.df = replace_own_col_with_eg(merged_df, state.data_own.column_names, external_data.column_names)
    return state.data_own


def step_trajectory_segmentation(state, max_angle_diff, min_segment_len):
    extract_tracks(state.data_own, max_angle_diff, min_segment_len)
    if not state.data_own.df[state.data_own.column_names.track_id_col].notna().any():
        raise ValueError("No tracks found")
    return state.data_own


def step_orientation(state):
    # check if there is enough acc data
    data = state.data_own
    accX, accY, accZ = (data.df[data.column_names.acc_cols[0]],
                        data.df[data.column_names.acc_cols[1]],
                        data.df[data.column_names.acc_cols[2]])
    accxna, accyna, acczna = accX.isna(), accY.isna(), accZ.isna()
    acs = state.data_own.column_names.acc_cols
    dgp = state.data_own.df.groupby([state.data_own.column_names.track_id_col])
    accx_c, accy_c, accz_c = dgp[acs[0]].count(), dgp[acs[1]].count(), dgp[acs[2]].count()
    min_dp = 50 # for 10 meters
    if ((accxna.any() or accyna.any() or acczna.any())
            and
            ((accx_c < min_dp).any() or (accy_c < min_dp).any() or (accz_c < min_dp).any())):
        # no tilt correction
        yaws_s1 = calculate_yaw(state.data_own, 1)
        yaws_s2 = calculate_yaw(state.data_own, 2)
    else:
        # apply tilt correction
        orig = state.data_own.df
        clean_df = state.data_own.df.dropna(subset=acs, how='any')
        state.data_own.df = clean_df
        yaws_s1 = calculate_tilted_yaw(state.data_own, 1)
        yaws_s2 = calculate_tilted_yaw(state.data_own, 2)
        state.data_own.df = orig

        yaws_s1 = pd.Series(yaws_s1, index=clean_df.index)
        yaws_s2 = pd.Series(yaws_s2, index=clean_df.index)

    y1na = yaws_s1.isna()
    y2na = yaws_s2.isna()
    m = ""
    if y1na.any():
        m += f"{state.data_own.column_names.yaw_s1_col} has NaN entries: {yaws_s1.index[y1na]}\n"
    if y2na.any():
        m += f"{state.data_own.column_names.yaw_s2_col} has NaN entries: {yaws_s2.index[y2na]}"
    if len(m):
        raise ValueError(m)

    state.data_own.df.loc[yaws_s1.index, state.data_own.column_names.yaw_s1_col] = yaws_s1
    state.data_own.df.loc[yaws_s2.index, state.data_own.column_names.yaw_s2_col] = yaws_s2

    state.data_own.df[state.data_own.column_names.yaw_track_col] = get_yaw_of_track(state.data_own)

    # add declination
    date = state.data_own.get_date()
    lat = state.data_own.df[state.data_own.column_names.lat_col].dropna().iloc[0]
    lon = state.data_own.df[state.data_own.column_names.lon_col].dropna().iloc[0]
    if state.data_own.df[state.data_own.column_names.alt_col].notna().any():
        h = state.data_own.df[state.data_own.column_names.alt_col].dropna().iloc[0]
        B_ref_angle = get_B_reference(date, lat, lon, h)
    else:
        B_ref_angle = get_B_reference(date, lat, lon)
    state.data_own.df[state.data_own.column_names.yaw_track_col] = (get_yaw_of_track(state.data_own) + B_ref_angle) % 360

    # check results
    yt_col = state.data_own.column_names.yaw_track_col
    tid_col = state.data_own.column_names.track_id_col

    mask = (state.data_own.df
           .dropna(subset=[tid_col])
           .groupby(tid_col)[yt_col]
           .apply(lambda x: x.isna().any()))
    tracks_no_yaw = mask[mask].index.to_list()
    if tracks_no_yaw:
         raise ValueError(f"No yaw is found for tracks {tracks_no_yaw}")

    return state.data_own


def step_facing(state, max_deviation):
    ft = get_facing_tracks(state.data_own, max_deviation)
    tf_col = state.data_own.column_names.track_facing_col
    tid_col = state.data_own.column_names.track_id_col
    state.data_own.add_column(tf_col, ft)
    ids = state.data_own.df[state.data_own.df[tf_col] == 0][tid_col].unique()
    if len(ids) > 0:
        raise ValueError(f"Impossible to determine the direction of movement in tracks: {ids} (track's facing = 0). Change fc_max_deviation or track_max_angle parameters.")
    return state.data_own


def step_side_tracks(state, use_facing):
    state.data_final = get_data_with_config(None, DATA_FINAL_CONFIG_PATH)
    state.data_final = add_sensor_positions_by_heading(state.data_own, state.data_final.config)
    state.data_final.gen_line_id()
    return state.data_final


def step_gnss_interpolation(state):
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
        "step": Step.CALIBRATE,
        "func": step_calibrate,
        "kwargs": lambda ctx: {
            "systeminfo_filepath": ctx["systeminfo_filepath"],
        },
        "condition": lambda ctx: ctx["systeminfo_filepath"] is not None,
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
        "step": Step.EXTERNAL_GNSS,
        "func": step_external_gnss,
        "kwargs": lambda ctx: {
            "external_gnss_filepath": ctx["external_gnss_filepath"],
        },
        "condition": lambda ctx: ctx["external_gnss_filepath"] is not None,
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
        "step": Step.ORIENTATION,
        "func": step_orientation,
        "kwargs": lambda ctx: {},
        "condition": lambda ctx: True,
    },
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
            "use_facing": ctx["use_facing"]
        },
        "condition": lambda ctx: True,
    },
    {
        "step": Step.GNSS_INTERPOLATION,
        "func": step_gnss_interpolation,
        "kwargs": lambda ctx: {},
        "condition": lambda ctx: True
    },
    {
        "step": Step.TOTAL_FIELD,
        "func": step_total_field,
        "kwargs": lambda ctx: {},
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

