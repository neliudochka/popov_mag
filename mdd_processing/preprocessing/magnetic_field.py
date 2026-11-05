import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from geopy import Point
from scipy.signal import butter, filtfilt

from mdd_processing.preprocessing.Data.Data import Data
from mdd_processing.preprocessing.lag_correction import get_median_step_distance


def total_field(B1, B2, B3):
    return np.round(np.sqrt(B1**2+B2**2+B3**2), 2)


def calculate_residual(tf_series: pd.Series, window_size: int):
    N = len(tf_series)
    trend = []

    half = window_size // 2

    start_window_mean = tf_series.iloc[0:window_size].mean()
    trend.extend([start_window_mean] * half)

    for i in range(half, N - half):
        window = tf_series.iloc[i - half: i + half + 1]
        trend.append(window.median())

    end_window_mean = tf_series.iloc[-window_size:].median()
    trend.extend([end_window_mean] * (N - len(trend)))

    residual = tf_series - pd.Series(trend, index=tf_series.index)
    return residual.round(2), pd.Series(trend, index=tf_series.index).round(2)


def velocity(time_series, lat_series, lon_series):
    vels = []
    for i in range(len(time_series)-1):
        t1 = time_series[i]
        lat1 = lat_series[i]
        lon1 = lon_series[i]
        p1 = Point(lat1, lon1)

        t2 = time_series[i+1]
        lat2 = lat_series[i+1]
        lon2 = lon_series[i+1]
        p2 = Point(lat2, lon2)

        dist = Geodesic.WGS84.Inverse(p1.latitude, p1.longitude, p2.latitude, p2.longitude)['s12']
        vel = dist / (t2-t1)
        vels.append(vel)

    return pd.Series(vels).mean()  # meter / ms

def velocity_data(data, n_elements):
    time_col = data.column_names.utc_timestamp_col
    lat_col = data.column_names.lat_col
    lon_col = data.column_names.lon_col
    line_id_col = data.column_names.line_id_col

    v = []
    for i in range(data.num_tracks()):
        df = data.df[data.df[line_id_col] == i]
        # decided to calculate velocity by the beginning of the track
        if n_elements < len(df):
            n_elements = len(df)
        df = df[:n_elements].reset_index(drop=True)
        # n = len(df)
        # df = df[int(n/2-n_elements/2):int(n/2+n_elements/2)].reset_index(drop=True)
        # df = df.reset_index(drop=True)
        v.append(velocity(df[time_col], df[lat_col], df[lon_col]))  # meter / ms

    if len(v) == 0:
        raise ValueError("No valid tracks with enough data points to compute velocity.")

    return pd.Series(v).mean()
    
def calculate_window_size(data, s = 15, n_elements=12000):
    v = velocity_data(data, n_elements)  # m / ms
    t_delta = s / v  # ms
    time_period = (1 / data.column_freq(data.column_names.timestamp_col)) * 1000
    return int(np.ceil(t_delta/time_period))

def add_total_field(data):
    tf_col = data.column_names.total_field_col
    bx_col, by_col, bz_col = data.column_names.mag_cols1
    data.df[tf_col] = total_field(
        data.df[bx_col],
        data.df[by_col],
        data.df[bz_col],
    )

def smooth_lp(col, cutoff=5, fs=200, order=4):
    """
    col: input signal to be filtered
    cutoff: cutoff frequency of the low-pass filter in Hz
    fs: sampling frequency of the input signal in Hz
    order: order of the Butterworth filter; higher values give a steeper cutoff
    """
    nyq = 0.5 * fs
    low = cutoff / nyq

    b, a = butter(order, low, btype='low')
    return filtfilt(b, a, col)

def apply_median_filt(data):
    s_id = data.column_names.sensor_id_col
    tf_col = data.column_names.total_field_col
    resid_col = data.column_names.residual_field_col

    window_size = calculate_window_size(data)
    df1 = data.df[data.df[s_id] == 1][tf_col]
    df2 = data.df[data.df[s_id] == 2][tf_col]

    residual1, _ = calculate_residual(df1, window_size)
    residual2, _ = calculate_residual(df2, window_size)

    mask1 = data.df[s_id] == 1
    mask2 = data.df[s_id] == 2

    data.df.loc[mask1, resid_col] = residual1
    data.df.loc[mask2, resid_col] = residual2


def add_residual_field(data):
    apply_median_filt(data)
    data.df[data.column_names.residual_field_col] = smooth_lp(data.df[data.column_names.residual_field_col])


def moving_median_n_win(col, window_n):
    res = (pd.Series(col)
        .rolling(window=window_n, center=True, min_periods=1)
        .median()
        .to_numpy())
    return col - res


def med_on_track(data: Data, col_name: str) -> pd.Series:
    return data.df.groupby(data.column_names.line_id_col)[col_name].transform("median")


def mov_med_on_track(data: Data, col_name: str, win_n: int) -> pd.Series:
    return data.df.groupby(data.column_names.line_id_col)[col_name].transform(
        lambda x: x.rolling(window=win_n, center=True, min_periods=1).median()
    )


def get_window_size(d, window_m: float = 15):
    """Get the window size by its length in meters"""
    step = get_median_step_distance(d.df, d.column_names.line_id_col,
                                    d.column_names.lat_col,
                                    d.column_names.lon_col,
                                    d.column_names.step_distance_col)
    return max(1, int(round(window_m / step))+1)


def total_field_anomaly(data: Data, filter_type: str, window_m: float, cutoff: int):
    target_col_name = data.column_names.total_field_col
    TF = data.df[target_col_name]
    if cutoff > 0:
        fs = data.get_frequency()
        TF = smooth_lp(TF, fs=fs, cutoff=cutoff)
    win_n = get_window_size(data, window_m)
    if filter_type in ('const_med', 'constant_med', 'constant_median'):
        return TF - med_on_track(data, target_col_name)
    elif filter_type in ('roll_med', 'rolling_med', 'rolling_median'):
        return TF - mov_med_on_track(data, target_col_name, win_n)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


def add_tfa(data: Data, tfa_col: str = None, filter_type: str = 'rolling', window_m: float = 15, cutoff_freq: int = 5):
    tfa = total_field_anomaly(data, filter_type, window_m, cutoff_freq)
    if tfa_col:
        data.df[tfa_col] = tfa
    else:
        data.df[data.column_names.total_field_anomaly_col] = tfa