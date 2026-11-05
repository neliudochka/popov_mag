import numpy as np
import math
from ahrs.utils import WMM
from numpy.ma.core import arctan2
from pandas import Series
from scipy.stats import circmean

from .Data.Data import Data
from .trajectory_segmentation import get_angle_of_track


def calculate_tilted_yaw(data: Data, sid: int):
    if sid == 1:
        Bx, By, Bz = (data.df[data.column_names.mag_cols1[0]],
                      data.df[data.column_names.mag_cols1[1]],
                      data.df[data.column_names.mag_cols1[2]])
    elif sid == 2:
        Bx, By, Bz = (data.df[data.column_names.mag_cols2[0]],
                      data.df[data.column_names.mag_cols2[1]],
                      data.df[data.column_names.mag_cols2[2]])
    else:
        raise ValueError(f"Sid (sensor id) must be equal to 1 or to 2, but got {sid}")


    Bxna, Byna, Bzna = Bx.isna(), By.isna(), Bz.isna()
    if Bxna.any() or Byna.any() or Bzna.any():
        raise ValueError(f"{data.column_names.mag_cols1} has NaN entries: {Bx.index[Bxna]}, {By.index[Byna]}, {Bz.index[Bzna]}")

    accX, accY, accZ = (data.df[data.column_names.acc_cols[0]].rolling(window=10, min_periods=1, center=True).median(),
                        data.df[data.column_names.acc_cols[1]].rolling(window=10, min_periods=1, center=True).median(),
                        data.df[data.column_names.acc_cols[2]].rolling(window=10, min_periods=1, center=True).median())

    accxna, accyna, acczna = accX.isna(), accY.isna(), accZ.isna()
    if accxna.any() or accyna.any() or acczna.any():
        raise ValueError(f"{data.column_names.acc_cols} has NaN entries: {accX.index[accxna]},\n {accY.index[accyna]},\n {accZ.index[acczna]}\n")

    acc_norm = np.sqrt(accX**2 + accY**2 + accZ**2)
    acc_norm = np.where(acc_norm == 0, 1.0, acc_norm)

    ax_n = accX / acc_norm
    ay_n = accY / acc_norm
    az_n = accZ / acc_norm

    pitch = np.arcsin(-ax_n)
    roll = np.arctan2(ay_n, az_n)

    pitch_na, roll_na = np.isnan(pitch), np.isnan(roll)

    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_r, sin_r = np.cos(roll), np.sin(roll)

    mx_comp = Bx * cos_p + Bz * sin_p
    my_comp = Bx * sin_r * sin_p + By * cos_r - Bz * sin_r * cos_p

    mx_na, my_na = mx_comp.isna(), my_comp.isna()

    if mx_na.any() or my_na.any():
        raise ValueError(
            f"Computed magnetic components contain NaNs.\n"
            f"mx_comp NaNs at indices: {mx_comp.index[mx_na]}\n"
            f"my_comp NaNs at indices: {my_comp.index[my_na]}"
        )

    headings_rad = np.arctan2(-my_comp, mx_comp)

    return np.degrees(headings_rad) % 360


def calculate_yaw(data: Data, sid: int):
    if sid == 1:
        Bx, By = data.df[data.column_names.mag_cols1[0]], data.df[data.column_names.mag_cols1[1]]
    elif sid == 2:
        Bx, By = data.df[data.column_names.mag_cols2[0]], data.df[data.column_names.mag_cols2[1]]
    else:
        raise ValueError(f"Sid (sensor id) must be equal to 1 or to 2, but got {sid}")

    date = data.get_date()
    ind = data.df[data.column_names.lat_col].first_valid_index()
    lat = data.df[data.column_names.lat_col].loc[ind]
    lon = data.df[data.column_names.lon_col].loc[ind]
    h = data.df[data.column_names.alt_col].loc[ind]

    B_body_angle = np.degrees(np.arctan2(By, Bx)) % 360
    if not math.isnan(h):
        B_ref_angle = get_B_reference(date, lat, lon, h)
    else:
        B_ref_angle = get_B_reference(date, lat, lon)
    # return (B_ref_angle-B_body_angle) % 360
    return (-B_body_angle) % 360


def get_B_reference(date, lat, lon, h=0.0):
    wmm = WMM(date=date, latitude=lat, longitude=lon, height=h if not None else 0.0)
    B_r_x = wmm.X
    B_r_y = wmm.Y
    return np.degrees(arctan2(B_r_y, B_r_x)) % 360


def angle_diff(a1, a2):
    """
    Compute the shortest difference between angles in degrees (0-360).
    Returns values in [0, 180].
    """
    diff = (a1 - a2) % 360
    return np.minimum(diff, 360 - diff)


def get_facing_point(heading_angle, yaw_angle, max_deviation):
    """
    :param heading_angle:
    :param yaw_angle:
    :param max_deviation:
    :return:
    1  - facing forward
    -1 - facing backward
    0  - neither
    """
    diff = angle_diff(heading_angle, yaw_angle)
    res = np.zeros_like(diff, dtype=int)
    res[diff <= max_deviation] = 1
    res[diff >= 180 - max_deviation] = -1
    return res


def get_facing_track(facings):
    counts = Series(facings).value_counts(normalize=True)
    # check the most popular
    if counts.iloc[0] > 0.5:
        return int(counts.index[0])
    else:
        return 0  # або 0, якщо не впевнений


def get_facing_tracks(data, max_deviation=30):
    fts = []
    for t_n in range(1, data.num_tracks()+1):
        df_by_t = data.df[data.df[data.column_names.track_id_col] == t_n].reset_index(drop=True)
        track_facings = get_facing_point(df_by_t[data.column_names.heading_track_col], df_by_t[data.column_names.yaw_track_col], max_deviation)
        fts.extend([get_facing_track(track_facings)]*len(df_by_t))
    return fts


def get_yaw_of_track(data: Data):
    yaw_of_track_s1 = get_angle_of_track(data, data.column_names.yaw_s1_col)
    yaw_of_track_s2 = get_angle_of_track(data, data.column_names.yaw_s2_col)

    data.df["yaw of tr1"] = data.df[data.column_names.track_id_col].map(yaw_of_track_s1)
    data.df["yaw of tr2"] = data.df[data.column_names.track_id_col].map(yaw_of_track_s2)

    # data.df["yaw of tr1"] = yaw_of_track_s1
    # data.df["yaw of tr2"] = yaw_of_track_s2
    combined_yaw = np.column_stack((data.df["yaw of tr1"], data.df["yaw of tr2"]))

    # 2. Compute circular mean along the columns axis (axis=1)
    # Set low/high bounds. If your yaw is in degrees, use low=-180, high=180 (or 0 to 360)
    # If your yaw is in radians, use low=-np.pi, high=np.pi (or 0 to 2*np.pi)
    return circmean(combined_yaw, high=360, low=0, axis=1)


