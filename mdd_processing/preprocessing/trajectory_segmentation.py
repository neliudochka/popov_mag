import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic
from scipy.stats import circmean, mode
from .Data.Data import Data


def compute_geodesic_heading(lat1, lon1, lat2, lon2):
    """Computes the geodesic azimuth (more accurate, WGS84 ellipsoid)."""
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return g['azi1'] % 360


def planar_heading(x1, y1, x2, y2):
    """Computes azimuth in degrees using projected (planar) coordinates."""
    return (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360) % 360


def heading_diff(h1, h2):
    """Returns the smallest angular difference between two headings."""
    return min(abs(h1 - h2), 360 - abs(h1 - h2))


def calculate_heading(df, start, end, lat_col='lat', lon_col='lon'):
    if end - start < 1:
        raise ValueError("End index must be greater than start index by at least 1.")
    h_ref = compute_geodesic_heading(df.iloc[start][lat_col], df.iloc[start][lon_col],
                            df.iloc[start+1][lat_col], df.iloc[start+1][lon_col])
    h_curr = compute_geodesic_heading(df.iloc[end][lat_col], df.iloc[end][lon_col],
                             df.iloc[end+1][lat_col], df.iloc[end+1][lon_col])
    return h_ref, h_curr


def segment_by_heading_incremental(df, lat_col, lon_col, max_angle_diff=50, min_segment_len=10):
    """
    Performs trajectory segmentation using an incremental greedy approach.
    Extends the segment one point at a time until the heading constraint is violated.
    """
    n = len(df)

    track_ids = np.full(n, np.nan)
    headings = np.full(n, np.nan)

    i = 0
    track = 1   #start count tracks from 1
    while i < n-1:
        j = i + 1
        while j < n-1:
            h_ref, h_curr = calculate_heading(df, i, j, lat_col, lon_col)
            if i == j - 1:
                headings[i] = h_ref
            headings[j] = h_curr
            hd = heading_diff(h_ref, h_curr)

            if hd > max_angle_diff:
                break

            j += 1
        if check_len_segment(df[lat_col].iloc[i], df[lon_col].iloc[i],
                             df[lat_col].iloc[j], df[lon_col].iloc[j],
                             min_segment_len):
            track_ids[i:j] = track
            track += 1
        i = j
    return track_ids, headings


def check_len_segment(lat1, lon1, lat2, lon2, min_len):
    geo = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    dist = geo['s12'] # in meters
    if dist >= min_len:
        return True
    return False


def calc_mean_angle(heading):
    angles = np.deg2rad(heading)
    cm = circmean(angles)
    return np.rad2deg(cm)


def normalize_angle_to180(angle):
    return np.mod(angle, 180)


def get_tracks_to_keep(data: Data, max_dev = 5):
    track_headings_to_keep = {}
    track_heading = get_angle_of_track(data, data.column_names.heading_col)

    track_headings_180 = normalize_angle_to180(np.array(list(track_heading.values()))).astype(int)

    heading_mode = mode(track_headings_180, axis=None).mode

    for track_id in track_heading.keys():
        if heading_diff(heading_mode, track_headings_180[track_id - 1]) <= max_dev:
            track_headings_to_keep[track_id] = track_heading[track_id]
    return track_headings_to_keep


def get_angle_of_track(data: Data, angle_col: str):
    track_angle = {}
    for track_id in range(1, data.num_tracks() + 1):
        m = (data.df[data.df[data.column_names.track_id_col] == track_id][angle_col]).dropna()
        track_angle[track_id] = calc_mean_angle(m)
    return track_angle

def heading_based_removal(data: Data, max_dev = 5):
    tracks_to_keep = get_tracks_to_keep(data, max_dev)
    data.df = data.df[
        data.df[data.column_names.track_id_col].isin(tracks_to_keep.keys())
    ]
    data.df[data.column_names.heading_track_col] = data.df[data.column_names.track_id_col].map(tracks_to_keep)
    data.renumber_tracks()
    data.df = data.df.reset_index(drop=True)


def extract_tracks(data: Data, max_angle_diff=30, min_segment_len=10):
    # df with indexes left
    clean_df = data.df.dropna(subset=[data.column_names.lat_col, data.column_names.lon_col])
    track_ids, headings = segment_by_heading_incremental(clean_df,
                                                         data.column_names.lat_col,
                                                         data.column_names.lon_col,
                                                         max_angle_diff,
                                                         min_segment_len)

    track_ids = pd.Series(track_ids, index=clean_df.index)
    headings = pd.Series(headings, index=clean_df.index)
    data.df.loc[track_ids.index, data.column_names.track_id_col] = track_ids
    data.df.loc[headings.index, data.column_names.heading_col] = headings

    fill_nan_trackid_heading(data)
    data.df = data.df.dropna(subset=[data.column_names.track_id_col]).reset_index(drop=True)
    heading_based_removal(data, 5)
    merge_parts_of_one_track(data, max_angle_diff, min_segment_len)
    data.renumber_tracks()


def merge_parts_of_one_track(data, max_angle_diff, min_segment_len):
    df = data.df
    track_heading = get_angle_of_track(data, data.column_names.heading_col)
    track_group = {tid: 1 if angle < 180 else 2
                   for tid, angle in track_heading.items()}
    tg_last = None
    tid_last = None
    for tid, tg in track_group.items():
        if tid_last is None:
            tg_last = tg
            tid_last = tid
            continue
        if tg == tg_last:
            # to cut out overlay
            ind_to_delete = []
            df_tid = df[df[data.column_names.track_id_col].isin([tid_last, tid])]
            df_tid_clear = df[df[data.column_names.track_id_col].isin([tid_last, tid])].dropna(
                subset=[data.column_names.lat_col, data.column_names.lon_col])

            flag_ind = 1
            prev_ind = df_tid_clear.loc[
                df_tid_clear[data.column_names.track_id_col] == tid_last
                ].index[-1]
            curr_ind = df_tid_clear.loc[
                df_tid_clear[data.column_names.track_id_col] == tid
                ].index[0]
            prev_point = df_tid_clear.loc[prev_ind]
            curr_point = df_tid_clear.loc[curr_ind]
            distance = geodesic(
                (prev_point[data.column_names.lat_col],
                 prev_point[data.column_names.lon_col]),
                (curr_point[data.column_names.lat_col],
                 curr_point[data.column_names.lon_col])
            ).meters

            while distance < min_segment_len and len(
                    df_tid_clear[df_tid_clear[data.column_names.track_id_col] == tid_last]) > 1:
                track_ids, _ = segment_by_heading_incremental(df_tid_clear,
                                                              data.column_names.lat_col,
                                                              data.column_names.lon_col,
                                                              max_angle_diff,
                                                              min_segment_len)
                track_ids = pd.Series(track_ids, index=df_tid_clear.index)

                if track_ids.nunique() == 1:
                    df.loc[
                        df[data.column_names.track_id_col] == tid,
                        data.column_names.track_id_col] = tid_last
                    df = df.drop(ind_to_delete)
                    tid = tid_last
                    break

                if flag_ind == 1:
                    new_prev_ind = df_tid_clear.loc[
                        df_tid_clear[data.column_names.track_id_col] == tid_last
                        ].index[-2]
                    ind_to_delete.extend(range(new_prev_ind + 1, prev_ind + 1))
                    df_tid = df_tid.drop(range(new_prev_ind + 1, prev_ind + 1))
                    df_tid_clear = df_tid_clear.drop(prev_ind)

                    # update values
                    prev_ind = new_prev_ind
                    prev_point = df_tid_clear.loc[prev_ind]
                else:
                    new_curr_ind = df_tid_clear.loc[
                        df_tid_clear[data.column_names.track_id_col] == tid
                        ].index[1]
                    ind_to_delete.extend(range(curr_ind, new_curr_ind))
                    df_tid = df_tid.drop(range(curr_ind, new_curr_ind))
                    df_tid_clear = df_tid_clear.drop(curr_ind)

                    # update values
                    curr_ind = new_curr_ind
                    curr_point = df_tid_clear.loc[curr_ind]

                flag_ind *= -1
                distance = geodesic(
                    (prev_point[data.column_names.lat_col],
                     prev_point[data.column_names.lon_col]),
                    (curr_point[data.column_names.lat_col],
                     curr_point[data.column_names.lon_col])
                ).meters

        tid_last = tid
        tg_last = tg
    data.df = df


def fill_nan_trackid_heading(data: Data):
    track_col = data.column_names.track_id_col
    heading_col = data.column_names.heading_col
    for tr_id in data.df[track_col].dropna().unique():
        mask = data.df[track_col] == tr_id
        idx = data.df.index[mask]

        data.df.loc[idx[0]:idx[-1], track_col] = tr_id
        data.df.loc[idx[0]:idx[-1], heading_col] = data.df.loc[idx[0]:idx[-1], heading_col].ffill()


# def segment_by_heading_doubling(df, max_angle_diff=10, lat_col='lat', lon_col='lon'):
#     """
#     Performs trajectory segmentation using the double-and-search method.
#     Combines exponential search with binary search for efficient segment boundary detection.
#     """
#     n = len(df)
#     track_ids = np.zeros(n, dtype=int)
#     i = 0
#     track = 0
#     while i < n - 1:
#         a = 1
#         while i + a < n and test_heading_simple(df, i, i + a, max_angle_diff, lat_col, lon_col):
#             a *= 2
#         lo = i + a // 2
#         hi = min(i + a, n - 1)
#         while lo <= hi:
#             mid = (lo + hi) // 2
#             if test_heading_simple(df, i, mid, max_angle_diff, lat_col, lon_col):
#                 lo = mid + 1
#             else:
#                 hi = mid - 1
#         j = hi
#         track_ids[i:j + 1] = track
#         i = j + 1 #CHECK!!!!
#         track += 1
#     if i == n - 1:
#         track_ids[i] = track
#     df['track_id'] = track_ids
#     return df
