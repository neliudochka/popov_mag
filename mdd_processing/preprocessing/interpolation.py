import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from geopy import Point

from mdd_processing.preprocessing.Data.Data import Data


def interpolate_for_lines(data: Data):
    line_id_col = data.column_names.line_id_col
    lat_col = data.column_names.lat_col
    lon_col = data.column_names.lon_col
    alt_col = data.column_names.alt_col
    timestamp_col = data.column_names.utc_timestamp_col

    interpolated_parts = []

    for line_id, df_part in data.df.groupby(line_id_col):
        interpolated = interpolate_by_time(
            df_part.copy(),
            lat_col,
            lon_col,
            alt_col,
            timestamp_col
        )
        interpolated_parts.append(interpolated)

    return pd.concat(interpolated_parts, ignore_index=True)

def interpolate_by_time(df, lat_col, lon_col, alt_col, t_col):
    df = df.copy()
    lat_interpolated = []
    lon_interpolated = []
    alt_interpolated = []

    valid_indices = df[lat_col].dropna().index
    if len(valid_indices) == 0:
        return None

    first_valid_index = valid_indices[0]
    last_valid_index = valid_indices[-1]
    df = df.loc[first_valid_index:last_valid_index].reset_index(drop=True)

    valid_indices = df[lat_col].dropna().index
    first_valid_index = valid_indices[0]

    p1_ind_ind=0
    lat = df[lat_col].loc[first_valid_index]
    lon = df[lon_col].loc[first_valid_index]
    alt = df[alt_col].loc[first_valid_index]

    add_lat_lon(lat_interpolated, lon_interpolated, lat, lon)
    alt_interpolated.append(alt)

    p1 = Point(lat, lon)

    for i in range(1, len(df)):
        if p1_ind_ind + 1 >= len(valid_indices):
            break

        if i == valid_indices[p1_ind_ind+1]:
            p1_ind_ind = p1_ind_ind+1
            lat = df[lat_col].loc[i]
            lon = df[lon_col].loc[i]
            alt = df[alt_col].loc[i]

            add_lat_lon(lat_interpolated, lon_interpolated, lat, lon)
            alt_interpolated.append(alt)

            p1 = Point(lat, lon)
            continue

        # find p2
        p1_ind = valid_indices[p1_ind_ind]
        p2_ind = valid_indices[p1_ind_ind+1]

        lat2 = df[lat_col].loc[p2_ind]
        lon2 = df[lon_col].loc[p2_ind]
        alt1 = df[alt_col].loc[p1_ind]
        alt2 = df[alt_col].loc[p2_ind]
        p2 = Point(lat2, lon2)

        # 2.3. Обчислення коефіцієнта часової інтерполяції:
        k = calculate_interpolation_coefficient(df[t_col].loc[i], df[t_col].loc[p1_ind], df[t_col].loc[p2_ind])

        # 2.4. Розрахунок повної відстані між точками P1 та P2
        geo = Geodesic.WGS84.Inverse(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
        total_distance = geo['s12']  # в метрах

        # 2.5. Розрахунок часткової відстані від P1 до шуканої точки P_rtk_interpolated
        partial_distance = k * total_distance

        # 2.6. Визначення азимута від P1 до P2
        azimuth = geo['azi1']

        # 2.7. Обчислення інтерпольованих координат
        interpolated = Geodesic.WGS84.Direct(p1.latitude, p1.longitude, azimuth, partial_distance)

        # Append the interpolated point along with the timestamp
        add_lat_lon(lat_interpolated, lon_interpolated, interpolated['lat2'], interpolated['lon2'])

        if alt1 is None:
            interpolated_alt = None
        else:
            interpolated_alt = np.round((1 - k) * alt1 + k * alt2, 2)
        alt_interpolated.append(interpolated_alt)

    df[lat_col] = lat_interpolated
    df[lon_col] = lon_interpolated
    df[alt_col] = alt_interpolated
    return df

def add_lat_lon(lat_list, lon_list, new_lat,new_lon):
    lat_list.append(round(new_lat, 9))
    lon_list.append(round(new_lon, 9))

def calculate_interpolation_coefficient(t_cur, t1, t2):
    if t2 == t1:
        raise ValueError("t1 і t2 can't be equal")

    if not (t1 <= t_cur <= t2):
        raise ValueError(f"t_cur = {t_cur} is out of bounds [{t1}, {t2}]")

    return (t_cur - t1) / (t2 - t1)


