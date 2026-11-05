import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from narwhals import String
from pandas import DataFrame

from .Data.Data import Data
from .Data.DataConfig import DataConfig


DIST_CENTER_SENSOR = 0.5


def stack_mag_data_popov(d: Data) -> DataFrame:
    time_col = d.column_names.timestamp_col
    sen_col = d.column_names.sensor_id_col or "Sensor_ID"
    track_id_col = d.column_names.track_id_col
    tf1_col = d.column_names.s1_tf
    tf2_col = d.column_names.s2_tf
    res_mag_col = d.column_names.total_field_col
    id_cols = [c for c in d.df.columns if c not in [tf1_col, tf2_col]]

    out = pd.concat([
        d.df[id_cols + [tf1_col]]
        .rename(columns={tf1_col: res_mag_col})
        .assign(**{sen_col: 1}),
        d.df[id_cols + [tf2_col]]
        .rename(columns={tf2_col: res_mag_col})
        .assign(**{sen_col: 2}),
    ], ignore_index=True)

    return out.sort_values([track_id_col, sen_col, time_col]).reset_index(drop=True)


def stack_mag_data(d: Data) -> DataFrame:
    time_col = d.column_names.timestamp_col
    sen_col = d.column_names.sensor_id_col or "Sensor_ID"
    track_id_col = d.column_names.track_id_col
    mag1_cols = d.column_names.mag_cols1
    mag2_cols = d.column_names.mag_cols2
    res_mag_cols = d.column_names.mag_cols1
    id_cols = [c for c in d.df.columns if c not in [*mag1_cols, *mag2_cols]]

    out = pd.concat([
        d.df[id_cols + list(mag1_cols)]
        .rename(columns={mag1_cols[0]: res_mag_cols[0], mag1_cols[1]: res_mag_cols[1], mag1_cols[2]: res_mag_cols[2]})
        .assign(**{sen_col: 1}),
        d.df[id_cols + list(mag2_cols)]
        .rename(columns={mag2_cols[0]: res_mag_cols[0], mag2_cols[1]: res_mag_cols[1], mag2_cols[2]: res_mag_cols[2]})
        .assign(**{sen_col: 2}),
    ], ignore_index=True)

    return out.sort_values([track_id_col, sen_col, time_col]).reset_index(drop=True)


def compute_sensor_angle(base_angle, sensor_id):
    # first sensor - on the left
    # second sensor - on the right
    sensor_id = int(sensor_id)
    if sensor_id == 1:
        return  (base_angle - 90) % 360
    if sensor_id == 2:
        return (base_angle + 90) % 360

    raise ValueError(f"Sensor ID is either 1 or 2, but got {sensor_id}")


def compute_sensor_angle_with_facing(base_angle, facing, sensor_id):
    angle = compute_sensor_angle(base_angle, sensor_id)
    if facing == 1:
        return angle
    elif facing == -1:
        return (angle + 180 ) % 360
    else:
        raise ValueError(f"Facing must be 1 or -1, got: {facing}")


def compute_sensor_position(angle, facing, lat, lon, sensor_id, dist_center_sensor, use_facing=False, ):
    if use_facing:
        s_angle = compute_sensor_angle_with_facing(angle, facing, sensor_id)
    else:
        s_angle = compute_sensor_angle(angle, sensor_id)

    s_pos = Geodesic.WGS84.Direct(lat, lon, s_angle, dist_center_sensor)

    return s_pos


def compute_sensor_positions(angles, facings, lats, lons, sensor_ids, s1_offset, s2_offset, use_facing=False):
    s_coords = []
    for i in range(len(angles)):
        if sensor_ids[i] == 1:
            s1_pos = compute_sensor_position(angles[i], facings[i], lats[i], lons[i], sensor_ids[i], s1_offset, use_facing)
        elif sensor_ids[i] == 2:
            s1_pos = compute_sensor_position(angles[i], facings[i], lats[i], lons[i], sensor_ids[i], s2_offset, use_facing)
        else:
            raise ValueError(f"Sensor id must be 1 or 2: {sensor_ids[i]}")
        s_coords.append([s1_pos["lat2"], s1_pos["lon2"]])
    return s_coords


def update_mag_sensors_coordinates(data: Data, use_facing: bool, angle_col_name, s1_offset, s2_offset) -> None:
    lats = data.df[data.column_names.lat_col].to_numpy()
    lons = data.df[data.column_names.lon_col].to_numpy()
    angles = data.df[angle_col_name].to_numpy()
    sen_ids = data.df[data.column_names.sensor_id_col].to_numpy()
    facings = None

    if use_facing:
        facings = data.df[data.column_names.track_facing_col].to_numpy()

    s_coords = compute_sensor_positions(angles, facings, lats, lons, sen_ids, s1_offset, s2_offset, use_facing)
    s_coords_arr = np.array(s_coords)
    data.df[data.column_names.lat_col] = pd.Series(s_coords_arr[:, 0])
    data.df[data.column_names.lon_col] = pd.Series(s_coords_arr[:, 1])


def add_sensor_positions(data: Data, data_final_config: DataConfig, use_facing: bool, angle_col_name: String) -> Data:
    data.df = stack_mag_data(data)
    data.replace_config(data_final_config)
    update_mag_sensors_coordinates(data, use_facing, angle_col_name)
    return data


def add_sensor_positions_by_yaw(source_data: Data, data_final_config: DataConfig) -> Data:
    return add_sensor_positions(
        source_data,
        data_final_config,
        angle_col_name=source_data.column_names.yaw_s1_col,
        use_facing=False
    )


def add_sensor_positions_by_heading(source_data: Data, data_final_config: DataConfig) -> Data:
    return add_sensor_positions(
        source_data,
        data_final_config,
        angle_col_name=source_data.column_names.heading_col,
        use_facing=True)

def add_sensor_positions_by_heading_popov(source_data: Data, data_final_config: DataConfig, s1_offset, s2_offset) -> Data:
    return add_sensor_positions_popov(
        source_data,
        data_final_config,
        s1_offset, s2_offset,
        angle_col_name=source_data.column_names.heading_col,
        use_facing=True)

def add_sensor_positions_popov(data: Data, data_final_config: DataConfig, s1_offset, s2_offset, use_facing: bool, angle_col_name: String) -> Data:
    data.df = stack_mag_data_popov(data)
    # data.replace_config(data_final_config)
    update_mag_sensors_coordinates(data, use_facing, angle_col_name, s1_offset, s2_offset)
    return data
