from dataclasses import dataclass, asdict


@dataclass
class ColumnNames:
    utc_timestamp_col: str = None
    timestamp_col: str = None
    acc_cols: tuple[str, str, str] = None
    mag_cols1: tuple[str, str, str] = None
    mag_cols2: tuple[str, str, str] = None
    lat_col: str = None
    lon_col: str = None
    alt_col: str = None
    heading_col: str = None
    track_id_col: str = None
    line_id_col: str = None
    yaw_s1_col: str = None
    yaw_s2_col: str = None
    facing_col: str = None
    track_facing_col: str = None
    sensor_id_col: str = None
    total_field_col: str = None
    total_field_anomaly_col: str = None
    residual_field_col: str = None
    gps_time_col: str = None
    gps_date_col: str = None
    gps_time_col_hh_mm_ss_sss: str = None
    step_distance_col: str = None
    heading_track_col: str = None
    yaw_track_col: str = None
    nmea_col: str = None
    s1_tf: str = None
    s2_tf: str = None
    s2_grad: str = None
    s1_qual: str = None
    s2_qual: str = None


def build_column_rename_map(src: ColumnNames, dst: ColumnNames) -> tuple[dict[str, str], list[str]]:
    """
        Compares two ColumnNames objects to generate a rename map and a drop list,
        handling single strings and lists/tuples of names.

        Args:
            src: The source ColumnNames object.
            dst: The destination ColumnNames object.

        Returns:
            A tuple containing:
            1. rename_map (Dict[str, str]): Mapping of old column name to new name.
            2. col_to_drop (List[str]): List of column names that should be dropped.
        """
    src_dict = asdict(src)
    dst_dict = asdict(dst)

    rename_map = {}
    col_to_drop = list()
    for key, src_val in src_dict.items():
        dst_val = dst_dict[key]
        if src_val is None and dst_val is None:
            continue
        elif src_val is not None and dst_val is None:
            if isinstance(src_val, (list, tuple)):
                col_to_drop.extend(src_val)
            else:
                col_to_drop.append(src_val)
            continue

        if src_val != dst_val:
            if isinstance(src_val, (list, tuple)) and isinstance(dst_val, (list, tuple)):
                if len(src_val) == len(dst_val):
                        rename_map.update(dict(zip(src_val, dst_val)))
                else:
                    raise ValueError(f"Length mismatch for key '{key}': {len(src_val)} != {len(dst_val)}")
            elif isinstance(src_val, str) and isinstance(dst_val, str):
                rename_map[src_val] = dst_val

    return rename_map, col_to_drop

