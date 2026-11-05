import numpy as np
from dataclasses import dataclass
from typing import List
from .Data import Data


@dataclass
class MagSensor:
    korr_matrix: np.ndarray
    offset_korr: np.ndarray

    def __repr__(self):
        return (f"MagSensor(\n"
                f"  korr_matrix=\n{self.korr_matrix},\n"
                f"  offset_korr={self.offset_korr}\n)")


def parse_slash_separated_floats(line: str) -> np.ndarray:
    parts = line.split('/')
    values = [float(p.strip()) for p in parts]
    return np.array(values)


def get_korr_matrix(lines: List[str], i_h: int) -> np.ndarray:
    matrix_lines = lines[i_h + 1: i_h + 4]
    flat_values = np.concatenate([parse_slash_separated_floats(line) for line in matrix_lines])
    korr_matrix_array = flat_values.reshape((3, 3))
    return korr_matrix_array


def get_offset_korr(lines: List[str], i_h: int) -> np.ndarray:
    offset_line = lines[i_h + 1]
    return parse_slash_separated_floats(offset_line)


def parse_mag_sensors(filename: str) -> List[MagSensor]:
    korr_matrix_header = ".KorrMatrix:"
    offset_korr_header = ".OffsetKorr:"

    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    korr_matrices = []
    offset_korrs = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == korr_matrix_header:
            if i + 3 < len(lines):
                korr_matrices.append(get_korr_matrix(lines, i))
                i += 4
            else:
                raise ValueError(f"Not enough lines after {korr_matrix_header} at line {i}")
        elif line == offset_korr_header:
            if i + 1 < len(lines):
                offset_korrs.append(get_offset_korr(lines, i))
                i += 2
            else:
                raise ValueError(f"Not enough lines after {offset_korr_header} at line {i}")
        else:
            i += 1

    if len(korr_matrices) != len(offset_korrs):
        raise ValueError("Mismatch in number of KorrMatrix and OffsetKorr blocks")

    sensors = [MagSensor(k, o) for k, o in zip(korr_matrices, offset_korrs)]
    return sensors

def apply_magnetometer_calibration(systeminfo_filename: str, data: Data):
    mag_sensors = parse_mag_sensors(systeminfo_filename)

    mgs1, mgs2 = mag_sensors[0], mag_sensors[1]

    df_b1_corrected = data.df[list(data.column_names.mag_cols1)] + mgs1.offset_korr
    df_b2_corrected = data.df[list(data.column_names.mag_cols2)] + mgs2.offset_korr

    b1_vals = df_b1_corrected.values
    b2_vals = df_b2_corrected.values

    b1_corrected = (mgs1.korr_matrix @ b1_vals.T).T
    b2_corrected = (mgs2.korr_matrix @ b2_vals.T).T

    data.df[list(data.column_names.mag_cols1)] = np.round(b1_corrected, 2)
    data.df[list(data.column_names.mag_cols2)] = np.round(b2_corrected, 2)


