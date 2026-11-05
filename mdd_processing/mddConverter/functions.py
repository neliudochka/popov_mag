import argparse
from typing import List

from .magRow import  MagRow, MagRowB, MagRowAcc, MagRowType, MagRowGps
from .combinedRow import CombinedRow, SensorData


def create_rows(data: bytes) -> List[MagRow]:
    width = 36
    output = []
    for i in range(0, len(data), width):
        row_data = data[i:i+width]
        if len(row_data) < width:
            continue
        type_ = row_data[1]
        try:
            if type_ == MagRowType.MAG:
                row = MagRowB(i // width, row_data)

            elif type_ == MagRowType.ACC:
                row = MagRowAcc(i // width, row_data)
            elif type_ == MagRowType.GPS:
                row = MagRowGps(i // width, row_data)
            else:
                row = MagRow(i // width, row_data)
            output.append(row)
        except Exception as e:
            print(f"Error parsing row {i // width}: {e}")
    return output


def create_combined_rows(rows: List[MagRow]) -> List[CombinedRow]:
    combined_rows: List[CombinedRow] = []

    for i, row in enumerate(rows):
        if isinstance(row, MagRowB) and row.channel == 1:
            start_idx = max(0, i - 3)
            range_slice = rows[start_idx:start_idx + 6]

            mag2 = next(
                (r for r in range_slice if isinstance(r, MagRowB) and r.timestamp == row.timestamp and r.channel == 2),
                None
            )
            if mag2 is None:
                raise Exception("Sensor 2 data not found")

            acc = next(
                (r for r in range_slice if isinstance(r, MagRowAcc) and abs(r.timestamp - row.timestamp) < 3),
                None
            )

            combined = CombinedRow(
                timestamp=row.timestamp,
                sensor1=SensorData(row),
                sensor2=SensorData(mag2),
                acc_x=acc.acc_x if acc else None,
                acc_y=acc.acc_y if acc else None,
                acc_z=acc.acc_z if acc else None,
                temp=row.temp
            )

            combined_rows.append(combined)

    i_gps = 0
    for gps_row in filter(lambda r: r.data_type == MagRowType.GPS, rows):
        gps = gps_row if isinstance(gps_row, MagRowGps) else None
        if gps is None:
            raise Exception("Incorrect gps row")

        while i_gps < len(combined_rows) and abs(gps.timestamp - combined_rows[i_gps].timestamp) > 3:
            i_gps += 1

        if i_gps == len(combined_rows):
            i_gps = 0
            continue

        cr = combined_rows[i_gps]
        cr.latitude = gps.latitude
        cr.longitude = gps.longitude
        cr.altitude = gps.altitude
        cr.gps_time = gps.gps_time

    if combined_rows:
        first_ts = combined_rows[0].timestamp
        for row in combined_rows:
            row.timestamp -= first_ts

    return combined_rows

def parse_arguments():
    parser = argparse.ArgumentParser(description="Binary Data Parser")
    parser.add_argument("--input", required=True, help="Input binary file path")
    return parser.parse_args()






