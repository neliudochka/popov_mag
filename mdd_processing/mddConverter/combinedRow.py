from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .magRow import MagRowB

@dataclass
class CombinedRow:
    timestamp: float = 0
    sensor1: Optional['SensorData'] = None
    sensor2: Optional['SensorData'] = None
    acc_x: Optional[float] = None
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None
    temp: float = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    satellites: Optional[int] = None
    quality: Optional[int] = None
    gps_time: Optional[float] = None
    gps_date: Optional[datetime] = None


    def export_to_csv(self, d="\t") -> str:
        def fmt(value, format_str):
            return format(value, format_str) if value is not None else ""

        def fmt_date(date: Optional[datetime], fmt_str: str) -> str:
            return date.strftime(fmt_str) if date else ""

        return (
            f"{self.timestamp}{d}"
            f"{self.sensor1.export_to_csv(d)}"
            f"{self.sensor2.export_to_csv(d)}"
            f"{fmt(self.acc_x, '.3f')}{d}"
            f"{fmt(self.acc_y, '.3f')}{d}"
            f"{fmt(self.acc_z, '.3f')}{d}"
            f"{fmt(self.temp, '.2f')}{d}"
            f"{fmt(self.latitude, '.7f')}{d}"
            f"{fmt(self.longitude, '.7f')}{d}"
            f"{fmt(self.altitude, '.1f')}{d}"
            f"{self.satellites if self.satellites is not None else ''}{d}"
            f"{self.quality if self.quality is not None else ''}{d}"
            f"{fmt(self.gps_time, '.3f')}{d}"
            f"{fmt_date(self.gps_date, '%Y/%m/%d')}{d}"
            f"{fmt_date(self.gps_date, '%H/%M/%S.%f')[:-3]}"
        )

class SensorData:
    def __init__(self, row: MagRowB):
        self.bx = row.Bx
        self.by = row.By
        self.bz = row.Bz

    def export_to_csv(self, d="\t"):
        return f"{self.bx:.2f}{d}{self.by:.2f}{d}{self.bz:.2f}{d}"
