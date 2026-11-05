import struct
from enum import IntEnum

class MagRowType(IntEnum):
    MAG = 219
    ACC = 218
    GPS = 220
    DATA = 221

    @classmethod
    def get_name(cls, value):
        return cls._value2member_map_.get(value, value)

    def __str__(self):
        return self.name


class MagRow:
    def __init__(self, idx: int, data: bytes):
        self.id = idx
        self.data = data
        self.first = data[0]
        self.data_type = data[1]
        self.channel = data[2]
        self.type2 = data[3]
        self.timestamp = struct.unpack_from('<I', data, 4)[0]

    def __str__(self):
        type_name = MagRowType.get_name(self.data_type)
        return f"{self.first}\t{type_name}\t{self.channel}\t{self.type2}\t{self.timestamp}"


class MagRowB(MagRow):
    def __init__(self, idx: int, data: bytes):
        super().__init__(idx, data)
        self.Bx, self.By, self.Bz = struct.unpack_from('<ddd', data, 8)
        self.temp = struct.unpack_from('<f', data, 32)[0]

    def __str__(self):
        return f"{super().__str__()}\t{self.Bx}\t{self.By}\t{self.Bz}\t{self.temp}"


class MagRowAcc(MagRow):
    def __init__(self, idx: int, data: bytes):
        super().__init__(idx, data)
        self.acc_x, self.acc_y, self.acc_z = struct.unpack_from('<ddd', data, 8)
        self.temp = struct.unpack_from('<f', data, 32)[0]

    def __str__(self):
        return f"{super().__str__()}\t{self.acc_x}\t{self.acc_y}\t{self.acc_z}\t{self.temp}"


class MagRowGps(MagRow):
    def __init__(self, idx: int, data: bytes):
        super().__init__(idx, data)
        self.latitude, self.longitude, self.altitude = struct.unpack_from('<ddd', data, 8)
        self.gps_time = struct.unpack_from('<f', data, 32)[0]

    def __str__(self):
        return f"{super().__str__()}\t{self.latitude}\t{self.longitude}\t{self.altitude}\t{self.gps_time}"
