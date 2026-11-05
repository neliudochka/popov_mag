import os
from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, Optional, Tuple
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
import yaml

@dataclass
class ColumnBase:
    def merge(self, other: "ColumnBase"):
        """Merge non-None fields from 'other' into this instance."""
        for f in fields(self):
            v = getattr(other, f.name)
            if v is not None:
                setattr(self, f.name, v)
        return self

@dataclass
class Column(ColumnBase):
    name: Optional[str] = None
    type: Optional[str] = None
    unit: Optional[str] = None
    required: Optional[bool] = None
    use: Optional[bool] = None
    save: Optional[bool] = None


class TimestampType(str, Enum):
    UTC = "utc timestamp"                # absolute UTC timestamp
    MISSION = "mission timestamp"        # relative to mission start
    DAY = "day timestamp"                # relative to day start


@dataclass
class DataConfig:
    delimiter: str
    columns: Dict[str, Column]

    def __repr__(self) -> str:
        columns_repr = "\n".join(
            f"\t\t{name}: {col!r}" for name, col in self.columns.items()
        )
        return f'DataConfig:\n\tDelimiter:{self.delimiter!r}\n\tColumns:\n{columns_repr}'

    @classmethod
    def from_yaml(cls, fp) -> Self:
        delim, cols_all_dict = cls.get_config(fp)
        used_cols = cls.drop_unused_col(cols_all_dict)
        return cls(delimiter=delim, columns=used_cols)

    @staticmethod
    def drop_unused_col(cols: Dict[str, Column]) -> Dict[str, "Column"]:
        used_col = {k: v for k, v in cols.items() if getattr(v, "use", False)}

        if not used_col:
            raise Exception("No columns with use=True found")

        return used_col

    @staticmethod
    def get_config(fp) -> Tuple[str, Dict[str, Column]]:
        with open(fp, "r", encoding='utf-8') as f:
            dc = yaml.safe_load(f)

        # Check if yaml is correct

        base_dc = DataConfig
        base_columns = {}
        base_delim = None
        if dc.get("extends"):
            base_file_name = dc["extends"]
            f_dir = os.path.dirname(fp)
            base_delim, base_columns = DataConfig.get_config(f"{f_dir}/{base_file_name}")

        current_columns = {}
        for key, val in dc["columns"].items():
                current_columns[key] = Column(**val)

        merged_col = base_columns.copy()
        for key, col in current_columns.items():
            if key in merged_col:
                merged_col[key].merge(col)
            else:
                merged_col[key] = col

        if "delimiter" in dc and dc["delimiter"] is not None:
            delimiter = dc["delimiter"]
        elif base_dc is not None and base_delim is not None:
            delimiter = base_delim

        # #win encoding fix
        # for col in merged_col.values():
        #     if isinstance(col.name, str):
        #         col.name = col.name.replace("Â°", "°").strip()

        return delimiter, merged_col

