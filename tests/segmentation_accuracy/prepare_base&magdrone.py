import os
from pathlib import Path
import re
import pandas as pd

from mdd_processing.mddConverter.main import convert_mdd_to_csv
from mdd_processing.preprocessing.Data.Data import Data
from mdd_processing.preprocessing.Data.DataConfig import DataConfig
from mdd_processing.preprocessing.frequency import reduce_frequency
from mdd_processing.preprocessing.handle_date_time import add_utc_timestamps, add_date, \
    round_timestamp_ms, get_datetime_str_by_filepath

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Vista needs time column with a specific format and with freq in seconds.
# Our data has higher freq, therefore we create fake data just for the sake of working with vista
ANY_DATE = "1977/1/1"
VISTA_COL = "vista_time"

DELIMITER = ","


def get_data(data_path: Path, conf_path: str, skiprows_c : int = None) -> Data:
    dc = DataConfig.from_yaml(conf_path)
    df = pd.read_csv(data_path, delimiter=dc.delimiter, skiprows=skiprows_c)
    return Data.from_df(df, data_path, dc)


def add_vista_col(data: Data) -> Data:
    data.add_column(VISTA_COL)
    data.df[VISTA_COL] = pd.date_range(start='1977/1/1', periods=len(data.df), freq='s')
    return data


def handle_base_timestamps(data: Data, step=200) -> Data:
    add_date(data)
    add_utc_timestamps(data)
    col = data.column_names.utc_timestamp_col
    data.df[col] = round_timestamp_ms(data.df[col], step)
    return data


def prepare_base_data(data: Data) -> Data:
    handle_base_timestamps(data)
    data.df = reduce_frequency(data)
    add_vista_col(data)
    return data


def prepare_magdrone(mag_data: Data, base_data: Data = None):
    mag_data.df = mag_data.df[mag_data.df[mag_data.column_names.sensor_id_col] == 1].copy()
    handle_base_timestamps(mag_data, 5)

    mag_data.df = mag_data.df[mag_data.df[mag_data.column_names.utc_timestamp_col] % 200 == 0].copy()

def get_dest_dir(path: str):
    dir_name = Path(path).name.split("_MD")[0]
    os.makedirs(dir_name, exist_ok=True)

def get_datetime_of_test_dir(dir_):
    return get_datetime_str_by_filepath(dir_.split("_", maxsplit=1)[1])

def get_test_dirs(test_dir):
    test_dirs = []
    for p in test_dir.iterdir():
        if p.name != "mdd":
            try:
                test_dirs.append(p)
            except ValueError as e:
                raise ValueError(f"this {p} should not be in here")
    return test_dirs

def get_mdds_path(mdd_dir: Path):
    return [p for p in mdd_dir.iterdir() if p.suffix == ".mdd"]

def find_mdd_files_to_process(mdds, test_dirs) -> list[Path]:
    datetime_from_dir = [get_datetime_of_test_dir(dt.name) for dt in test_dirs]
    to_process = []
    for mdd in mdds:
        if get_datetime_str_by_filepath(mdd) in datetime_from_dir:
            continue
        to_process.append(mdd)
    return to_process


def get_dir_by_str(path_str):
    try:
        target_dir = Path(path_str)
        if not target_dir.is_dir():
            raise NotADirectoryError(f"{path_str} is not a valid directory")
    except (TypeError, NotADirectoryError) as e:
        raise ValueError(f"Invalid path: {e}")
    return target_dir

def make_dir_for_testcase(parent_dir: Path, datetime_, n: int):
    test_dir_name = f"T{n}_{datetime_}"
    target_dir = Path(f"{parent_dir}/{test_dir_name}")
    target_dir.mkdir()
    return target_dir


def max_t_dir_number(test_dirs):
    max_num = None
    pattern = re.compile(r"^T(\d+)_")  # match 'T<number>_' at start

    for p in test_dirs:
        match = pattern.match(p.name)
        if match:
            num = int(match.group(1))
            if max_num is None or num > max_num:
                max_num = num
    return max_num or 0

def is_csv_exist(csv: Path) -> bool:
    return csv.is_file()

def make_base(testcase_dir: Path, mdd: Path):
    csv_path = Path(f"{mdd.parent}/{mdd.stem}_RF.csv")
    if not is_csv_exist(csv_path):
        csv_path = convert_mdd_to_csv(mdd, save_OF=False)
    base_data = get_data(csv_path, VISTA_CONF)
    prepare_base_data(base_data)
    base_data.df.to_csv(f"{testcase_dir}/base.csv", DELIMITER, index=False)
    print("Base is ready!")
    print("You can proceed to segmenting gt with VISTA")
    print(base_data.df)
    return base_data


def make_magdrone(testcase_dir: Path, mag_path: Path):
    #magdrone
    magdrone_data = get_data(mag_path, MAGDRONE_CONF, 1)
    prepare_magdrone(magdrone_data)
    magdrone_data.df.to_csv(f"{testcase_dir}/magdrone.csv", DELIMITER, index=False)
    print(magdrone_data)

def find_magdrone_files_to_process(mg_files: list[Path], all_tc_dirs: list[Path]):
    result = []

    mg_by_datetime = {
        get_datetime_str_by_filepath(mg): mg
        for mg in mg_files
    }

    for tc in all_tc_dirs:
        if not tc.is_dir():
            continue

        # 1. Skip dirs that already have magdrone
        has_magdrone = any(
            f.is_file() and "magdrone" in f.name
            for f in tc.iterdir()
        )
        if has_magdrone:
            continue

        # 2. Get testcase datetime
        tc_datetime = get_datetime_of_test_dir(tc.name)

        # 3. Check if matching mg_file exists
        mg_file = mg_by_datetime.get(tc_datetime)
        if mg_file:
            result.append((tc, mg_file))

    return result

def get_magdrone_pathes(mdd_dir: Path):
    return [p
            for p in mdd_dir.iterdir()
            if p.name.endswith("rtr.asc")]

if __name__ == "__main__":
    root_dir = Path(__file__).parent.resolve()
    VISTA_CONF = root_dir / "configs/vista_prep.yaml"
    MAGDRONE_CONF = root_dir / "../../mdd_processing/preprocessing/Data/data_configs/magdrone.yaml"

    cur_mdd_dir = root_dir / "Test_Data/mdd"

    mdd_dir_path = get_dir_by_str(cur_mdd_dir)
    testcases_dir = mdd_dir_path.parent

    mdds_pathes = get_mdds_path(mdd_dir_path)
    tc_dirs = get_test_dirs(testcases_dir)
    mdds = find_mdd_files_to_process(mdds_pathes,tc_dirs)
    if len(mdds) == 0:
        print("no mdd file to process")

    max_num = max_t_dir_number(tc_dirs) + 1
    for m in mdds:
        print(f"Start processing {m}")
        tc_dir = make_dir_for_testcase(testcases_dir, get_datetime_str_by_filepath(m), max_num)
        print(f"{tc_dir} was created")
        base = make_base(tc_dir, m)
        max_num+=1

    magdrone_files = get_magdrone_pathes(mdd_dir_path)
    magrone_tc_dir_tc_path = find_magdrone_files_to_process(magdrone_files, get_test_dirs(testcases_dir))
    for d, p in magrone_tc_dir_tc_path:
        make_magdrone(d, p)

