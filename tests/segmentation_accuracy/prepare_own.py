from pathlib import Path
import pandas as pd
from mdd_processing.preprocessing.Data.Data import Data
from mdd_processing.preprocessing.Data.DataConfig import DataConfig
from mdd_processing.preprocessing.trajectory_segmentation import segment_by_heading_incremental, extract_tracks

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

BASE_FILE_NAME = "base.csv"
DELIMITER = ","


def get_data(data_path: Path, conf_path: Path, delim = DELIMITER) -> Data:
    dc = DataConfig.from_yaml(conf_path)
    dc.delimiter = delim
    df = pd.read_csv(data_path, delimiter=dc.delimiter)
    return Data.from_df(df, data_path, dc)


def prepare_own(data: Data, max_angle_diff, min_segment_len) -> Data:
    extract_tracks(data, max_angle_diff, min_segment_len)
    return data


def get_test_dirs(test_dir):
    test_dirs = []
    for p in test_dir.iterdir():
        if p.name != "mdd":
            try:
                test_dirs.append(p)
            except ValueError as e:
                raise ValueError(f"this {p} should not be in here")
    return test_dirs


def make_own(testcase_dir: Path, base_data: Data):
    # own
    own_data = prepare_own(base_data, 30,10)
    own_data.df.to_csv(f"{testcase_dir}/own.csv", DELIMITER, index=False)
    print("Own is ready!")
    print(own_data.df)


def get_tc_dirs_with_base(dirs: list[Path]):
    to_process = []

    for d in dirs:
        files = {p.name for p in d.iterdir()}

        if BASE_FILE_NAME in files:
            to_process.append(d)

    return to_process


def get_tc_dirs_with_base_and_noown(dirs: list[Path]):
    to_process = []
    for d in dirs:
        files = {p.name for p in d.iterdir()}

        if BASE_FILE_NAME in files and "own.csv" not in files:
            to_process.append(d)
    return to_process


if __name__ == "__main__":
    root_dir = Path(__file__).parent.resolve()
    OWN_CONF = root_dir / "../../mdd_processing/preprocessing/Data/data_configs/own.yaml"

    testcases_dir = root_dir / "Test_Data"
    tc_dirs = get_tc_dirs_with_base(get_test_dirs(testcases_dir))

    if len(tc_dirs) == 0:
        print("nothing to process")

    for tc_dir in tc_dirs:
        print(f"Start processing {tc_dir}")
        base = get_data(Path(f"{tc_dir}/{BASE_FILE_NAME}"), OWN_CONF)
        make_own(tc_dir, base)


