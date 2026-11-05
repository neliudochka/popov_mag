import os
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from mdd_processing.preprocessing.Data.ColumnNames import ColumnNames, build_column_rename_map
from mdd_processing.preprocessing.Data.Data import gen_column_names, Data, get_to_save_col
from mdd_processing.preprocessing.Data.DataConfig import DataConfig, Column


# ---------- shared test config ----------

cn = ColumnNames()
cn.timestamp_col = "timestamp_col"
cn.lat_col = "lat_col"

conf1 = DataConfig(
    delimiter=",",
    columns={
        cn.timestamp_col: Column(
            name="timestamp_col",
            type="timestamp",
            unit="ms",
            required=True,
            use=True,
            save=False,
        ),
        cn.lat_col: Column(
            name="lat_col",
            type="piu",
            unit="1",
            required=True,
            use=True,
            save=True,
        ),
    },
)

conf2 = DataConfig(
    delimiter=",",
    columns={
        cn.timestamp_col: Column(
            name="TickyTacky",
            type="timestamp",
            unit="ms",
            required=True,
            use=True,
            save=True,
        )
    },
)

c2_t_cn = conf2.columns[cn.timestamp_col].name


@pytest.fixture
def data_obj():
    df = pd.DataFrame(
        {
            "timestamp_col": [1, 2],
            "lat_col": [10, 20],
        }
    )
    data = Data(df, filepath="filepath", config=conf1)
    data.column_names = gen_column_names(conf1)
    return data


def test_replace_config(data_obj):
    data_obj.replace_config(conf2)

    assert data_obj.config == conf2
    assert c2_t_cn in data_obj.df.columns
    assert "timestamp_col" not in data_obj.df.columns
    assert "lat_col" not in data_obj.df.columns
    assert data_obj.column_names.timestamp_col == "TickyTacky"


def test_save(data_obj):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "output.csv")

        data_obj.save(new_filepath=path)

        assert os.path.exists(path)

        saved_df = pd.read_csv(path)
        expected_cols = ["lat_col"]

        assert list(saved_df.columns) == expected_cols

        pd.testing.assert_frame_equal(
            saved_df,
            data_obj.df[expected_cols].reset_index(drop=True),
        )


def test_build_column_rename_map():
    src = ColumnNames(
        timestamp_col="Timestamp [ms]",
        acc_cols=["AccX [g]", "AccY [g]", "AccZ [g]"],
        mag_cols1=["Mag-X [nT]", "Mag-Y [nT]", "Mag-Z [nT]"],
        mag_cols2=["B2x [nT]", "B2y [nT]", "B2z [nT]"],
        lat_col="Latitude [°]",
        lon_col="lon",
    )

    dst = ColumnNames(
        timestamp_col="Timestamp [ms]",
        mag_cols1=["B1x [nT]", "B1y [nT]", "B1z [nT]"],
        mag_cols2=["B2x [nT]", "B2y [nT]", "B2z [nT]"],
        lat_col="Latitude [°]",
        total_field_col="tf",
    )

    rename_map, drop_col = build_column_rename_map(src, dst)

    assert rename_map == {
        "Mag-X [nT]": "B1x [nT]",
        "Mag-Y [nT]": "B1y [nT]",
        "Mag-Z [nT]": "B1z [nT]",
    }

    assert drop_col == [
        "AccX [g]",
        "AccY [g]",
        "AccZ [g]",
        "lon",
    ]


def test_build_column_rename_map_2():
    rename_map, drop_col = build_column_rename_map(
        gen_column_names(conf1),
        gen_column_names(conf2),
    )

    assert rename_map == {
        "timestamp_col": "TickyTacky",
    }

    assert drop_col == ["lat_col"]


def test_get_to_save_col():
    test_dir = Path(__file__).parent
    config_path = test_dir / "t_default.yaml"

    dc = DataConfig.from_yaml(config_path)
    res = get_to_save_col(dc)

    expected = [
        "Latitude [°]",
        "Track ID",
        "B1x [nT]",
        "B1y [nT]",
        "B1z [nT]",
        "Test_vector",
        "Test_vector2",
        "Test_vector3",
    ]

    assert res == expected
