import pytest
from datetime import date, time

from mdd_processing.preprocessing.handle_date_time import get_date_by_str, get_time_by_str, get_date_and_time_by_str, \
    get_date_and_time_by_filepath


def test_get_date_by_str_valid():
    result = get_date_by_str("20240115")
    assert result == date(2024, 1, 15)

def test_get_date_by_str_invalid_format():
    with pytest.raises(ValueError, match="expected YYYYMMDD"):
        get_date_by_str("2024-01-15")

def test_get_date_by_str_invalid_date():
    with pytest.raises(ValueError):
        get_date_by_str("20230230")  # Feb 30 doesn't exist

def test_get_date_by_str_non_numeric():
    with pytest.raises(ValueError):
        get_date_by_str("abcdefgh")

def test_get_date_by_str_non_numeric_and_date():
    result = get_date_by_str("abcdefgh 20250304")
    assert result == date(2025, 3, 4)

def test_get_time_by_str_valid():
    result = get_time_by_str("84541.797")
    assert result == time(8, 45, 41, 797000)

def test_get_time_by_str_valid_no_ms():
    result = get_time_by_str("84541")
    assert result == time(8, 45, 41)

def test_get_time_by_str_invalid_format():
    with pytest.raises(ValueError, match="expected HHMMSS.mmm"):
        get_time_by_str("261390.222")

def test_get_time_by_str_invalid_time():
    with pytest.raises(ValueError):
        get_time_by_str("261390.222")

def test_get_time_by_str_non_numeric():
    with pytest.raises(ValueError):
        get_time_by_str("abcdefgh")


def test_get_date_and_time_by_str_valid():
    result = get_date_and_time_by_str("20240322_121212.797")
    assert result == (date(2024, 3, 22), time( 12, 12, 12, 797000))

def test_get_date_and_time_by_str_invalid_format():
    datetime_str = "2024-03-22-12-12-12-797"
    with pytest.raises(ValueError, match=f"Invalid datetime '{datetime_str}', expected 'YYYYMMDD_HHMMSS.mmm'"):
        get_date_and_time_by_str(datetime_str)

def test_get_date_and_time_by_str_invalid_date():
    with pytest.raises(ValueError, match="expected YYYYMMDD"):
        get_date_and_time_by_str("20240332_121212.797")

def test_get_date_and_time_by_str_invalid_time():
    with pytest.raises(ValueError, match="HHMMSS.mmm"):
        get_date_and_time_by_str("20240322_331212.797")

def test_get_date_and_time_by_str_non_numeric():
    with pytest.raises(ValueError):
        get_date_and_time_by_str("abcdefgh")


def test_get_date_and_time_by_filepath_valid():
    result = get_date_and_time_by_filepath("20250322_121212_MD-R3_#0112_RF.csv")
    assert result == (date(2025, 3, 22), time(12, 12, 12))

def test_get_date_and_time_by_filepath_invalid_format():
    filepath_str = "2024-03-22-12-12-12-797-smth.csv"
    with pytest.raises(ValueError, match=f"Invalid filepath format '{filepath_str}', expected 'YYYYMMDD_HHMMSS_\[\.\.\.\]'"):
        get_date_and_time_by_filepath(filepath_str)

def test_get_date_and_time_by_filepath_invalid_date():
    with pytest.raises(ValueError, match="expected YYYYMMDD"):
        get_date_and_time_by_filepath("20250342_121212_MD-R3_#0112_RF.csv")

def test_get_date_and_time_by_filepath_invalid_time():
    with pytest.raises(ValueError, match="HHMMSS.mmm"):
        get_date_and_time_by_filepath("20250322_721212_MD-R3_#0112_RF.csv")

def test_get_date_and_time_by_filepath_non_numeric():
    with pytest.raises(ValueError):
        get_date_and_time_by_filepath("abcdefgh")



def test_get_date_and_time_by_filepath_extended_path():
    result = get_date_and_time_by_filepath("files_for_test/20250509/084539/20250509_084539_MD-R3_#0112_RF.csv")
    assert result == (date(2025, 5, 9), time(8, 45, 39))



