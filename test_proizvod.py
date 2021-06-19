from unittest import mock
import pytest

from datetime import datetime

from proizvod import prepare_train_set


@pytest.mark.parametrize(
    "logs_path, session_length, window_size, max_duration, expect_error",
    [
        ("other_user_logs", 0, 2, 30, ValueError),
        ("other_user_logs", 4, -1, 30, TypeError),
        ("some_directory", 4, 2, 30, FileNotFoundError),
        ("other_user_logs", 4, 2.2, 30, TypeError),
    ],
)
def test_validation(
    logs_path, session_length, window_size, max_duration, expect_error
):
    with mock.patch(
        "os.listdir", return_value=["user0010.csv"]
    ), pytest.raises(expect_error):
        prepare_train_set(logs_path, session_length, window_size, max_duration)


def test_default_work():
    with mock.patch("os.listdir", return_value=["user0010.csv"]):
        res = prepare_train_set("other_user_logs", 4, 2, 30)
    assert res["site01"][0] == "vk.com"
    assert res["site02"][1] is None
    assert res["time04"][2] == datetime(2013, 11, 15, 11, 40, 35)


def test_empty_file():
    with mock.patch(
        "os.listdir", return_value=["userEMPTY.csv"]
    ), pytest.raises(StopIteration):
        prepare_train_set("other_user_logs", 4, 2, 30)


@pytest.mark.parametrize(
    "given, expect_error",
    [
        ("userIP.csv", ValueError),
        ("userDATE.csv", ValueError),
        ("userSITE.csv", IndexError),
    ],
)
def test_bad_file(given, expect_error):
    with mock.patch("os.listdir", return_value=[given]), pytest.raises(
        expect_error
    ):
        prepare_train_set("other_user_logs", 4, 2, 30)
