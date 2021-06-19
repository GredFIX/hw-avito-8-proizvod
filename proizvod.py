import csv
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


def append_to_session(
    file_name: str, session_length: int, session: list
) -> list:
    """
    Дополняет сессию полями None при необходимости и получает user_id
    из регулярного выражения к названию файла
    """
    diff = session_length * 2 - len(session)
    session.extend([None for _ in range(diff)])
    file_name = re.search(r"\d+", file_name)
    session.append(int(file_name.group(0)))
    return session


def read_csv(file_name: str) -> list:
    """
    Считывает csv-файл и меняет поля местами (для удобства)
    """
    with open(file_name, newline="") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
        next(spam_reader)

        return [
            [row[1], datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")]
            for row in spam_reader
        ]


def prepare_train_set(
    logs_path: str, session_length: int, window_size: int, max_duration: int
) -> pd.DataFrame:
    """
    Основная функция из задания
    """
    list_of_files = os.listdir(logs_path)
    max_duration *= 60
    session_list = []

    for file_name in list_of_files:
        list_of_log = read_csv(f"{logs_path}/{file_name}")
        length = len(list_of_log)
        start, end = 0, 0

        while end != length:
            session = list_of_log[start]
            end = start + session_length

            if end > length:
                end = length
            if session_length < 1:
                raise ValueError
            elif session_length == 1:
                start += 1
            else:
                for i in range(start + 1, end):
                    delta_time = list_of_log[i][1] - session[-1]

                    if delta_time.total_seconds() > max_duration:
                        break

                    session.extend(list_of_log[i])

                if i - start + 1 > window_size:
                    start += window_size
                else:
                    start = i
            upd_session = append_to_session(file_name, session_length, session)
            session_list.append(upd_session)

    col = [
        f"{j}{i}"
        for i in range(1, session_length + 1)
        for j in ["site0", "time0"]
    ]
    col.append("user_id")
    df = pd.DataFrame(np.array(session_list), columns=col)

    return df


if __name__ == "__main__":
    prepare_train_set("other_user_logs", 4, 2, 30)
