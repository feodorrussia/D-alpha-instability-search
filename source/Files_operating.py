import numpy as np
import pandas as pd
import os

import shtReader_py.shtRipper as shtRipper


def clear_space(line) -> None:
    """
    Function to clear tabs & spaces from lines of data
    :param line: line of data
    :return: line w/o needless spaces
    """
    len_l = len(line)
    line = line.replace("\t", " ")
    line = line.replace("  ", " ")
    while len_l > len(line):
        len_l = len(line)
        line = line.replace("  ", " ")
    return line


def read_dataFile(file_path: str, id_file="") -> pd.DataFrame:
    """
    Function to read exported files. It's clearing spaces & unnecessary formatting
    :param file_path: full path w/ filename
    :param id_file: id of reading file (5 digits)
    :return: pd.DataFrame
    """
    new_filename = f'fil_{id_file}.dat'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # удаляем первую строку
    lines.pop(0)
    # удаляем последние 4 строк (с запасом, чтобы не было мусора)
    lines = lines[:-4]
    # удаляем пробелы в начале и конце каждой строки
    lines = [line.strip() + "\n" for line in lines]
    # чистим пробелы
    lines = list(map(clear_space, lines))

    with open(new_filename, 'w') as f:
        f.writelines(lines)

    # Загрузка всех столбцов из файла
    data = pd.read_table(new_filename, sep=" ", names=["t"] + ["ch{}".format(i) for i in range(1, 10)])

    os.remove(new_filename)

    return data.dropna(axis=1)


def sht_rewrite(filepath='D:\Edu\Lab\D-alpha-instability-search/data/sht/', filename="sht44168",
                result_path="../data/d-alpha/", result_format="txt") -> str:
    """
    Function to export D-alpha data from SHT files to txt/csv/dat/... files
    :param filepath: Full path to file
    :param filename: Filename w/o format
    :param result_path: Path to save the result file
    :param result_format: Format of the result file
    :return: ok/Exception
    """
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    try:
        if ".sht" in filename.lower():
            res = shtRipper.ripper.read(filepath + filename)
            filename = filename[:-4]
        else:
            res = shtRipper.ripper.read(filepath + filename + ".SHT")

        data = np.array([res["D-alfa  хорда R=50 cm"]["x"], res["D-alfa  хорда R=50 cm"]["y"]])
        sht_df = pd.DataFrame(data.transpose(), columns=["t", "D-alpha_h50"])
        sht_df.to_csv(result_path + filename + "." + result_format, index=False)
        return "ok"
    except Exception as e:
        return e


def rewrite_sht_fromDir(dir_path, result_path="../data/d-alpha/", result_format="txt") -> None:
    """
    Function to export data from entire dir
    :param dir_path:
    :param result_path:
    :param result_format:
    :return:
    """
    print("|", end="")

    for name in os.listdir(dir_path):
        report = sht_rewrite(filepath=dir_path, filename=name, result_path=result_path, result_format=result_format)

        if report == "ok":
            print(".", end="")
        else:
            print("-", end="")

    print("|")

