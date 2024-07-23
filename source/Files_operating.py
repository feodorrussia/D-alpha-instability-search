import numpy as np
import pandas as pd
import os

import shtReader_py.shtRipper as shtRipper


def sht_rewrite(filepath='D:\Edu\Lab\D-alpha-instability-search/data/sht/', filename="sht44168",
                result_path="../data/d-alpha/", result_format="txt"):
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


def rewrite_sht_fromDir(dir_path, result_path="../data/d-alpha/", result_format="txt"):
    print("|", end="")

    for name in os.listdir(dir_path):
        report = sht_rewrite(filepath=dir_path, filename=name, result_path=result_path, result_format=result_format)

        if report == "ok":
            print(".", end="")
        else:
            print("-", end="")
