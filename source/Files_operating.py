import numpy as np
import pandas as pd

import shtReader_py.shtRipper as shtRipper


def sht_rewrite(filepath='D:\Edu\Lab\D-alpha-instability-search/data/sht/', filename="sht44168",
                result_path="../data/d-alpha/", result_format="txt"):
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
