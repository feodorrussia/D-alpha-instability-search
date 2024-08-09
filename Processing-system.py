import os.path
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime

import shtReader_py.shtRipper as shtRipper
from source.Files_operating import read_dataFile, read_sht_dalpha
from source.NN_environment import process_fragments, get_borders, normalise_series, down_to_zero
from source.NN_environment import get_prediction_unet


def init_proc(filename: str, filepath: str, ckpt_v: int):
    
    # filename = "sht44351"
    # filepath = "C:/Users/f.belous/Work/Projects/D-alpha-instability-search/data/sht/"  # D:/Edu/Lab/D-alpha-instability-search/data/sht/
    filepath += "\\"
    F_ID = filename[-5:]
    
    df = read_sht_dalpha(filename, filepath)  # pd.read_csv(signal_path + filename + ".txt", sep=',')  # read_dataFile(interval_path + filename + "_exportGlobus2.dat", F_ID)
    df = df.rename(columns={"D-alpha_h50": "ch1"})
    df["ch1_marked"] = 0
    df["ch1_ai_marked"] = 0
    start_time = time.time()

    # ckpt_v=4
    df["ch1_ai_marked"] = get_prediction_unet(df["ch1"].to_numpy(), ckpt_v=ckpt_v)  # , old=True
    
    df["ch1_marked"] = down_to_zero(np.array(df["ch1_ai_marked"]), edge=0.5)
    df["ch1_marked"] = process_fragments(np.array(df["ch1"]), np.array(df["ch1_marked"]), length_edge=30, scale=1.5)  # old version: length_edge=20, , scale=0
    
    to_pack = {
        "D-alpha, chord=50 cm": {
            'comment': f'SHOT: #{F_ID}',
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.ch1.to_list()
        },
        "Mark": {
            'comment': f'ELMs marks (by proc-sys v2.1-1.5scl; {datetime.now().strftime("%d.%m.%Y")})',
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.ch1_marked.to_list()
        },
        "AI prediction": {
            'comment': f'Processed NN prediction of ELMs (v{ckpt_v}; trn-on: #44[168|184|194] )',
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.ch1_ai_marked.to_list()
        }
    }
    
    packed = shtRipper.ripper.write(path=filepath + "marked/", filename=f'{F_ID}_data.SHT', data=to_pack)
    
    print(f"Result saved successfully to {filepath}marked/{F_ID}_data.SHT")
    print(f"Took - {round(time.time() - start_time, 2)} s")

    df.to_csv(filepath + f"marked/df/{F_ID}_data.csv", index=False)


if __name__ == "__main__" and not (sys.stdin and sys.stdin.isatty()):
    # get args from CL
    print("Sys args (full filepath | filename | checkpoint version):\n", sys.argv)
    init_proc(sys.argv[2], sys.argv[1], int(sys.argv[3]))

else:
    print("Program is supposed to run out from command line.")
