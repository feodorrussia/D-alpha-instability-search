import os.path
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime

import shtReader_py.shtRipper as shtRipper
from source.Files_operating import read_dataFile, read_sht_data
from source.NN_environment import process_fragments, get_borders, normalise_series, down_to_zero
from source.NN_environment import get_prediction_multi_unet


def init_proc_multi(filename: str, filepath: str, ckpt_v_list: list):
    filepath += "\\"
    F_ID = filename[-5:]
    
    df = read_sht_data(filename, filepath, data_name="D-alfa  хорда R=50 cm")
    start_time = time.time()

    predictions = get_prediction_multi_unet(df.ch1.to_numpy(), ckpt_v=ckpt_v_list[0])
    if len(ckpt_v_list) > 1:
        for ckpt_v in ckpt_v_list[1:]:
            predictions += get_prediction_multi_unet(df.ch1.to_numpy(), ckpt_v=ckpt_v)
        predictions /= len(ckpt_v_list)

    
    df["unsync_ai_marked"] = predictions[0, :]
    df["sync_ai_marked"] = predictions[1, :]
    
    df["unsync_marked"] = down_to_zero(np.array(df["unsync_ai_marked"]), edge=0.5)
    df["unsync_marked"] = process_fragments(np.array(df["ch1"]), np.array(df["unsync_marked"]), length_edge=30, scale=1.5)  # old version: length_edge=20, , scale=0
    
    df["sync_marked"] = down_to_zero(np.array(df["sync_ai_marked"]), edge=0.5)
    df["sync_marked"] = process_fragments(np.array(df["ch1"]), np.array(df["sync_marked"]), length_edge=30, scale=0)  # old version: length_edge=30, , scale=1.5
    
    sxr_df = read_sht_data(filename, filepath, data_name="SXR 50 mkm")
    
    comment = {"ai_marking": f'Processed NN prediction of ELMs (v{ckpt_v_list} assembly multiclass model)',
               "sync_proc_marks": f'Sync ELMs marks (by proc-sys v2.1-0scl; {datetime.now().strftime("%d.%m.%Y")})',
               "unsync_proc_marks": f'Unsync ELMs marks (by proc-sys v2.2-1.5scl; {datetime.now().strftime("%d.%m.%Y")})'}
    
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
        "SXR 50 mkm": {
            'comment': f'ADC #4, CH #6, SHOT: #{F_ID}',
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': sxr_df.ch1.to_list()
        },
        "Unsync ELM AI prediction": {
            'comment': comment["ai_marking"],
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.unsync_ai_marked.to_list()
        },
        "Unsync ELM mark": {
            'comment': comment["unsync_proc_marks"],
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.unsync_marked.to_list()
        },
        "Sync ELM AI prediction": {
            'comment': comment["ai_marking"],
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.sync_ai_marked.to_list()
        },
        "Sync ELM mark": {
            'comment': comment["sync_proc_marks"],
            'unit': 'U(V)',
            # 'x': df.t,
            'tMin': df.t.min(),  # minimum time
            'tMax': df.t.max(),  # maximum time
            'offset': 0.0,  # ADC zero level offset
            'yRes': 0.001,  # ADC resolution: 0.0001 Volt per adc bit
            'y': df.sync_marked.to_list()
        }
    }
    
    packed = shtRipper.ripper.write(path=filepath + "marked/", filename=f'{F_ID}_ai_data.SHT', data=to_pack)
    
    print(f"Result saved successfully to {filepath}marked/{F_ID}_ai_data.SHT")
    print(f"Took - {round(time.time() - start_time, 2)} s")
    
    df.to_csv(filepath + f"marked/df/{F_ID}_ai_data.csv", index=False)

if __name__ == "__main__" and not (sys.stdin and sys.stdin.isatty()):
    # get args from CL
    print("Sys args (full filepath | filename | checkpoint version):\n", sys.argv)
    init_proc_multi(sys.argv[2], sys.argv[1], list(map(int, sys.argv[3:])))

else:
    print("Program is supposed to run out from command line.")
