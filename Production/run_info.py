import json 
import os
from glob import glob

def concat_json(folder, delete_origin=False):
    merged = {}
    for runfile in glob(f"{folder}/*.json"):
        with open(runfile, "r") as f:
            data = json.load(f)
            merged.update(data)
    with open(f"{folder}/../run_info.json", "w") as f:
        json.dump(merged, f)
    return merged

print(concat_json("/vols/cms/lcr119/HiggsDNA/output2802/tt/DYto2L_M-50_madgraphMLM_ext1/nominal/run_info"))




