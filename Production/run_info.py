import json 
import os
from glob import glob

def concat_json(folder, delete_origin=False):
    merged = {}
    for runfile in glob(f"{folder}/*.json"):
        with open(runfile, "r") as f:
            data = json.load(f)
            merged.update(data)
        if delete_origin:
            os.remove(runfile)
    with open(f"{folder}/../run_info.json", "w") as f:
        json.dump(merged, f)
    return merged