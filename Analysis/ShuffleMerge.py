import numpy as np
import pandas as pd
from glob import glob

out = "/vols/cms/lcr119/tuples/TauCP/ShuffleMerge/"

higgs_ds = ['GluGluHToTauTau_M125', 'VBFHTauTau_M125']
higgs_files = []


for ds in higgs_ds:
    higgs_files.append(glob(f"/vols/cms/lcr119/tuples/TauCP/TauTau_Weighted/{ds}/*.parquet"))
    
df = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in higgs_files], ignore_index = True)

df["truth"] = 1 # Higgs

df = df.sample(frac = 1, random_state = 22)

events_per_file = 2500

n_shards = len(df["truth"])//2500 
if len(df["truth"])%2500 != 0:
    n_shards += 1

dfs = [df.iloc[events_per_file*i:events_per_file*(i+1)] for i in range(n_shards)]

for i, chunk in enumerate(dfs):
    chunk.to_parquet(f"{out}Higgs_{i}.parquet", engine='pyarrow')


