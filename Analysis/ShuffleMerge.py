import numpy as np
import pandas as pd
from glob import glob

base_dir = '/vols/cms/lcr119/tuples/TauCP/NN_weighted'
out_dir = '/vols/cms/lcr119/tuples/TauCP/ShuffleMerge'

datasets = glob(f"{base_dir}/*/*.parquet")

# Read in all files and shuffle
shuffled_df = pd.concat([pd.read_parquet(ds, engine='pyarrow') for ds in datasets], 
            ignore_index = True).sample(frac=1, random_state = 99).reset_index(drop=True)

n_events = len(shuffled_df['weight'])
print(f"Total number of events available: {n_events}")

# Save in several shards 
shard_size = 10000

n_shards = n_events // shard_size
if n_events % shard_size != 0:
    n_shards += 1

dfs = [shuffled_df.iloc[shard_size*i:shard_size*(i+1)] for i in range(n_shards)]

for i, chunk in enumerate(dfs):
    counts = chunk['true_category'].value_counts()
    print(f"Saving chunk {i}, with {counts.get(0, 0)} Taus {counts.get(1, 1)} Higgs, {counts.get(2, 2)} Bkg")
    chunk.to_parquet(f"{out_dir}/ShuffleMerge_{i}.parquet", engine='pyarrow')


