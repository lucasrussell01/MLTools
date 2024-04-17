import pandas as pd
import numpy as np
import yaml
import os
from glob import glob


def shuffle_merge(file_list, out_dir, n_shards = 5):
    # Read in all files and shuffle
    shuffled_df = pd.concat([pd.read_parquet(ds, engine='pyarrow') for ds in file_list], 
                ignore_index = True).sample(frac=1, random_state = 99).reset_index(drop=True)
    n_entries = len(shuffled_df['weight'])
    print(f"Shuffling and Merging: {n_entries} entries")
    shard_size = int(np.ceil(n_entries/n_shards)) 
    # Split dataframes into shards
    dfs = [shuffled_df.iloc[shard_size*i:shard_size*(i+1)].reset_index(drop=True) for i in range(n_shards)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, chunk in enumerate(dfs):    
        # TODO: This is currently Higgs/Tau/Bkg Specific -> Config with truth in future?
        counts = chunk['true_category'].value_counts()
        print(f"Saving chunk {i}, with {counts.get(0, 0)} Taus {counts.get(1, 1)} Higgs, {counts.get(2, 2)} Bkg")
        chunk.to_parquet(f"{out_dir}/ShuffleMerge_{i}.parquet", engine='pyarrow')



if __name__ == "__main__":
    # Load directory/filenames from config
    config = yaml.safe_load(open("../configs/ShuffleMerge.yaml"))
    base_dir = config["Setup"]["merge_dir"]
    output_dir = config["Setup"]["out_dir"]
    file_list = glob(f"{base_dir}/*/*.parquet")
    # Load and Shuffle Files
    shuffle_merge(file_list, output_dir, n_shards = config["Setup"]["n_output_files"])
    
    
    
