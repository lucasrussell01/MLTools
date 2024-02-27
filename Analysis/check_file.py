import numpy as np
import pandas as pd
from glob import glob
import os


base_dir = '/vols/cms/lcr119/HiggsDNA/output/tt/'
file_end = "/*/*.parquet"
samples = ['Bloop']


for samp in samples:
    

    files = glob(base_dir + samp + file_end)
    
    for f in files:
        
        df = pd.read_parquet(f, engine='pyarrow')
        
        print("AVAILABLE COLUMNS ARE:")
        print(df.columns)
        
        print(df.head())
        
        print(df["nLHEjets"])

        print(df["os"])
        
        print(df["weight"])
        
   
