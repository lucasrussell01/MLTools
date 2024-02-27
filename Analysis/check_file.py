import numpy as np
import pandas as pd
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Quick scan of available info in a file')
parser.add_argument('--file', required=True, type=str, help="path to file to check")
parser.add_argument('--column', required=False, type=str, help="column to print")
args = parser.parse_args()

print(f"Loading file {args.file}")
df = pd.read_parquet(args.file, engine='pyarrow')

print("Preview of Dataframe:")
print(df.head())

print("Available columns:")
print(df.columns)

if args.column:
    print("---------------------------------------------------")
    print(f"Preview of column {args.column}:")
    print(df[args.column])