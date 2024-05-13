import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd

hep.style.use("CMS")
plt.rcParams.update({"font.size": 16})

def plot_metric(df, train_metric, val_metric=None):
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(df["epoch"], df[train_metric], marker = "o", label = "Training")
    ax.plot(df["epoch"], df[val_metric], marker = "o", label = "Validation")
    ax.grid()
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(train_metric)
    ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
    ax.text(0.12, 1.02, 'Simulation Preliminary', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
    plt.savefig(f"{path_to_exp}/metrics/{train_metric}.pdf")
    

mlpath = "../../Training/python/mlruns"

exp_n = "5"
run_id = "a1ee3d4b5ed647038466354ba0a3d474"

path_to_exp = f"{mlpath}/{exp_n}/{run_id}"
path_to_metrics = f"{path_to_exp}/artifacts/metrics.log"


with open(path_to_metrics, 'r') as f:
    column_names = f.readline().strip().split(',')
df = pd.read_csv(path_to_metrics, names=column_names, skiprows=1)
df["epoch"] += 1 # start on epoch 1 to make more readable
metrics = [c for c in column_names if "epoch" not in c]
train_metrics = [m for m in metrics if "train" in m]
val_metrics = [m for m in metrics if "val" in m]
train_metrics = ["train_loss", "train_accuracy"]
val_metrics = ["val_loss", "val_accuracy"]

print(f"Successfully loaded metrics for Experiment: {exp_n} and run ID: {run_id}")

for tm, vm in zip(train_metrics, val_metrics):
    plot_metric(df, tm, val_metric=vm)
    print(f"Plot produced for {tm}")








