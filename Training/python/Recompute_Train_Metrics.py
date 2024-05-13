from DataLoader import DataLoader
import tensorflow as tf
from tensorflow.keras.models import load_model
import yaml
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from losses import NNLosses

# Training metrics are calculated as mean during training, so to properly
# compare to validation metrics, we need to recompute them by evaluating with 
# the NN weights frozen

mlpath = "../../Training/python/mlruns"
    
exp_n = "5"
run_id = "a1ee3d4b5ed647038466354ba0a3d474"
    
if __name__ == "__main__":
    # Load Model
    
    path_to_exp = f"{mlpath}/{exp_n}/{run_id}"
    path_to_artifacts = f"{path_to_exp}/artifacts"
    # Initialise DataLoader and Config
    config = yaml.safe_load(open("../../Training/configs/training.yaml"))
    data_loader = DataLoader(config)
    print("Initialised DataLoader")
    
    # Load Training Dataset
    gen_train = data_loader.get_generator(primary_set = True)
    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = data_loader.input_types, output_shapes = data_loader.input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(380*250, drop_remainder=True)
    print("Training Dataset Loaded")

    train_loss = []
    train_accuracy = []
    
    for e in range(20):
        path_to_model = f"{path_to_artifacts}/model_checkpoints/DNN_Classifier_step_e{e}.tf"
        model = load_model(path_to_model)#, {{name: lambda _: None for name in metric_names.keys()}}) # TODO: fix this as it's empty
        print(f"Model Loaded for Epoch {e}")
        # Compute Training Loss
        i_data_train = iter(data_train)
        x, y, w = next(i_data_train)
        y_pred = model(x, training=False)
        loss_vec = NNLosses.classification_loss(y, y_pred)
        acc = tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None)
        acc.update_state(y, y_pred, sample_weight= w)
        loss_val = tf.reduce_sum(tf.multiply(loss_vec, w))/tf.reduce_sum(w)
        acc_val = acc.result()
        print(f"Training Loss: {loss_val}, Accuracy: {acc_val}")
        train_loss.append(loss_val)
        train_accuracy.append(acc_val)
        
    # save metrics to metric.log
    df = pd.read_csv(f"{path_to_artifacts}/metrics.log")
    df["train_loss"] = train_loss
    df["train_accuracy"] = train_accuracy
    df.to_csv(f"{path_to_artifacts}/metrics.log", index=False)