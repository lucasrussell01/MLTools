# mlflow logging inspired by DeepTau
from DataLoader import DataLoader
from setup_gpu import setup_gpu
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, MaxPooling2D
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler
import mlflow
mlflow.tensorflow.autolog(log_models=False)
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import json
import os
from glob import glob
import numpy as np
from losses import NNLosses, EpochCheckpoint


def run_training(model, data_loader):

    print(f"Warning: Model name for training is: {data_loader.model_name}")
    gen_train = data_loader.get_generator(primary_set = True)
    gen_val = data_loader.get_generator(primary_set = False)

    # Datasets from generators
    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = data_loader.input_types, output_shapes = data_loader.input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n, drop_remainder=True)
    if data_loader.n_batches != -1: # if not running on full dataset
        data_train = data_train.take(data_loader.n_batches) 
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = data_loader.input_types, output_shapes = data_loader.input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n, drop_remainder=True)
    if data_loader.n_batches_val != -1: # if not running on full dataset
        data_val = data_val.take(data_loader.n_batches_val)
    
    # logs/callbacks
    model_name = data_loader.model_name
    log_name = f"{model_name}_step"
    csv_log_file = "metrics.log"
    if os.path.isfile(csv_log_file):
        close_file(csv_log_file)
        os.remove(csv_log_file)
    csv_log = CSVLogger(csv_log_file, append=True)
    epoch_checkpoint = EpochCheckpoint(log_name)
    callbacks = [epoch_checkpoint, csv_log]

    # Run training
    fit = model.fit(data_train, validation_data = data_val, epochs = data_loader.n_epochs, 
                    initial_epoch = data_loader.epoch, callbacks = callbacks)

    model_path = f"{log_name}_final.tf"
    model.save(model_path, save_format="tf")

    # mlflow logs
    for checkpoint_dir in glob(f'{log_name}*.tf'):
         mlflow.log_artifacts(checkpoint_dir, f"model_checkpoints/{checkpoint_dir}")
    mlflow.log_artifacts(model_path, "model")
    mlflow.log_artifact(csv_log_file)
    mlflow.log_param('model_name', model_name)

    return fit


@hydra.main(config_path='.', config_name='hydra_train') # TODO: Move hydra train somewhere
def main(cfg: DictConfig) -> None:
    
    # set up mlflow experiment id
    mlflow.set_tracking_uri(f"file://{to_absolute_path(cfg.path_to_mlflow)}")
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is not None:
        run_kwargs = {'experiment_id': experiment.experiment_id}
        if cfg["pretrained"] is not None: # initialise with pretrained run, otherwise create a new run
            run_kwargs['run_id'] = cfg["pretrained"]["run_id"]
    else: # create new experiment
        experiment_id = mlflow.create_experiment(cfg.experiment_name)
        run_kwargs = {'experiment_id': experiment_id}
        

    # run the training with mlflow tracking
    with mlflow.start_run(**run_kwargs) as main_run:

        if cfg["pretrained"] is not None:
            mlflow.start_run(experiment_id=run_kwargs['experiment_id'], nested=True)
        active_run = mlflow.active_run()
        run_id = active_run.info.run_id
        
        # load configs
        setup_gpu(cfg.gpu_cfg)
        training_cfg = OmegaConf.to_object(cfg.training_cfg) # convert to python dictionary
        dataloader = DataLoader(training_cfg)

        # main training: create appropriate model
        if dataloader.model_name == "DNN_Classifier":
            from DNN_Classifier import create_model, compile_model
            model = create_model(dataloader)
        else:
            raise ValueError("Unknown Model Name - Options are 'DNN_Classifier'")

        if cfg.pretrained is None:
            print("Warning: no pretrained NN -> training will be started from scratch")
        else:
            # If starting from a pretrained model, load in relevant layer weights
            print("Warning: training will be started from pretrained model.")
            print(f"Model: run_id={cfg.pretrained.run_id}, experiment_id={cfg.pretrained.experiment_id}, model={cfg.pretrained.starting_model}")
            path_to_pretrain = to_absolute_path(f'{cfg.path_to_mlflow}/{cfg.pretrained.experiment_id}/{cfg.pretrained.run_id}/artifacts/')
            old_model = load_model(path_to_pretrain+f"/model_checkpoints/{cfg.pretrained.starting_model}",
                compile=False, custom_objects = None)
            for layer in model.layers:
                weights_found = False
                for old_layer in old_model.layers:
                    if layer.name == old_layer.name:
                        layer.set_weights(old_layer.get_weights())
                        weights_found = True
                        break
                if not weights_found:
                    print(f"Weights for layer '{layer.name}' not found.")

        if dataloader.model_name == "DNN_Classifier":
            compile_model(model)

        # Begin training
        fit = run_training(model, dataloader)

        # log NN params
        with open(to_absolute_path(f'{cfg.path_to_mlflow}/{run_kwargs["experiment_id"]}/{run_id}/artifacts/model_summary.txt')) as f:
            for l in f:
                if (s:='Trainable params: ') in l:
                    mlflow.log_param('n_train_params', int(l.split(s)[-1].replace(',', '')))

        # log training related files
        mlflow.log_dict(training_cfg, 'input_cfg/training_cfg.yaml')
        mlflow.log_artifact(to_absolute_path("TrainNN.py"), 'input_cfg')

        # log hydra files
        mlflow.log_artifacts('.hydra', 'input_cfg/hydra')
        mlflow.log_artifact('TrainNN.log', 'input_cfg/hydra')

        # log misc. info
        mlflow.log_param('run_id', run_id)
        print(f'\nTraining has finished! Corresponding MLflow experiment name (ID): {cfg.experiment_name}({run_kwargs["experiment_id"]}), and run ID: {run_id}\n')
        mlflow.end_run()


if __name__ == '__main__':
    main()