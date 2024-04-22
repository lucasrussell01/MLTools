import tensorflow as tf 
import numpy as np
import pandas as pd
import glob 


class DataLoader:

    def __init__(self, config):

        self.config = config

        # Training Configuration
        self.n = self.config["Setup"]["n_events"] # number of events/batch
        self.n_batches = self.config["Setup"]["n_batches"] # number of batches for training
        self.n_batches_val = self.config["Setup"]["n_batches_val"] # number of batches for validation
        self.n_epochs = self.config["Setup"]["n_epochs"] # number of epochs
        self.epoch = self.config["Setup"]["epoch"] # starting epoch
        self.val_split = self.config["Setup"]["val_split"] # training/val split
        
        # NN Setup/Architecture
        self.model_name = self.config["SetupNN"]["model_name"] 
        self.dropout_rate = self.config["SetupNN"]["dropout"]
        self.use_weights = self.config["SetupNN"]["use_weights"]
        self.n_outputs = self.config["SetupNN"]["n_outputs"] # Classification onehot encoding
        
        # Input Files
        self.file_path = self.config["Setup"]["input_dir"]
        print(f"Loading files from: {self.file_path}")
        files = glob.glob(self.file_path + "/*.parquet") # TODO: Add file extension in config
        # np.random.shuffle(files)
        self.train_files, self.val_files = np.split(files, [int(len(files)*(1-self.val_split))])
        print(f"{len(self.train_files)} files for training || {len(self.val_files)} files for validation")
        
        # Feature Configuration
        self.features = self.config["Inputs"]["feature_list"] # variable names of input features
        if self.use_weights:
            self.input_shape = (len(self.features), self.n_outputs, None)
            self.input_types = (tf.float32, tf.float32, tf.float32)
        else:
            self.input_shape = (len(self.features), self.n_outputs)
            self.input_types = (tf.float32, tf.float32)
        self.weights = self.config["Inputs"]["weights"]
        self.truth = self.config["Inputs"]["truth"] # name of truth variable
        
    def get_generator(self, primary_set = True, show_progress = False, 
                      evaluation = False):

        _files = self.train_files if primary_set else self.val_files
        print(("Training" if primary_set else "Validation") + " file list loaded" )
        if len(_files)==0:
            raise RuntimeError(("Training" if primary_set else "Validation")+\
                                " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val
 
        # TODO: Generalise the generator (specify more from config)
        def _generator():
            for j in range(len(_files)):
                # TODO: Generalise to any data format (not flat)
                df = pd.read_parquet(_files[j], engine='pyarrow') 
                for i in range(len(df)):
                    x = (np.stack([df[f][i] for f in self.features]))
                    truth = df[self.truth][i]
                    if evaluation:
                        # Evaluation: simply output truth
                        if self.weights is None:
                            yield(x, truth)
                        else:
                            w = df[self.weights][i]
                            yield(x, truth, w)
                    else:
                        # Training: onehot encoding for multiclass DNN
                        y = tf.one_hot(truth, self.n_outputs)
                        if self.weights is None:
                            yield(x, truth)
                        else:
                            w = df[self.weights][i]
                            yield(x, y, w)           
        return _generator

    