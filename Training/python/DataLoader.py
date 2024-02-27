import tensorflow as tf 
import numpy as np
import pandas as pd
import glob 


class DataLoader:

    def __init__(self, config):

        self.config = config

        self.n = self.config["Setup"]["n_tau"] # number of events/batch
        self.n_batches = self.config["Setup"]["n_batches"] # number of batches for training
        self.n_batches_val = self.config["Setup"]["n_batches_val"] # number of batches for validation
        self.n_epochs = self.config["Setup"]["n_epochs"] # number of epochs
        self.epoch = self.config["Setup"]["epoch"] # starting epoch
        self.val_split = self.config["Setup"]["val_split"] # training/val split
        
        self.model_name = self.config["SetupNN"]["model_name"]
        self.dropout_rate = self.config["SetupNN"]["dropout"]
        self.use_weights = self.config["SetupNN"]["use_weights"]
        
        
        self.file_path = self.config["Setup"]["input_dir"]
        print(f"Loading files from: {self.file_path}")
        
        files = glob.glob(self.file_path + "/*,.parquet") # TODO: Add file extension in config
        self.train_files, self.val_files = 
                            np.split(files, [int(len(files)*(1-self.val_split))])
        print(f"{len(self.train_files)} files for training || 
                {len(self.validation_files)} files for validation")


    def get_generator(self, primary_set = True, show_progress = False):

        _files = self.train_files if primary_set else self.val_files
        print(("Training" if primary_set else "Validation") + " file list loaded" )
        if len(_files)==0:
            raise RuntimeError(("Training" if primary_set else "Validation")+\
                                " file list is empty.")

        n_batches = self.n_batches if primary_set else self.n_batches_val

        # **********************************************************************
        
        # TEMPLATE FROM DEEPPI: WILL BE UPDATED ASAP
        
        # **********************************************************************
        def _generator():
            for j in range(len(_files)):
                df = pd.read_parquet(_files[j]) # TODO: Generalise to any type!
                for i in range(len(df)):
                    # # Image inputs
                    # Tracks = df["Tracks"][i]
                    # ECAL = df["ECAL"][i]
                    # PF_HCAL = df["PF_HCAL"][i]
                    # PF_ECAL = df["PF_ECAL"][i]
                    # addTracks = df["addTracks"][i]
                    # # Clip outliers to max determined from 99.7%:
                    # if np.sum(Tracks)>200:
                    #     Tracks = 200*Tracks/np.sum(Tracks)
                    # if np.sum(ECAL)>200:
                    #     ECAL = 200*ECAL/np.sum(ECAL)
                    # if np.sum(PF_HCAL)>150:
                    #     PF_HCAL = 150*PF_HCAL/np.sum(PF_HCAL)
                    # if np.sum(PF_ECAL)>200:
                    #     PF_ECAL = 200*PF_ECAL/np.sum(PF_ECAL)
                    # if np.sum(addTracks)>70:
                    #     addTracks = 70*addTracks/np.sum(addTracks)
                    # x = (np.stack([Tracks, ECAL, PF_HCAL, PF_ECAL, addTracks], axis=-1))
                    # if self.use_HPS:
                    #     x_mass = (np.stack([df["HPS_tau_dm"][i], df["HPS_tau_pt"][i], df["HPS_tau_E"][i], df["HPS_tau_eta"][i], df["HPS_tau_mass"][i], df["HPS_pi_px"][i], 
                    #             df["HPS_pi_py"][i], df["HPS_pi_pz"][i], df["HPS_pi_E"][i], df["HPS_pi0_px"][i], df["HPS_pi0_py"][i], df["HPS_pi0_pz"][i], df["HPS_pi0_E"][i], 
                    #             df["HPS_pi0_dEta"][i], df["HPS_pi0_dPhi"][i], df["HPS_strip_mass"][i], df["HPS_strip_pt"][i], df["HPS_rho_mass"][i], df["HPS_pi2_px"][i], 
                    #             df["HPS_pi2_py"][i], df["HPS_pi2_pz"][i], df["HPS_pi2_E"][i], df["HPS_pi3_px"][i], df["HPS_pi3_py"][i], df["HPS_pi3_pz"][i], df["HPS_pi3_E"][i], 
                    #             df["HPS_mass0"][i], df["HPS_mass1"][i], df["HPS_mass2"][i], df["HPS_pi0_releta"][i], df["HPS_pi0_relphi"][i]], axis=-1))
                    #     x = tuple([x, x_mass])
                    # DM = df["DM"][i]
                    # if self.regress_kinematic:
                    #     max_index = np.where(df["relp"][i] == np.max(df["relp"][i]))[0] # find leading neutral
                    #     yKin = (df["relp"][i][max_index][0], df["releta"][i][max_index][0], df["relphi"][i][max_index][0])
                    # if DM_evaluation:
                    #     yDM = DM
                    #     HPSDM = df["HPS_tau_dm"][i]
                    #     MVADM = df["MVA_DM"][i]
                    #     VSjet = df["deeptauVSjet"][i]
                    #     VSe = df["deeptauVSe"][i]
                    #     VSmu = df["deeptauVSmu"][i]
                    #     yield(x, yDM, HPSDM, MVADM, VSjet, VSe, VSmu)
                    # elif Kin_evaluation:
                    #     PV = df["PV"][i]
                    #     HPSDM = df["HPS_tau_dm"][i]
                    #     HPS_pi0 = [df["HPS_pi0_px"][i], df["HPS_pi0_py"][i], df["HPS_pi0_pz"][i]]
                    #     jet = [df["jet_eta"][i], df["jet_phi"][i]]
                    #     yield(x, yKin, PV, DM, HPSDM, HPS_pi0, jet)
                    # else:
                    #     if DM == 0 or DM == 10:
                    #         yDM = tf.one_hot(0, 3) # no pi0
                    #         w = 0
                    #     elif DM ==1 or DM ==11:
                    #         yDM = tf.one_hot(1, 3) # one pi0
                    #         w = 1
                    #     elif DM == 2:
                    #         yDM = tf.one_hot(2, 3) # two pi0
                    #         w = 0
                    #     else: 
                    #         raise RuntimeError(f"Unknown DM {DM}")
                    #     if self.regress_kinematic:
                    #         yield (x, yDM, yKin, w)
                    #     else:
                    #         yield (x, yDM)

        return _generator

    