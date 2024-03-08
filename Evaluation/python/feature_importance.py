import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import numpy as np
import shap 
import matplotlib.pyplot as plt                                  
tf.compat.v1.disable_v2_behavior()
    
def explainer(x, model):
    # Return a SHAP feature explainer
    explainer = shap.DeepExplainer(model, x)
    print("Initialised Explainer!")
    return explainer


if __name__ == "__main__":
    
    # Explainer Model
    path_to_model = "explainer_model"
    model = tf.keras.models.load_model(path_to_model) 
    # Dataset with HL and LATENT cell features
    path_to_ds = "inputs.npy" 
    x = np.load(path_to_ds)
    
    # TODO: Put these in a config somewhere
    hl_names = ["rho","tau_pt","tau_eta","tau_mass","tau_E_over_pt",
                "tau_charge","tau_n_charged_prongs","tau_n_neutral_prongs",
                "tau_chargedIsoPtSum","tau_chargedIsoPtSumdR03_over_dR05",
                "tau_footprintCorrection","tau_neutralIsoPtSum","tau_neutralIsoPtSumWeight_over_neutralIsoPtSum",
                "tau_neutralIsoPtSumWeightdR03_over_neutralIsoPtSum","tau_neutralIsoPtSumdR03_over_dR05",
                "tau_photonPtSumOutsideSignalCone","tau_puCorrPtSum","tau_dxy_valid",
                "tau_dxy","tau_dxy_sig","tau_ip3d_valid","tau_ip3d","tau_ip3d_sig",
                "tau_dz","tau_dz_sig_valid","tau_dz_sig","tau_flightLength_x",
                "tau_flightLength_y","tau_flightLength_z","tau_flightLength_sig",
                "tau_pt_weighted_deta_strip","tau_pt_weighted_dphi_strip",
                "tau_pt_weighted_dr_signal","tau_pt_weighted_dr_iso","tau_leadingTrackNormChi2",
                "tau_e_ratio_valid","tau_e_ratio","tau_gj_angle_diff_valid",
                "tau_gj_angle_diff","tau_n_photons","tau_emFraction","tau_inside_ecal_crack",
                "tau_leadChargedCand_etaAtEcalEntrance_minus_tau_eta"]

    
    inn_names = [f"LATENT_InnerCells{i}" for i in range(64)]
    out_names = [f"LATENT_OuterCells{i}" for i in range(64)]
    
    # Names of the input features
    feat_names = np.concatenate([hl_names, inn_names, out_names])
    # Names of the output classes
    class_names = ["e", "mu", "tau", "jet"]

    print("Explainer model loaded, beginning feature importance checks")
    

    exp = explainer(x, model) 

    shap_values = exp.shap_values(x)
    
    shap.summary_plot(shap_values, x, feature_names = feat_names, 
                      class_names = class_names, plot_size=[14,8], max_display=25)
    plt.savefig('shap_summary_plot.pdf')
    

    print("SHAP Summary complete!")
    
    # print(shap_values)
    
    
    
  