import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import numpy as np
# import shap   # not currenty working
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Don't need to use GPU for evaluation


def load_model(path_to_model):
    # Load a saved tensorflow model
    with open(f"{path_to_model}/../input_cfg/metric_names.json", "r") as f:
        metric_names = json.load(f) 
    print("Model artifacts retrieved...")
    model = tf.keras.models.load_model(path_to_model, {name: lambda _: 
                                        None for name in metric_names.keys()}) 
    print("Model successfully loaded!")
    return model

def get_iterator(path_to_dataset):
    # Return an iterator for a tf dataset
    dataset = tf.data.experimental.load(path_to_dataset, 
                                compression="GZIP").prefetch(tf.data.AUTOTUNE)
    data_iter = iter(dataset)   
    print("Dataset successfully loaded!")
    return data_iter

def test(data, model):
    # Test model predictions, useful to check model works
    # TODO: Make this generalisable to non adversarial case
    x, y, y_adv, sample_weight, sample_weight_adv = data
    y_class_pred, y_adv_pred = model(x, training=False)
    print("Model prediction complete")
    return y_class_pred, y_adv_pred

def data_struct(data):
    # check structure of dataset
    x, y, y_adv, sample_weight, sample_weight_adv = data
    print("x shape:")
    for xs in x:
        print(xs.shape)
    
# def explain(data, model):
#     # Try to explain features with SHAP -> DOES NOT WORK YET
#     x, y, y_adv, sample_weight, sample_weight_adv = data
#     print("Trying?")
#     explainer = shap.DeepExplainer(model, x)
#     print("Alive?")


if __name__ == "__main__":
    
    # DeepTau v2p5 Model stored here...
    path_to_model = "/vols/cms/lcr119/FeatureImportance/MLTools/Training"\
                "/trained_models/DeepTauv2p5/e1f3ddb3c4c94128a34a7635c56322eb"\
                "/artifacts/model"
    # DeepTau v2p5 Dataset with 100 default and 100 adversarial taus/batch
    path_to_ds = "/vols/cms/lcr119/tuples/DeepTauTF" # 2000 batches here?
    
    # Load DeepTau Model and TF Dataset
    model = load_model(path_to_model) 
    data_iter = get_iterator(path_to_ds)
    
    print(model.summary())    # structure of model
    
    data_struct(next(data_iter))
    
    # test(next(data_iter), model)   # check model runs (optional)
    
    # explain(next(data_iter), model)    # explain features?
    
    
  