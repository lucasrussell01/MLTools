import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import numpy as np
# import shap   # not currenty working
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Don't need to use GPU for evaluation

from tensorflow.keras.layers import Input, Dense,  Dropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU
                                    
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
    
def add_block_ending(name_format, layer):
    norm_layer = BatchNormalization(name=name_format.format('norm'))(layer)
    activation_layer = PReLU(shared_axes=None,
                                name=name_format.format('activation'))(norm_layer)
    return Dropout(0.2, name=name_format.format('dropout'))(activation_layer)

def dense_block(prev_layer, kernel_size, block_name, n, basename='dense'):
    dense = Dense(kernel_size, name="{}_{}_{}".format(block_name, basename, n),
                      kernel_initializer='he_uniform')(prev_layer)
    return add_block_ending('{}_{{}}_{}'.format(block_name, n), dense)

def tau_block(input_layer, basename='dense'):
    prev_layer = input_layer
    layer_sizes = [104, 74, 52]
    for n, layer_size in enumerate(layer_sizes):
        prev_layer = dense_block(prev_layer, layer_size, "tau", n+1, basename=basename)
    return prev_layer


def final_block(input_layer, basename='dense'):
    prev_layer = input_layer
    layer_sizes = [200, 200, 200, 200]
    for n, layer_size in enumerate(layer_sizes):
        prev_layer = dense_block(prev_layer, layer_size, "final", n+1, basename=basename)
    return prev_layer
    
    


def explainer_model(original_model):
    # a simplified model of DeepTau which takes a flat tensor as input
    input_layer = tf.keras.Input(name = "input_flat", shape = (171,))

    # Split into high level and latent space variables
    tau_input = tf.keras.layers.Lambda(lambda x: x[:, :43], name="input_tau")(input_layer)
    proc_grid = tf.keras.layers.Lambda(lambda x: x[:, 43:], name="proc_grid")(input_layer)

    # Process Tau features
    proc_tau = tau_block(tau_input)
    
    # Concatenate all processed features
    features_concat = Concatenate(name="features_concat", axis=1)([proc_tau, proc_grid])
    
    # Create final dense block as usual
    final_dense = final_block(features_concat)
    output_layer = Dense(4, name="final_dense_last",
                         kernel_initializer='he_uniform')(final_dense)
    softmax_output = Activation("softmax", name="main_output")(output_layer)
    
    # Initilise Model
    model = tf.keras.Model(inputs=input_layer, outputs=softmax_output)
    
    # Important: load in weights for the layers from the DeepTau full model
    for layer in model.layers:
        weights_found = False
        for old_layer in original_model.layers:
            if layer.name == old_layer.name:
                layer.set_weights(old_layer.get_weights())
                weights_found = True
                break
        if not weights_found:
            print(f"Weights for layer '{layer.name}' not found.")
    
    return model
    
    
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
    
    # Unpack one batch for now
    x, y, y_adv, sample_weight, sample_weight_adv = next(data_iter)
    
    # Uncomment this section if you want to check the predictions of the DeepTau model
    # y_class_pred, y_adv_pred = model(x, training=False)
    # print(y_class_pred)

    # create subnetworks for the grid processing
    outer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(name = "outer_cells_flatten").output)
    inner_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(name = "inner_cells_flatten").output)
    
    # Generate the latent space predictions (processed grid inputs)
    x_out = outer_model(x, training=False)
    x_in = inner_model(x, training=False)
    
    # Create the input tensor for the simplified explainer model
    x_analysis = tf.concat([x[0], x_in, x_out], axis = 1) # (171 inputs)
    
    # initialise model
    exp_model = explainer_model(model)
    
    # print(exp_model.summary())
    

    # Uncomment to check predictions of explainer(eg to compare vs original)
    y = exp_model(x_analysis, training=False)
    # print(y)

    
    # explain(next(data_iter), model)    # explain features?
    
    
  