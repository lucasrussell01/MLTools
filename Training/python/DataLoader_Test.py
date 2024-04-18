from DataLoader import DataLoader
import tensorflow as tf
import yaml

# Script to test DataLoader...

def scan_tf_ds(data):
    i_scan = iter(data)
    x, y, w = next(i_scan)
    print(x.shape, y.shape, w.shape)
    

if __name__ == "__main__":
    config = yaml.safe_load(open("../configs/training.yaml"))
    data_loader = DataLoader(config)
    print("Initialised DataLoader")
    
    # Test Training Dataset
    gen_train = data_loader.get_generator(primary_set = True)
    data_train = tf.data.Dataset.from_generator(
        gen_train, output_types = data_loader.input_types, output_shapes = data_loader.input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n, drop_remainder=True)
    print("Sample of Training DataLoader:")
    scan_tf_ds(data_train)
    
    # Test Validation Dataset
    gen_val = data_loader.get_generator(primary_set = False)
    data_val = tf.data.Dataset.from_generator(
        gen_val, output_types = data_loader.input_types, output_shapes = data_loader.input_shape
        ).prefetch(tf.data.AUTOTUNE).batch(data_loader.n, drop_remainder=True)
    print("Shape of Validation DataLoader:")
    scan_tf_ds(data_val)
    
