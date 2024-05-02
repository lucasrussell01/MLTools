from model_blocks import *
from tensorflow.keras.models import Model
from losses import NNLosses
import mlflow

class CustomModel(keras.Model):

    def __init__(self, *args, class_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_loss is None:
            self.class_loss = NNLosses.classification_loss
        else:
            self.class_loss = loss
        self.class_loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None)

    def train_step(self, data):
        # Unpack the data
        x, y, w = data
        n = tf.shape(y)[0]
        
        # Forward pass:
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss_vec = self.class_loss(y, y_pred)
            loss = tf.reduce_sum(tf.multiply(loss_vec, w))/tf.cast(n, dtype=tf.float32)
            
        # Compute gradients
        trainable_pars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_pars)
        
        # Update trainable parameters
        self.optimizer.apply_gradients(zip(gradients, trainable_pars))
        
        # Update metrics
        self.class_loss_tracker.update_state(loss)
        self.accuracy.update_state(y, y_pred, sample_weight= w)

        # Return a dict mapping metric names to current value (printout)
        metrics_out =  {m.name: m.result() for m in self.metrics}
        return metrics_out
    
    def test_step(self, data):
        # Unpack the data
        x, y, w = data
        n = tf.shape(y)[0]

        # Evaluate Model
        y_pred = self(x, training=False)
        
        # Calculate loss
        loss_vec = self.class_loss(y, y_pred)
        loss = tf.reduce_sum(tf.multiply(loss_vec, w))/tf.cast(n, dtype=tf.float32)
        
        # Update the metrics 
        self.class_loss_tracker.update_state(loss)
        self.accuracy.update_state(y, y_pred, sample_weight= w)

        # Return a dict mapping metric names to current value
        metrics_out = {m.name: m.result() for m in self.metrics}
        return metrics_out
    
    @property
    def metrics(self):
        # define metrics here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`
        metrics = []
        metrics.append(self.class_loss_tracker) 
        metrics.append(self.accuracy) 
        for l in self._flatten_layers():
            metrics.extend(l._metrics)  # pylint: disable=protected-access
        return metrics
    
    
def create_model(dataloader):

    # Architecture
    input_flat = Input(name="input_flat", shape=(dataloader.input_shape[0])) 
    dense_1 = dense_block(input_flat, 50, dropout=dataloader.dropout_rate, n="_dense_1")
    dense_2 = dense_block(dense_1, 50, dropout=dataloader.dropout_rate, n="_dense_2")
    dense_3 = dense_block(dense_2, 50, dropout=dataloader.dropout_rate, n="_dense_3")
    dense_final = dense_block(dense_3, 3, dropout=dataloader.dropout_rate, n="_dense_final")
    output = Activation("softmax", name="output")(dense_final)

    # Create model
    # TODO: Do this with a custom training loop in future
    model = CustomModel(input_flat, output, name=dataloader.model_name)

    return model

def compile_model(model):

    # TODO: Specify Optimiser via config
    opt = tf.keras.optimizers.Nadam(learning_rate=1e-4)

    # model here
    model.compile(loss=None, optimizer=opt, metrics=None)
    # model.compile(loss=NNLosses.classification_loss, optimizer=opt, metrics=["accuracy"])
    
    # mlflow log
    metrics = {}
    mlflow.log_dict(metrics, 'input_cfg/metric_names.json')


# TODO: Define custom model:
# class DNN_ClassificationModel(keras.Model):