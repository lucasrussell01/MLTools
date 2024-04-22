# Import necessary libraries
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Define a simple deep learning model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),  # Assuming 10 features as input
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax') # 2 output classes
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

# Generate synthetic data for demonstration
np.random.seed(42)
X_train = np.random.rand(100, 10)  # 100 samples with 10 features each
y_train = np.random.randint(0, 2, size=(100,))  # Binary classification labels
y_train = to_categorical(y_train, num_classes=2)

# Train the model
model = create_model()
model.fit(X_train, y_train, epochs=50, batch_size=32) # epoch

# Create a SHAP DeepExplainer instance
explainer = shap.DeepExplainer(model, data=X_train)

# Get SHAP values for the chosen sample
shap_values = explainer.shap_values(X_train)

# Visualize the summary plot
shap.summary_plot(shap_values, feature_names=['Feature ' + str(i) for i in range(1, 11)], show=False)#
# plotting the summary plot for shap_values[0] or shape_values[1] will give a plot with variation of SHAP value for each epoch
plt.savefig("shap_summary_plot_cnn.png")  # Save the plot