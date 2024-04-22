# Import libraries
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Sample data and features
X = np.random.rand(100, 5)  # Replace with your actual data and features
y = np.random.randint(0, 2, size=(100,))  # Replace with your target variable

# Train a model (simple RandomForestClassifier for demonstration purposes)
model = RandomForestClassifier()
model.fit(X, y)

# Explain individual predictions (optional)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[0:2])  # Analyze first data point

# SHAP summary plot and save as PNG
shap.summary_plot(shap_values, X[0:2], show=False)  # Prevent immediate display with show=False
plt.savefig("shap_summary_plot.png")  # Save the plot