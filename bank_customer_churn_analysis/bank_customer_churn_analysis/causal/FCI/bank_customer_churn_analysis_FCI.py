import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, kci, d_separation
from causallearn.utils.GraphUtils import GraphUtils

# Load the dataset
file_path = 'https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_dataset.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname','Geography'])

# Convert boolean columns to integers
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

# Ensure all data is numeric and handle non-numeric data
# data = data.apply(pd.to_numeric, errors='coerce') (NOT NECESSARY FOR THIS CASE)

# Handle missing values by filling them with the mean of the column (or another method if preferred)
# data.fillna(data.mean(), inplace=True) (NOT NECESSARY FOR THIS CASE)

# Convert the DataFrame to a NumPy array
data_np = data.values

# Feature names for labeling
feature_names = data.columns.tolist()

# Define a function to visualize the causal graph
def visualize_causal_graph(g, labels, filename='causal_graph.png'):
    pyd = GraphUtils.to_pydot(g, labels=labels)
    pyd.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(50, 50))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# Run FCI and visualize the causal graph
def run_fci_and_visualize(data_np, feature_names):
    g, edges = fci(data_np, fisherz, 0.05)
    visualize_causal_graph(g, feature_names)
    return g

# Execute the function
run_fci_and_visualize(data_np, feature_names)
