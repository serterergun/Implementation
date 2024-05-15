import sys
import subprocess

# Install required system dependencies for pygraphviz
try:
    if sys.platform.startswith('linux'):
        subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'graphviz', 'graphviz-dev'])
    elif sys.platform == 'darwin':  # macOS
        subprocess.check_call(['brew', 'install', 'graphviz'])
    elif sys.platform.startswith('win'):  # Windows
        print("Please install Graphviz from https://graphviz.org/download/ and add it to your PATH.")
except Exception as e:
    print(e)

# Install pygraphviz
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pygraphviz'])
except Exception as e:
    print(e)

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dowhy
import graphviz
from IPython.display import Image, display

# Load the new dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/serterergun/Implementation/main/heart_failure_analysis/data/heart_failure_analysis.csv?token=GHSAT0AAAAAACRMURIXN52RIZGALH2B6TPMZSFI3XA')

# Display the first few rows of the dataset to understand its structure
print(dataset.head())
print(dataset.columns)
print(dataset.info())

# Check for missing values and handle if necessary
print(dataset.isnull().sum())

# For simplicity, let's fill missing values with the mean for numerical features and mode for categorical features
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column].fillna(dataset[column].mode()[0], inplace=True)
    else:
        dataset[column].fillna(dataset[column].mean(), inplace=True)

# Creating a copy of the dataset
dataset_copy = dataset.copy(deep=True)

# Define the outcome variable
outcome_variable = 'DEATH_EVENT'

# Identify a potential treatment variable - adjust based on domain knowledge
# For now, we will use 'serum_sodium' as an example; please adjust based on your specific analysis
treatment_variable = 'serum_sodium'  # Example, replace with actual treatment feature if different

# Define all features in the dataset excluding the outcome variable
features = [feature for feature in dataset.columns if feature != outcome_variable]
nodes = [f'{feature}[label="{feature}"]' for feature in features]
edges = [f'{feature} -> {outcome_variable}' for feature in features]

# Define the causal graph in a detailed manner
causal_graph = f"""digraph {{
    {"; ".join(nodes)};
    {"; ".join(edges)};
    {outcome_variable}[label="{outcome_variable}"];
}}"""
print(causal_graph)

# Initialize the dowhy model with the new dataset and adjusted causal graph
model = dowhy.CausalModel(
    data=dataset,
    graph=causal_graph.replace("\n", " "),
    treatment=treatment_variable,  # Replace with the actual treatment feature
    outcome=outcome_variable  # The outcome feature
)

# Generate the causal graph image
model.view_model()

# Display the causal graph
display(Image(filename="causal_model.png"))

# Save the causal graph as a PDF
graphviz.Source(causal_graph).render("causal_model", format='pdf')
