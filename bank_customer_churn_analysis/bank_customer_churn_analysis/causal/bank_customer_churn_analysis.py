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
dataset = pd.read_csv('https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_analysis.csv?token=GHSAT0AAAAAACRMURIXQD6PB2P4AKEV2GXSZSFJWSA')

# Drop the specified columns
dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode 'Gender' and 'Geography' columns
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})
dataset = pd.get_dummies(dataset, columns=['Geography'], drop_first=True)

# Check for missing values and handle if necessary
dataset.isnull().sum()

# Creating a copy of the dataset
dataset_copy = dataset.copy(deep=True)

# Define the causal graph in a detailed manner
causal_graph = """digraph {
    Gender[label="Gender"];
    Age[label="Age"];
    Tenure[label="Tenure"];
    Balance[label="Balance"];
    NumOfProducts[label="Number of Products"];
    HasCrCard[label="Has Credit Card"];
    IsActiveMember[label="Is Active Member"];
    EstimatedSalary[label="Estimated Salary"];
    Exited[label="Churn"];
    Geography_Germany[label="Geography: Germany"];
    Geography_Spain[label="Geography: Spain"];
    Gender -> Age;
    Age -> Exited;
    Balance -> Exited;
    NumOfProducts -> Exited;
    HasCrCard -> Exited;
    IsActiveMember -> Exited;
    Geography_Germany -> Exited;
    Geography_Spain -> Exited;
}"""

# Initialize the dowhy model with the new dataset and adjusted causal graph
model = dowhy.CausalModel(
        data=dataset,
        graph=causal_graph.replace("\n", " "),
        treatment="IsActiveMember",  # Adjust this based on the causal question
        outcome='Exited'
)

# Generate the causal graph image
model.view_model()

# Display the causal graph
display(Image(filename="causal_model.png"))
