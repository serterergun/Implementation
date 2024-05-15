import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dowhy
import graphviz

# Load the new dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/serterergun/Implementation/main/bank_customer_churn_analysis/data/bank_customer_churn_analysis.csv?token=GHSAT0AAAAAACRMURIWREBERBCK3KZAWPM6ZSFG3OQ')
dataset.head()

# Drop the specified columns
dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode 'Gender' and 'Geography' columns
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})
dataset = pd.get_dummies(dataset, columns=['Geography'], drop_first=True)

dataset.columns

# Check for missing values and handle if necessary
dataset.isnull().sum()

# For this example, we assume no additional handling is needed, but you may need to adjust based on the dataset

# Example of printing first few rows after processing
print(dataset.head())

# Creating a copy of the dataset
dataset_copy = dataset.copy(deep=True)

# Adjust the causal graph as per the new dataset. This is an example and needs to be adapted based on the dataset's features and domain knowledge.
causal_graph = """digraph {
    CreditScore;
    Gender;
    Age;
    Tenure;
    Balance;
    NumOfProducts;
    HasCrCard;
    IsActiveMember;
    EstimatedSalary;
    Exited[label="Churn"];
    Geography_Germany[label="Geography: Germany"];
    Geography_Spain[label="Geography: Spain"];
    U[label="Unobserved Confounders",observed="no"];
    U->{CreditScore, Balance, EstimatedSalary};
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
model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment="IsActiveMember",  # Adjust this based on the causal question
        outcome='Exited')
model.view_model()

from IPython.display import Image, display
display(Image(filename="causal_model.png"))