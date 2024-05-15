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

# Load the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/serterergun/Implementation/main/telco_churn_analysis/data/telco_churn_analysis.csv?token=GHSAT0AAAAAACRMURIWLGAIVKVZSRA5UZEGZSFJD3Q')

# Drop the "customerID" column
dataset = dataset.drop(['customerID'], axis=1)

# Encode categorical features
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'
]

# Convert categorical variables into dummy/indicator variables
dataset = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)

# Check for missing values and handle if necessary
dataset.isnull().sum()

# Creating a copy of the dataset
dataset_copy = dataset.copy(deep=True)

# Define the causal graph
causal_graph = """digraph {
    SeniorCitizen[label="Senior Citizen"];
    tenure[label="Tenure"];
    MonthlyCharges[label="Monthly Charges"];
    TotalCharges[label="Total Charges"];
    Churn[label="Churn"];
    gender_Male[label="Gender: Male"];
    Partner_Yes[label="Partner"];
    Dependents_Yes[label="Dependents"];
    PhoneService_Yes[label="Phone Service"];
    MultipleLines_Yes[label="Multiple Lines"];
    InternetService_Fiber_optic[label="Internet Service: Fiber Optic"];
    InternetService_No[label="Internet Service: No"];
    OnlineSecurity_Yes[label="Online Security"];
    OnlineBackup_Yes[label="Online Backup"];
    DeviceProtection_Yes[label="Device Protection"];
    TechSupport_Yes[label="Tech Support"];
    StreamingTV_Yes[label="Streaming TV"];
    StreamingMovies_Yes[label="Streaming Movies"];
    Contract_One_year[label="Contract: One Year"];
    Contract_Two_year[label="Contract: Two Year"];
    PaperlessBilling_Yes[label="Paperless Billing"];
    PaymentMethod_Credit_card_automatic[label="Payment Method: Credit Card (Automatic)"];
    PaymentMethod_Electronic_check[label="Payment Method: Electronic Check"];
    PaymentMethod_Mailed_check[label="Payment Method: Mailed Check"];

    gender_Male -> Churn;
    Partner_Yes -> Churn;
    Dependents_Yes -> Churn;
    PhoneService_Yes -> Churn;
    MultipleLines_Yes -> Churn;
    InternetService_Fiber_optic -> Churn;
    InternetService_No -> Churn;
    OnlineSecurity_Yes -> Churn;
    OnlineBackup_Yes -> Churn;
    DeviceProtection_Yes -> Churn;
    TechSupport_Yes -> Churn;
    StreamingTV_Yes -> Churn;
    StreamingMovies_Yes -> Churn;
    Contract_One_year -> Churn;
    Contract_Two_year -> Churn;
    PaperlessBilling_Yes -> Churn;
    PaymentMethod_Credit_card_automatic -> Churn;
    PaymentMethod_Electronic_check -> Churn;
    PaymentMethod_Mailed_check -> Churn;
    SeniorCitizen -> Churn;
    tenure -> Churn;
    MonthlyCharges -> Churn;
    TotalCharges -> Churn;
}"""

# Initialize the dowhy model with the new dataset and adjusted causal graph
model = dowhy.CausalModel(
    data=dataset,
    graph=causal_graph.replace("\n", " "),
    treatment="Contract_Two_year",  # Adjust this based on the causal question
    outcome='Churn'
)

# Generate the causal graph image
model.view_model()

# Display the causal graph
display(Image(filename="causal_model.png"))

# Save the causal graph as a PDF
graphviz.Source(causal_graph).render("causal_model", format='pdf')
