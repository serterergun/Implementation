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
dataset = pd.read_csv('https://raw.githubusercontent.com/serterergun/Implementation/main/digital_marketing_analysis/data/digital_marketing_analysis.csv?token=GHSAT0AAAAAACRMURIW2KLPR3ABUXU3ARYMZSFIP7Q')

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
outcome_variable = 'churn'

# Identify a potential treatment variable - adjust based on domain knowledge
treatment_variable = 'MarketingSpend'  # Example, replace with actual treatment feature if different

# Define the causal graph in a detailed manner
causal_graph = """digraph {
    AccountLength[label="Account Length"];
    Location[label="Location"];
    AddtoWishlist[label="Add to Wishlist"];
    DesktopSessions[label="Desktop Sessions"];
    AppSessions[label="App Sessions"];
    DesktopTransactions[label="Desktop Transactions"];
    TotalProductDetailViews[label="Total Product Detail Views"];
    SessionDuration[label="Session Duration"];
    PromotionClicks[label="Promotion Clicks"];
    AverageOrderValue[label="Average Order Value"];
    SaleProductViews[label="Sale Product Views"];
    DiscountRatePerVisitedProducts[label="Discount Rate per Visited Products"];
    ProductDetailViewPerAppSession[label="Product Detail View per App Session"];
    AppTransactions[label="App Transactions"];
    AddToCartPerSession[label="Add to Cart per Session"];
    CustomerServiceCalls[label="Customer Service Calls"];
    churn[label="Churn"];
    AccountLength -> churn;
    Location -> churn;
    AddtoWishlist -> churn;
    DesktopSessions -> churn;
    AppSessions -> churn;
    DesktopTransactions -> churn;
    TotalProductDetailViews -> churn;
    SessionDuration -> churn;
    PromotionClicks -> churn;
    AverageOrderValue -> churn;
    SaleProductViews -> churn;
    DiscountRatePerVisitedProducts -> churn;
    ProductDetailViewPerAppSession -> churn;
    AppTransactions -> churn;
    AddToCartPerSession -> churn;
    CustomerServiceCalls -> churn;
}"""

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
