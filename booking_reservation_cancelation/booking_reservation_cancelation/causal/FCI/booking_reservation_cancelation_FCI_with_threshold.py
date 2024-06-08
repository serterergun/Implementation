import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xgboost as xgb
import shap
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'https://raw.githubusercontent.com/serterergun/Implementation/main/booking_reservation_cancelation/data/booking_reservation_cancelation.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data = pd.get_dummies(data, columns=['customer_type','distribution_channel', 'country', 'deposit_type','hotel','market_segment',
                                     'arrival_date_month','meal', 'reserved_room_type', 'assigned_room_type'], drop_first=True)

data = data.drop(columns=['agent','company','reservation_status','reservation_status_date'])

# Convert boolean columns to integers
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

# Ensure all data is numeric and handle non-numeric data
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values by filling them with the mean of the column (or another method if preferred)
data.fillna(data.mean(), inplace=True)

# Feature names for labeling
feature_names = data.columns.tolist()

# Split data into features and target
X = data.drop('is_canceled', axis=1)
y = data['is_canceled']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', verbosity=1, seed=42)
model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Calculate mean absolute SHAP values for each feature
shap_sum = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['Feature', 'Mean SHAP Value']
importance_df.sort_values(by='Mean SHAP Value', ascending=True, inplace=True)

# Set a threshold and filter out low importance features
threshold = 0.01  # Set your threshold value here
low_importance_features = importance_df[importance_df['Mean SHAP Value'] < threshold]['Feature'].tolist()
# print(f"Features to drop: {low_importance_features}")

# Drop low importance features from data
data_reduced = data.drop(columns=low_importance_features)

# Convert the DataFrame to a NumPy array for causal analysis
data_np_reduced = data_reduced.values

# Feature names for labeling (after dropping low importance features)
feature_names_reduced = data_reduced.columns.tolist()

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

# Execute the function with reduced data
run_fci_and_visualize(data_np_reduced, feature_names_reduced)
