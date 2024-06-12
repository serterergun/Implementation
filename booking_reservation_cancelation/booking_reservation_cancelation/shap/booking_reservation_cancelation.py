import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the dataset from a specified filepath.
    """
    return pd.read_csv(filepath)

def train_model(X, y):
    """
    Train an XGBoost classifier on the provided features and target.
    """
    model = xgb.XGBClassifier(objective='binary:logistic', verbosity=1, seed=42)
    model.fit(X, y)
    return model

def model_explanation(model, X):
    """
    Explain the model using SHAP values and plot the summary.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    plt.show()

def mean_shap_plot(model, X):
    """
    Generate a plot of mean absolute SHAP values for each feature.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Calculate the mean absolute SHAP values for each feature
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['Feature', 'Mean SHAP Value']
    importance_df.sort_values(by='Mean SHAP Value', ascending=True, inplace=True)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Mean SHAP Value'], color='salmon')
    plt.xlabel('Mean(|SHAP Value|) (impact on model output magnitude)')
    plt.title('Mean SHAP Values')
    plt.show()

def main():
    # Load real dataset
    df = load_data(r"https://raw.githubusercontent.com/serterergun/Implementation/main/booking_reservation_cancelation/data/booking_reservation_cancelation_dataset.csv")

    # Drop any irrelevant or non-feature columns
    df.drop(['hotel','reservation_status_date','reservation_status','agent','adults','country','market_segment','reserved_room_type','meal','is_repeated_guest','assigned_room_type','arrival_date_month','distribution_channel','children','babies','company'], axis=1, inplace=True)

    # Encode categorical features
    df = pd.get_dummies(df, columns=['deposit_type','customer_type'], drop_first=True)

    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Explain model using SHAP values on the test set
    model_explanation(model, X_test)

    # Generate the mean SHAP value plot
    mean_shap_plot(model, X_test)

if __name__ == "__main__":
    main()
