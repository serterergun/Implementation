import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load the dataset from a specified filepath.

    age: Age of the patient
    anaemia: If the patient had the haemoglobin below the normal range
    creatinine_phosphokinase: The level of the creatine phosphokinase in the blood in mcg/L
    diabetes: If the patient was diabetic
    ejection_fraction: Ejection fraction is a measurement of how much blood the left ventricle pumps out with each contraction
    high_blood_pressure: If the patient had hypertension
    platelets: Platelet count of blood in kiloplatelets/mL
    serum_creatinine: The level of serum creatinine in the blood in mg/dL
    serum_sodium: The level of serum sodium in the blood in mEq/L
    sex: The sex of the patient
    smoking: If the patient smokes actively or ever did in past
    time: It is the time of the patient's follow-up visit for the disease in months
    DEATH_EVENT: If the patient deceased during the follow-up period
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
    df = load_data(r"C:\Users\ergun\Desktop\SHAP_VS_CAUSAL\heart_failure_analysis\data\heart_failure_analysis.csv")

    # Drop any irrelevant or non-feature columns
    # df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)

    # Encode categorical features
    # df = pd.get_dummies(df, columns=['Gender', 'Geography'], drop_first=True)

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

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
