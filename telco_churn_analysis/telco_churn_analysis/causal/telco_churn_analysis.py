import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg

# Load the dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/serterergun/Implementation/main/telco_churn_analysis/data/telco_churn_analysis.csv?token=GHSAT0AAAAAACRMURIWZGFWQPL44O52G4QYZSL64YA")

# Drop specified columns
df.drop(['customerID'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['Contract','OnlineSecurity','StreamingTV','PaperlessBilling','OnlineBackup','TechSupport','StreamingMovies','gender','Partner','Dependents','PhoneService','MultipleLines','DeviceProtection','InternetService','PaymentMethod'], drop_first=True)

# Ensure all columns are numeric and fill any missing values
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)

# Remove columns with zero variance
df = df.loc[:, df.var() != 0]

# Check for multicollinearity by calculating the correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features with high correlation
df.drop(columns=to_drop, inplace=True)

# Convert all columns to float64 to ensure compatibility with np.isnan
df = df.astype(np.float64)

# Get labels and convert data to numpy array
labels = df.columns.tolist()
data = df.to_numpy()

# Apply PC algorithm
cg = pc(data)

# Visualization using pydot
pyd = GraphUtils.to_pydot(cg.G, labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.figure(figsize=(50, 50))  # Adjust the figure size as needed
plt.axis('off')
plt.imshow(img)
plt.show()
