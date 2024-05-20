import pandas as pd
import numpy as np
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg

# Load the dataset
df = pd.read_csv(r"C:\Users\ergun\Desktop\SHAP_VS_CAUSAL\Implementation\bank_customer_churn_analysis\data\bank_customer_churn_analysis.csv")

# Drop specified columns
df.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender', 'Geography'], axis=1, inplace=True)

# Verify data types and fill any missing values
for column in df.columns:
    if not np.issubdtype(df[column].dtype, np.number):
        df[column] = pd.to_numeric(df[column], errors='coerce')
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)

# Create LiNGAM model and fit it
model_lingam = lingam.ICALiNGAM()
model_lingam.fit(df)

# Visualize the causal graph
labels = df.columns.tolist()
dot = make_dot(model_lingam.adjacency_matrix_, labels=labels)
dot.render('lingam_causal_graph', format='png', cleanup=True)

# Display the generated graph
image_path = 'lingam_causal_graph.png'
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()