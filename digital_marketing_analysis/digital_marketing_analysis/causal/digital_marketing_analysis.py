import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg

# Load the dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/serterergun/Implementation/main/digital_marketing_analysis/data/digital_marketing_analysis.csv?token=GHSAT0AAAAAACRMURIX3VHCXZI46DZ7TS24ZSL3XAQ")

# Drop specified columns
df.drop([], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=[], drop_first=True)

# Ensure all columns are numeric and fill any missing values
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)

# Convert all columns to float64 to ensure compatibility with np.isnan
df = df.astype(np.float64)

# Get labels and convert data to numpy array
labels = df.columns.tolist()
data = df.to_numpy()

cg = pc(data)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(cg.G, labels=labels)
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.figure(figsize=(100, 100))
plt.axis('off')
plt.imshow(img)
plt.show()