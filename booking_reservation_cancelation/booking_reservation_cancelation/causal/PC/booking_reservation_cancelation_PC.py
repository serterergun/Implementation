import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg

# Load the dataset
file_path = 'https://raw.githubusercontent.com/serterergun/Implementation/main/booking_reservation_cancelation/data/booking_reservation_cancelation_dataset.csv'
df = pd.read_csv(file_path)
# Drop specified columns

# all columns (hotel,is_canceled,lead_time,arrival_date_year,arrival_date_month,
              # arrival_date_week_number,arrival_date_day_of_month,stays_in_weekend_nights,
              # stays_in_week_nights,adults,children,babies,meal,country,market_segment,
              # distribution_channel,is_repeated_guest,previous_cancellations,
              # previous_bookings_not_canceled,reserved_room_type,assigned_room_type,
              # booking_changes,deposit_type,agent,company,days_in_waiting_list,customer_type,
              # adr,required_car_parking_spaces,total_of_special_requests,reservation_status,reservation_status_date)

df.drop(['agent','company','reservation_status','reservation_status_date'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['customer_type','distribution_channel', 'country', 'deposit_type','hotel',
                                 'market_segment','arrival_date_month','meal', 'reserved_room_type', 'assigned_room_type'], drop_first=True)

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
