import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, kci, d_separation
from causallearn.utils.GraphUtils import GraphUtils

# Load the dataset
file_path = 'https://raw.githubusercontent.com/serterergun/Implementation/main/booking_reservation_cancelation/data/booking_reservation_cancelation.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data = pd.get_dummies(data, columns=['customer_type','distribution_channel', 'deposit_type','hotel','market_segment',
                                     'arrival_date_month','meal', 'reserved_room_type', 'assigned_room_type'], drop_first=True)

data = data.drop(columns=['agent','company','reservation_status','reservation_status_date', 'country'])

# all columns = ['hotel', 'is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_month',
#               'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
#               'stays_in_week_nights', 'adults', 'children', 'babies', 'meal', 'country', 'market_segment',
#               'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
#               'previous_bookings_not_canceled', 'reserved_room_type', 'assigned_room_type', 'booking_changes',
#               'deposit_type', 'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr',
#               'required_car_parking_spaces', 'total_of_special_requests', 'reservation_status', 'reservation_status_date']

# Convert boolean columns to integers
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

# Ensure all data is numeric and handle non-numeric data
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values by filling them with the mean of the column (or another method if preferred)
data.fillna(data.mean(), inplace=True)

# Convert the DataFrame to a NumPy array
data_np = data.values

# Feature names for labeling
feature_names = data.columns.tolist()

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

# Execute the function
run_fci_and_visualize(data_np, feature_names)
