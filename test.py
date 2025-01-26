import pandas as pd
import kagglehub
import os

# Step 1: Download and load the dataset
path = kagglehub.dataset_download("marwa80/userbehavior")
print("Path to dataset files:", path)

# Step 2: Find the CSV file in the downloaded directory
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file_path = os.path.join(path, file)
        break

# Step 3: Load a sample of the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, nrows = 100000, fill_mean = 1)

# Step 4: Rename columns based on your provided explanation
df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# Step 5: Filter data for users who have performed the 'buy' behavior at least 2 times
user_behavior_count = df.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0)

# Step 6: Filter users who have bought at least 2 items
users_with_min_2_buys = user_behavior_count[user_behavior_count['buy'] >= 2]

# Step 7: Show the first few users who have purchased at least 2 items
print(users_with_min_2_buys.head())

# You can also check the total number of users who meet the criteria
print(f"Total number of users who bought at least 2 items: {len(users_with_min_2_buys)}")
