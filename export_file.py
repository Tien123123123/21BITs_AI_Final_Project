from preprocessing import preprocess_data
import kagglehub
import os

# Step 1: Download and locate the dataset
path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")

# Locate the CSV file
csv_file_path = os.path.join(path, "2019-Nov.csv")

if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Dataset file not found at {csv_file_path}")

# Step 2: Preprocess data
print("Preprocessing data...")
df, df_weighted = preprocess_data(csv_file_path)
df = df[1:1000001]
df.to_csv("new_data.csv", index=False)