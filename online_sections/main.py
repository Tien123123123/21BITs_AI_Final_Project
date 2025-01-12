from process_data.preprocessing import preprocess_data
from collaborative.train_model import train_model
from collaborative.session_based_recommend import recommend_products
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

# Step 3: Train model only if it doesn't already exist
model_file = "../model.pkl"

if not os.path.exists(model_file):
    print("Training model...")
    train_model(df_weighted, model_file)
else:
    print(f"Model already exists at {model_file}. Skipping training.")

# Step 4: Generate recommendations
print("Generating recommendations...")
target_user_id = 1  # Example target user ID
target_product_id = 1  # Example target product ID

# Use the preprocessed dataframe for recommendations
recommendations = recommend_products(model_file, df, target_user_id, target_product_id)

# Display recommendations
print("Top 10 Recommendations:")
print(recommendations)
