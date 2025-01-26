import pandas as pd
from process_data.preprocessing import preprocess_data
from surprise import Reader, Dataset, accuracy
import pickle

# Load and preprocess data
root = "data/one_data.csv"

df_main, _ = preprocess_data(root, nrows=1000000)

# Extract ground truth (GT) where users made purchases
df_GT = df_main[df_main["event_type"] == "purchase"].drop_duplicates(subset=["user_id", "product_id"], keep="first")
is_buy = df_GT[df_GT["event_type"] == "purchase"].groupby(["user_id"]).size().reset_index(name="total_buy")
print(df_GT.head())

# Split GT into half to create test set
test_set = df_GT.sample(frac=0.5, random_state=123)[["user_id", "product_id"]]

# Load model
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

# Predict scores
test_set["predicted_score"] = test_set.apply(
    lambda row: model.predict(row["user_id"], row["product_id"]).est, axis=1
)

# Sort and get top N=5 items per user
top_N = 3
test_set = (
    test_set.groupby("user_id", group_keys=False)
    .apply(lambda group: group.nlargest(top_N, "predicted_score"))
)

# Calculate precision
precision_list = []
for user_id, group in test_set.groupby("user_id"):
    recommended_items = set(group["product_id"])
    actual_items = set(df_GT[df_GT["user_id"] == user_id]["product_id"])

    match_count = len(recommended_items.intersection(actual_items))
    precision = match_count / min(len(actual_items), top_N)  # Only take enough items in GT
    precision_list.append({"user_id": user_id, "precision": precision})

# Convert precision results to DataFrame
df_precision = pd.DataFrame(precision_list)

print(df_precision)