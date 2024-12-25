import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Download and load the dataset
path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")

# Confirm path to dataset files
print("Path to dataset files:", path)

# Find the CSV file in the downloaded directory
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file_path = os.path.join(path, file)
        break

# Load a sample of the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, nrows=100000)

# Step 2: Data Preprocessing
drop_features = ["category_id", "price"]
df = df.drop(drop_features, axis=1)
df = df.replace([0, '0'], np.nan)
df = df.dropna()
def adjust_category(loc):
    return str(loc).replace(".", " ")
df["category_code"] = df["category_code"].apply(adjust_category)

enc = LabelEncoder()
df["product_id"] = enc.fit_transform(df["product_id"])
df["user_id"] = enc.fit_transform(df["user_id"])
df["user_session"] = enc.fit_transform(df["user_session"])

# Calculate Total view, cart, purchase
is_view = df[df["event_type"] == "view"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(name="is_view")
is_cart = df[df["event_type"] == "cart"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(name="is_cart")
is_purchase = df[df["event_type"] == "purchase"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(name="is_purchase")

total_view = df[df["event_type"] == "view"].groupby(["user_id", "user_session"]).size().reset_index(name="total_view")
total_cart = df[df["event_type"] == "cart"].groupby(["user_id", "user_session"]).size().reset_index(name="total_cart")
total_purchase = df[df["event_type"] == "purchase"].groupby(["user_id", "user_session"]).size().reset_index(name="total_purchase")

df = (df.merge(is_view, on=["user_id", "user_session", "product_id"], how="left")
        .merge(is_cart, on=["user_id", "user_session", "product_id"], how="left")
        .merge(is_purchase, on=["user_id", "user_session", "product_id"], how="left")
        .merge(total_view, on=["user_id", "user_session"], how="left")
        .merge(total_cart, on=["user_id", "user_session"], how="left")
        .merge(total_purchase, on=["user_id", "user_session"], how="left"))
df.fillna(0, inplace=True)

# Calculate Duration and Total Duration by seconds
df = df.sort_values(by=["user_id", "user_session", "event_time"], ascending=[True, True, True])
df["event_time"] = pd.to_datetime(df["event_time"])
df["duration"] = df.groupby(["user_id", "user_session"])["event_time"].diff().fillna(pd.Timedelta(seconds=0))
single_item_sessions = df.groupby(["user_id", "user_session"]).size() == 1
sessions_to_modify = df[df[["user_id", "user_session"]].apply(tuple, axis=1).isin(single_item_sessions[single_item_sessions].index)]
df.loc[sessions_to_modify.index, "duration"] = pd.Timedelta(seconds=900)

df["duration"] = df["duration"].dt.total_seconds()
df["total_duration"] = df.groupby(["user_id", "user_session"])["duration"].transform("sum")

# F1: View - F2: Cart - F3: Purchase - F4: Duration
df['F1'] = df.apply(lambda row: row['is_view'] / row['total_view'] if row['event_type'] == 'view' and row['total_view'] != 0 else 0, axis=1)
df['F2'] = df.apply(lambda row: row['is_cart'] / row['total_cart'] if row['event_type'] == 'cart' and row['total_cart'] != 0 else 0, axis=1)
df['F3'] = df.apply(lambda row: row['is_purchase'] / row['total_purchase'] if row['event_type'] == 'purchase' and row['total_purchase'] != 0 else 0, axis=1)
df['F4'] = df.apply(lambda row: row['duration'] / row['total_duration'] if row['total_duration'] != 0 else 0, axis=1)

# Calculate Interesting Score of User in each Item
initial_weights = {"F1": 0.1, "F2": 0.25, "F3": 0.45, "F4": 0.2}
df["score"] = (initial_weights["F1"] * df["F1"] +
               initial_weights["F2"] * df["F2"] +
               initial_weights["F3"] * df["F3"] +
               initial_weights["F4"] * df["F4"])
df_weighted = df.groupby(["user_id", "product_id"])["score"].sum().reset_index(name="score")
selected_features = ["product_id", "category_code", "brand"]
df_content = df[selected_features].drop_duplicates(subset=['product_id'])

# Content Base
vectorizer = TfidfVectorizer()
vectorizer_mat = vectorizer.fit_transform(df_content["category_code"])
cos_sm_content = cosine_similarity(vectorizer_mat)
df_content_base = pd.DataFrame(data=cos_sm_content, index=df_content["product_id"], columns=df_content["product_id"])

# Item Item Base
pivot_tb = df_weighted.pivot_table(index="user_id", columns="product_id", values="score").fillna(0)
cos_sm_item = cosine_similarity(pivot_tb.values.T)
df_item_base = pd.DataFrame(data=cos_sm_item, index=pivot_tb.columns, columns=pivot_tb.columns)

# Combine
product_id = 1
content_w = 0.7
item_w = 0.3

filtered_content = df_content_base[product_id].sort_values(ascending=False)
filtered_item = df_item_base[product_id].sort_values(ascending=False)
combined_related_items = filtered_content*content_w + filtered_item*item_w
combined_related_items = combined_related_items.reset_index()
combined_related_items.columns = ["product_id", "sim_score"]

df_combined = combined_related_items.merge(df_content, on="product_id", how="left")
df_combined = df_combined.sort_values(by="sim_score", ascending=False)

print(df_combined.head(30))