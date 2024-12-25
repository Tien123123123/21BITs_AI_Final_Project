from itertools import islice
import kagglehub
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.preprocessing import LabelEncoder
import logging
from Practice.huggingface import predictions


logging.getLogger('surprise').setLevel(logging.CRITICAL)
path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")

# Confirm path to dataset files
print("Path to dataset files:", path)

# Find the CSV file in the downloaded directory
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file_path = os.path.join(path, file)
        break

# Load a sample of the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, nrows=100)

drop_features = ["category_id", "price", "event_time", "event_type", "user_id", "user_session"]
df = df.drop(drop_features, axis=1)
df = df.dropna()
enc = LabelEncoder()
df["product_id"]=enc.fit_transform(df["product_id"])
def adjust_category(loc):
    return str(loc).replace(".", " ")
df["category_code"] = df["category_code"].apply(adjust_category)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(df)
category_vec = vectorizer.fit_transform(df["category_code"])
brand_vec = vectorizer.fit_transform(df["brand"])

cos_sm = cosine_similarity(category_vec)

category_similarity_df = pd.DataFrame(cos_sm, index=df["product_id"], columns=df["product_id"])
product_id = 3
related_items = category_similarity_df[product_id].sort_values(ascending=False)[1:10]
product_idxs = related_items.index

df_related_items = df[df["product_id"].isin(product_idxs)]
model = joblib.load('svd_model.pkl')

predictions_dict = {}
user_id = 1
for idx in product_idxs:
    prediction = model.predict(user_id, idx)
    predictions_dict[idx] = prediction.est

sorted_predictions = dict(islice(sorted(predictions_dict.items(), key=lambda item: item[1], reverse=True), 5))
print([(product, score) for product, score in sorted_predictions.items()])