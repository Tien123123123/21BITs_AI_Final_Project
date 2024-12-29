import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_data

root = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation-Ecomerece\dataset\one_mil_data.csv"
df, df_weighted = preprocess_data(root, nrows=50000)

selected_features = ["product_id", "category_code", "brand"]
df_content = df[selected_features].drop_duplicates(subset=['product_id'])

# Content Base
def Content_Base():
    vectorizer = TfidfVectorizer()
    vectorizer_mat = vectorizer.fit_transform(df_content["category_code"])
    cos_sm_content = cosine_similarity(vectorizer_mat)
    df_content_base = pd.DataFrame(data=cos_sm_content, index=df_content["product_id"], columns=df_content["product_id"])
    return df_content_base

# Item Item Base
def Item_Item():
    pivot_tb = df_weighted.pivot_table(index="user_id", columns="product_id", values="score").fillna(0)
    cos_sm_item = cosine_similarity(pivot_tb.values.T)
    df_item_base = pd.DataFrame(data=cos_sm_item, index=pivot_tb.columns, columns=pivot_tb.columns)
    return df_item_base

# Combine
# product_id = 1003461
# content_w = 0.7
# item_w = 0.3
# k_out = 10

def Combined(product_id, content_w, item_w, k_out):
    df_content_base = Content_Base()
    df_item_base = Item_Item()
    filtered_content = df_content_base[product_id].sort_values(ascending=False)
    filtered_item = df_item_base[product_id].sort_values(ascending=False)
    combined_related_items = filtered_content*content_w + filtered_item*item_w
    combined_related_items = combined_related_items.reset_index()
    combined_related_items.columns = ["product_id", "sim_score"]

    df_combined = combined_related_items.merge(df_content, on="product_id", how="left")
    df_combined = df_combined.sort_values(by="sim_score", ascending=False)
    return df_combined[["product_id", "sim_score", "category_code", "brand"]].head(k_out).to_dict(orient="records")

# result = Combined(product_id, content_w, item_w, k_out)
# print(result)