import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Practice.implicit_test import product_id
from preprocessing import preprocess_data
import pickle

def Content_Base():
    df_content['combined_features'] = df_content['name'] + " " + df_content['category_code'] + " " + df_content['brand']
    vectorizer = TfidfVectorizer()
    vectorizer_mat = vectorizer.fit_transform(df_content['combined_features'])
    cos_sm_content = cosine_similarity(vectorizer_mat)
    df_content_base = pd.DataFrame(data=cos_sm_content, index=df_content["product_id"], columns=df_content["product_id"])
    return df_content_base


def select_top_k(recommend_type, top_k, save_path):
    # Save data into dict
    df_sim = recommend_type
    top_k_similarity = {}
    for p_id in df_sim.index:
        similarity_items = df_sim.loc[p_id, :]
        result = similarity_items.sort_values(ascending=False)
        if p_id in result:
            result = result.drop(p_id)
        result = result[:top_k]
        result = result.to_dict()
        top_k_similarity[p_id] = result

    # Save dict as module file
    with open(save_path, 'wb') as f:
        pickle.dump(top_k_similarity, f)

    return f"Module is saved successfully at {save_path}"

if __name__ == '__main__':
    root = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation-Ecomerece\data/merged_data.csv"
    df, df_weighted = preprocess_data(root, nrows=100)

    selected_features = ["product_id", "name", "category_code", "brand"]
    df_content = df[selected_features].drop_duplicates(subset=['product_id'])
    df_content = df_content.sort_values(by="product_id", ascending=False)

    module = Content_Base()
    top_k = 100
    save_path = "models/top_k_content_base.pkl"
    optimized_module = select_top_k(recommend_type=module, top_k= top_k, save_path= save_path)

    unique_product_ids = df_content['product_id'].unique()
    unique_product_ids = unique_product_ids.tolist()
    print(f"p_ids for texting {unique_product_ids}")
