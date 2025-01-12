import pandas as pd
import pickle


def recommend_products(model_file, df, target_user_id, target_product_id, top_n=10):
    # Load the trained model
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    # target p_id = 1
    # Filter sessions containing the target product
    filtered_sessions = df[df["product_id"] == target_product_id]["user_session"].unique()
    related_sessions = df[df["user_session"].isin(filtered_sessions)]
    candidate_products = related_sessions["product_id"].unique()

    # Predict scores for the candidate products
    predicted_scores = [
        (product_id, model.predict(target_user_id, product_id).est)
        for product_id in candidate_products
    ]

    # Merge predicted scores with product details
    predicted_scores_df = pd.DataFrame(predicted_scores, columns=["product_id", "predicted_score"])
    product_details = df[["product_id", "category_code", "brand"]].drop_duplicates(subset=["product_id"])
    recommendations = predicted_scores_df.merge(product_details, on="product_id", how="left")

    # Sort by predicted scores
    recommendations = recommendations.sort_values(by="predicted_score", ascending=False)

    return recommendations.head(top_n)
