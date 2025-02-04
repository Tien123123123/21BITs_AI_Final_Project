import pandas as pd
from process_data.preprocessing import preprocess_data
from process_data.train_test_split import train_test_split
from collaborative.train_model import train_model

def evaluate_model(df_test, model, top_N):
    # Evaluate data
    df_test["predicted_score"] = df_test.apply(
        lambda x: model.predict(x["user_id"], x["product_id"]).est, axis=1
    )

    # Reset index to avoid user_id is row and col
    df_test = df_test.reset_index()

    # Take out top N products of that user
    top_N = top_N
    df_test = (
        df_test.groupby("user_id", group_keys=False)
        .apply(lambda group: group.nlargest(top_N, "predicted_score"))
    )

    # Calculate precision recall f1
    metrics_list = []
    for user_id, group in df_test.groupby("user_id"):
        recommended_items = set(group["product_id"])  # list predicted product of user
        actual_items = set(df_GT[df_GT["user_id"] == user_id]["product_id"])  # list actual product of user

        match_count = len(recommended_items.intersection(actual_items))  # compare 2 list
        precision = match_count / top_N
        recall = match_count / len(actual_items)
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics_list.append({
            "user_id": user_id,
            "precision_score": precision,
            "recall_score": recall,
            "f1_score": f1_score,
            "match_count": match_count
        })

    # Convert precision results to DataFrame
    df_metrics = pd.DataFrame(metrics_list)
    # print(f"Mean value: {df_metrics['f1_score'].mean()}")

    return df_test, df_metrics, df_metrics['f1_score'].mean()

if __name__ == '__main__':
    # Load and preprocess data
    # write data.csv on the root below
    root = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece/data/one_data.csv"
    df, df_weighted = preprocess_data(root, nrows=1000000)
    df_test, df_weighted, df_GT = train_test_split(df, df_weighted)
    # Train model
    model, _ = train_model(df_weighted)
    # Evaluate model
    df_test, df_metrics, mean = evaluate_model(df_test, model, top_N=3)

    print(df_test.head())
    print(df_metrics.head())
