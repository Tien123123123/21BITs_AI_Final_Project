import pandas as pd

from process_data.calc_score import calc_score
from process_data.preprocessing import preprocess_data
from process_data.train_test_split import train_test_split
from collaborative.train_model import train_model
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant


def evaluate_model(df_test, df_GT, model, top_N):
    # Predict scores
    df_test["predicted_score"] = df_test.apply(
        lambda x: model.predict(x["user_id"], x["product_id"]).est, axis=1
    )

    # Kiểm tra dữ liệu sau khi dự đoán
    print("df_test.shape sau khi predict:", df_test.shape)
    print("Số user_id duy nhất trong df_test:", df_test["user_id"].nunique())
    print("Thống kê predicted_score:\n", df_test["predicted_score"].describe())

    # Loại bỏ NaN
    df_test = df_test.dropna(subset=["predicted_score"])

    if df_test.empty:
        print("Warning: df_test trống sau khi tính predicted_score!")
        return df_test, pd.DataFrame(), 0.0  # Trả về F1-score = 0 nếu không có dữ liệu hợp lệ

    # Lọc top N sản phẩm theo user
    df_test = df_test.groupby("user_id", group_keys=False).apply(
        lambda group: group.nlargest(top_N, "predicted_score")
    ).reset_index(drop=True)

    print("df_test.shape sau khi lọc top N:", df_test.shape)

    # Tính Precision, Recall, F1-score
    metrics_list = []
    for user_id, group in df_test.groupby("user_id"):
        recommended_items = set(group["product_id"])
        actual_items = set(df_GT[df_GT["user_id"] == user_id]["product_id"])

        match_count = len(recommended_items.intersection(actual_items))
        precision = match_count / top_N
        recall = match_count / len(actual_items) if len(actual_items) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics_list.append({
            "user_id": user_id,
            "precision_score": precision,
            "recall_score": recall,
            "f1_score": f1_score,
            "match_count": match_count
        })

    # Chuyển kết quả thành DataFrame
    df_metrics = pd.DataFrame(metrics_list)

    if df_metrics.empty:
        print("Warning: df_metrics is empty! No f1_score calculated.")
        return df_test, df_metrics, 0.0  # Trả về F1-score = 0 nếu không có dữ liệu hợp lệ

    return df_test, df_metrics, df_metrics['f1_score'].mean()



if __name__ == '__main__':
    # Load and preprocess data
    # write data.csv on the root below
    # root = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece/data/one_data.csv"
    # df = pd.read_csv(root, nrows=500000)

    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"
    end_point = q_drant_end_point
    client = connect_qdrant(end_point=end_point, collection_name=q_drant_collection_name)
    df = load_to_df(client=client, collection_name=q_drant_collection_name)

    df = preprocess_data(df, is_encoded=True, nrows=None)
    df, df_weighted = calc_score(df, 0.1, 0.3, 0.4, 0.2)
    print("df")
    print(df.head(10))
    print("df weighted")
    print(df_weighted.head(10))
    df_test, df_weighted, df_GT = train_test_split(df, df_weighted)
    print("-"*10)
    print("df test")
    print(df_test.head(1))
    print("-" * 10)
    print("df weighted")
    print(df_weighted.head(1))
    print("-" * 10)
    print("df gt")
    print(df_GT.head(1))
    print("-" * 10)
    # Train model
    model, _ = train_model(df_weighted)
    # Evaluate model
    df_test, df_metrics, mean = evaluate_model(df_test, df_GT, model, top_N=3)

    print(df_test.head())
    print(df_metrics.head())
