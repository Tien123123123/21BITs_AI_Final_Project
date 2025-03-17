import pickle
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

    # Display data
    print("df_test.shape sau khi predict:", df_test.shape)
    print("Số user_id duy nhất trong df_test:", df_test["user_id"].nunique())
    print("Thống kê predicted_score:\n", df_test["predicted_score"].describe())

    # Loại bỏ NaN
    df_test = df_test.dropna(subset=["predicted_score"])

    if df_test.empty:
        print("Warning: df_test trống sau khi tính predicted_score!")
        return df_test, pd.DataFrame(), 0.0

    # Each user in df_test will contain only Top 1,2,3 items with high score
    df_test = df_test.groupby("user_id", group_keys=False).apply(
        lambda group: group.nlargest(top_N, "predicted_score")
    ).reset_index(drop=True)

    print("df_test.shape sau khi lọc top N:", df_test.shape)

    # Calculate Precision_All and Recall_All
    total_intersection = 0  # Tổng |Tu ∩ Cu| trên tất cả user
    total_recommended = 0   # Tổng |Cu| (tập dự đoán)
    total_actual = 0        # Tổng |Tu| (tập ground truth)

    for user_id, group in df_test.groupby("user_id"):
        recommended_items = set(group["product_id"])  # Take all user's items (N = 3)
        actual_items = set(df_GT[df_GT["user_id"] == user_id]["product_id"])  # user's GT items

        match_count = len(recommended_items.intersection(actual_items)) # Compare how many predicted items (model) match actual items (GT)

        # Sum total value
        total_intersection += match_count
        total_recommended += len(recommended_items)  # |Cu|
        total_actual += len(actual_items)  # |Tu|
        # print(f"User {user_id}")
        # print(f"total_intersection: {total_intersection}")
        # print(f"total_recommended: {total_recommended}")
        # print(f"total_actual: {total_actual}")


    # Calculate Precision, Recall, F1 for all users
    precision_all = total_intersection / total_recommended if total_recommended > 0 else 0
    recall_all = total_intersection / total_actual if total_actual > 0 else 0
    f1_score = 2 * precision_all * recall_all / (precision_all + recall_all) if precision_all + recall_all > 0 else 0

    print(f"Precision All: {precision_all:.4f}")
    print(f"Recall All: {recall_all:.4f}")
    print(f"F1-score: {f1_score:.4f}")

    return df_test, f1_score



if __name__ == '__main__':
    # Load and preprocess data
    # write data.csv on the root below
    root = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece/data/one_data.csv"
    df = pd.read_csv(root, nrows=5000000)

    # q_drant_end_point = "http://103.155.161.100:6333"
    # q_drant_collection_name = "recommendation_system"
    # end_point = q_drant_end_point
    # client = connect_qdrant(end_point=end_point, collection_name=q_drant_collection_name)
    # df = load_to_df(client=client, collection_name=q_drant_collection_name)

    df = preprocess_data(df, is_encoded=True, nrows=None)
    df_weighted = calc_score(df, 0.1, 0.3, 0.4, 0.2)

    df_test, df_weighted, df_GT = train_test_split(df, df_weighted)

    # Train model
    file_path = "D:\Pycharm\Projects\pythonProject\AI\ML\Projects\Recommendation_Ecomerece\models/collaborative_09_03_25_22_07.pkl"

    # Tải mô hình từ file .pkl
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    # Evaluate model
    df_test, f1_score = evaluate_model(df_test, df_GT, model, top_N=3)
