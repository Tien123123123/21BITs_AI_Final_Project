import os
import pickle
import sys
from datetime import datetime
import pytz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from process_data.preprocessing import preprocess_data
import argparse
from minio_server.push import push_object
from content_base.hybrid import content_base

def arg_parse_contentbase():
    parser = argparse.ArgumentParser("Collaborative Pretrain Process!")
    parser.add_argument("--save", "-s", type=bool, default="True", help="Save model after training and evaluating", required=False)
    parser.add_argument("--bucket", "-b", type=str, default="recommendation", help="minio bucket name",
                        required=False)
    parser.add_argument("--data", "-d", type=str, default="dataset.csv", help="file path and name of dataset on bucket", required=False)
    parser.add_argument("--model", "-m", type=str, default="contentbase", help="Name of model after saving",
                        required=False)
    parser.add_argument("-k", type=int, default=10, help="Total recommended items for a input item",
                        required=False)
    args = parser.parse_args()
    return args

def pretrain_contentbase(args, bucket_name=False, dataset=False, k=False):
    # Load and Split data
    bucket_name = bucket_name if bucket_name else args.bucket
    file_name = dataset if dataset else args.data
    df, df_weighted = preprocess_data(bucket_name, file_name, is_encoded=True, nrows=100000)
    selected_features = ["name", "product_id", "category_code", "brand", "price"]
    df_content = df[selected_features].drop_duplicates(subset=['product_id'])
    df_content = df_content.sort_values(by="product_id", ascending=False)
    print("Data Preprocessing Complete")

    # Train Model
    model = content_base(df_content, k=k if k else args.k)
    print("Training Model Complete")

    # Save model
    timezone = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(timezone)
    formatted_time = now.strftime("%d_%m_%y_%H_%M")
    model_name = f"content_base_{formatted_time}.pkl"
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_name}"))
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    if os.path.exists(save_path):
        print(f"Model was saved at {save_path}")
        bucket_models = "models"
        push_object(bucket_name=bucket_models, file_path=save_path, object_name=model_name)
        # return bucket_models, model_name


if __name__ == '__main__':
    args = arg_parse_contentbase()
    print("Content Base is running...")
    pretrain_contentbase(args, bucket_name="recommendation", dataset="merged_data_second.csv", k=10)
