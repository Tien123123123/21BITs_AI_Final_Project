import os
import pickle
import sys
from datetime import datetime
import pytz
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from process_data.preprocessing import preprocess_data
import argparse
from  minio_server.server import create_bucket
from minio_server.push import push_object
from content_base.hybrid import content_base
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
import logging
logging.basicConfig(level=logging.INFO)

def pretrain_contentbase(args, q_drant_end_point, q_drant_collection_name, minio_bucket_name = "models", k=False):
    # Load data
    end_point = q_drant_end_point
    client = connect_qdrant(end_point=end_point, collection_name=q_drant_collection_name)
    df = load_to_df(client=client, collection_name=q_drant_collection_name)

    # Preprocess and Split data
    df = preprocess_data(df, is_encoded=False, nrows=None)
    selected_features = ["name", "product_id", "category_code", "brand", "price"]
    df_content = df[selected_features].drop_duplicates(subset=['product_id'])
    df_content = df_content.sort_values(by="product_id", ascending=False)
    logging.info("Data Preprocessing Complete")

    # Train Model
    model = content_base(df_content, k=k if k else args.k)
    logging.info("Training Model Complete")

    # Save model
    timezone = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(timezone)
    formatted_time = now.strftime("%d_%m_%y_%H_%M")
    model_name = f"content_base_{formatted_time}.pkl"
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logging.info(f"Created directory: {save_dir}")
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_name}"))
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    if os.path.exists(save_path):
        logging.info(f"Model was saved at {save_path}")
        bucket_models = minio_bucket_name
        create_bucket(bucket_models)
        push_object(bucket_name=bucket_models, file_path=save_path, object_name=model_name)
        return bucket_models, model_name


if __name__ == '__main__':
    args = arg_parse_contentbase()
    logging.info("Content Base is running...")
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"
    pretrain_contentbase(args, q_drant_end_point, q_drant_collection_name, minio_bucket_name = "models", k=False)
