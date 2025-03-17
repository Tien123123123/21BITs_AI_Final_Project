import pickle
import random
import os
from datetime import datetime
import pytz
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from minio_server.push import push_object
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant
from arg_parse.arg_parse_coldstart import arg_parse_coldstart

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_cold_start_clusters(
    args,
    df,
    bucket_name=None
):

    try:
        # Use provided bucket_name or fall back to args
        bucket_name = bucket_name if bucket_name else args.bucket

        # Validate DataFrame
        if df is None or df.empty:
            logging.error("No data retrieved from Qdrant or DataFrame is empty.")
            raise ValueError("Failed to load data from Qdrant for cold-start training.")

        if not all(col in df.columns for col in ["user_session", "product_id"]):
            logging.error("DataFrame must contain 'user_session' and 'product_id' columns.")
            raise ValueError("DataFrame missing required columns: 'user_session' and 'product_id'.")

        logging.info(f"Data loaded successfully: {len(df)} records")

        # Group sessions: each session gives a list of unique product ids
        session_groups = df.groupby("user_session")["product_id"].apply(lambda x: list(set(x)))

        # Build candidate frequency counts for each target product
        cluster_candidates = {}
        for products in session_groups:
            for target in products:
                cluster_candidates.setdefault(target, {})
                for candidate in products:
                    if candidate == target:
                        continue
                    cluster_candidates[target][candidate] = cluster_candidates[target].get(candidate, 0) + 1

        # Build clusters: for each target, sort candidates by frequency and select top_n and random_n
        cold_start_clusters = {}
        for target, candidates in cluster_candidates.items():
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            sorted_candidate_ids = [cid for cid, freq in sorted_candidates]
            top_candidates = sorted_candidate_ids[:args.top_n]
            remaining = sorted_candidate_ids[args.top_n:]
            other_candidates = random.sample(remaining, min(args.random_n, len(remaining))) if remaining else []
            cold_start_clusters[target] = {"top": top_candidates, "others": other_candidates}

        if not args.save:
            logging.info("Model training completed but not saved as per args.save=False")
            return None

        # Generate timestamped filename with model prefix from args
        timezone = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(timezone)
        formatted_time = now.strftime("%d_%m_%y_%H_%M")
        model_name = f"{args.model}_{formatted_time}.pkl"
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models", model_name))

        # Ensure the models directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model locally
        with open(save_path, "wb") as f:
            pickle.dump(cold_start_clusters, f)
        logging.info(f"✅ Cold start clusters saved locally to {save_path}")

        # Push to MinIO
        bucket_models = bucket_name
        obj_name = model_name
        push_object(bucket_name=bucket_models, file_path=save_path, object_name=obj_name)
        logging.info(f"✅ Cold start clusters pushed to MinIO: {bucket_models}/{obj_name}")

        # Clean up local file
        if os.path.exists(save_path):
            os.remove(save_path)
            logging.info(f"🗑️ Local file {save_path} removed after upload")

        return save_path

    except Exception as e:
        logging.error(f"❌ Error in train_cold_start_clusters: {str(e)}")
        raise

if __name__ == "__main__":
    args = arg_parse_coldstart()
    logging.info("Cold-start pretraining is running...")
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"
    train_cold_start_clusters(
        args,
        q_drant_end_point=q_drant_end_point,
        q_drant_collection_name=q_drant_collection_name,
        bucket_name="models"
    )