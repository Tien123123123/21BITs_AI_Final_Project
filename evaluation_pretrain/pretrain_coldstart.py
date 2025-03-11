# evaluation_pretrain/pretrain_coldstart.py
import pickle
import random
import os
from datetime import datetime
import pytz
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from minio_server.push import push_object
from arg_parse.arg_parse_coldstart import arg_parse_coldstart

def train_cold_start_clusters(args, df=None, bucket_name=False):
    """
    Train cold-start clusters and push to MinIO based on args.

    Args:
        args: Parsed arguments from arg_parse_coldstart
        df: DataFrame with user_session and product_id columns (optional if loading from MinIO)
        bucket_name: MinIO bucket name to store the model (overrides args.bucket if provided)
    """
    try:
        # Use provided bucket_name or fall back to args
        bucket_name = bucket_name if bucket_name else args.bucket

        # If no DataFrame is provided, raise an error (assuming Flask loads it)
        if df is None:
            raise ValueError("DataFrame must be provided for cold-start training")

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
            return None  # Return None if not saving

        # Generate timestamped filename with model prefix from args
        timezone = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(timezone)
        formatted_time = now.strftime("%d_%m_%y_%H_%M")
        model_name = f"{args.model}_{formatted_time}.pkl"
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_name}"))

        # Ensure the models directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model locally
        with open(save_path, "wb") as f:
            pickle.dump(cold_start_clusters, f)
        logging.info(f"Cold start clusters saved locally to {save_path}")

        # Push to MinIO
        bucket_models = bucket_name
        obj_name = model_name
        push_object(bucket_name=bucket_models, file_path=save_path, object_name=obj_name)
        logging.info(f"Cold start clusters pushed to MinIO: {bucket_models}/{obj_name}")

        # Clean up local file
        if os.path.exists(save_path):
            os.remove(save_path)
            logging.info(f"Local file {save_path} removed after upload")

        return save_path

    except Exception as e:
        logging.error(f"Error in train_cold_start_clusters: {str(e)}")
        raise

if __name__ == "__main__":
    import pandas as pd
    args = arg_parse_coldstart()
    print("Cold-start pretraining is running...")
    df = pd.DataFrame({
        "user_session": ["s1", "s1", "s2", "s2"],
        "product_id": [1, 2, 2, 3]
    })
    train_cold_start_clusters(args, df=df, bucket_name="models")