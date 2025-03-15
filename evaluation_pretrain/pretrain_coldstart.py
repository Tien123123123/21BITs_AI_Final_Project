import pickle
import random
import os
from datetime import datetime
import pytz
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from minio_server.push import push_object
from minio_server.server import create_bucket
from arg_parse.arg_parse_coldstart import arg_parse_coldstart

logging.basicConfig(level=logging.INFO)

def train_cold_start_clusters(args, df, minio_bucket_name="models"):

    try:
        df=df
        # Validate DataFrame
        if df is None or df.empty:
            logging.error("Provided DataFrame is empty or None.")
            raise ValueError("DataFrame is empty or None for cold-start training.")

        if not all(col in df.columns for col in ["user_session", "product_id"]):
            logging.error("DataFrame must contain 'user_session' and 'product_id' columns.")
            raise ValueError("DataFrame missing required columns: 'user_session' and 'product_id'.")

        logging.info(f"Data validated successfully: {len(df)} records")

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

        # Save the cold_start_clusters model to disk
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(cold_start_clusters, f)
            logging.info(f"Model was saved at {save_path}")
        except IOError as e:
            logging.error(f"❌ Failed to save model to {save_path}: {str(e)}")
            raise

        # Push to MinIO
        bucket_models = minio_bucket_name
        create_bucket(bucket_models)  # Ensure the bucket exists
        try:
            push_object(bucket_name=bucket_models, file_path=save_path, object_name=model_name)
            logging.info(f"✅ Cold start clusters pushed to MinIO: {bucket_models}/{model_name}")
        except Exception as e:
            logging.error(f"❌ Failed to push to MinIO: {str(e)}")
            raise

        # Clean up local file only if upload succeeds
        if os.path.exists(save_path):
            os.remove(save_path)
            logging.info(f"🗑️ Local file {save_path} removed after upload")

        return bucket_models, model_name

    except Exception as e:
        logging.error(f"❌ Error in train_cold_start_clusters: {str(e)}")
        raise

if __name__ == "__main__":
    args = arg_parse_coldstart()
    logging.info("Cold-start pretraining is running...")
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"

    client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
    df = load_to_df(client=client, collection_name=q_drant_collection_name)

    train_cold_start_clusters(args, df, minio_bucket_name="models")