import sys, os
from kafka import KafkaConsumer
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from arg_parse.arg_parse_coldstart import arg_parse_coldstart  # Add import for cold-start args
from kafka_server.producer import send_message
import logging
import pickle
import random
from datetime import datetime
import pytz
from minio_server.push import push_object
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Command-line arguments: " + str(sys.argv))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

def train_cold_start_clusters(
    args,
    q_drant_end_point="http://103.155.161.100:6333",
    q_drant_collection_name="recommendation_system",
    bucket_name=None
):
    """
    Train cold-start clusters using data from Qdrant and push to MinIO based on args.

    Args:
        args: Parsed arguments from arg_parse_coldstart
        q_drant_end_point: Qdrant server endpoint (default: "http://103.155.161.100:6333")
        q_drant_collection_name: Qdrant collection name (default: "recommendation_system")
        bucket_name: MinIO bucket name to store the model (overrides args.bucket if provided)

    Returns:
        str: Path to the saved model file (if saved), None otherwise
    """
    try:
        # Use provided bucket_name or fall back to args
        bucket_name = bucket_name if bucket_name else args.bucket

        # Connect to Qdrant and load data
        logging.info(f"Connecting to Qdrant at {q_drant_end_point}...")
        client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
        logging.info(f"Loading data from Qdrant collection: {q_drant_collection_name}")
        df = load_to_df(client=client, collection_name=q_drant_collection_name)

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

def kafka_consumer(topic_name, bootstrap_servers='kafka.d2f.io.vn:9092'):
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='my-consumer-group',
            value_deserializer=lambda v: v.decode('utf-8') if v else None
        )

        logging.info(f"📡 Đang lắng nghe topic: {topic_name}")

        for message in consumer:
            if message.value is not None:
                logging.info(f"📢 Nhận được message: {message.value}")
                logging.info(f"  - Topic: {message.topic}")
                logging.info(f"  - Partition: {message.partition}")
                logging.info(f"  - Offset: {message.offset}")
                logging.info(f"  - Key: {message.key}")
                logging.info(f"  - Status: Ready to Pretrain!")

                # Pretrain Content-based
                logging.info("Starting Content-based Pretraining...")
                pretrain_contentbase(
                    arg_parse_contentbase(),
                    q_drant_end_point="http://103.155.161.100:6333",
                    q_drant_collection_name="recommendation_system",
                    minio_bucket_name="models",
                    k=5
                )
                logging.info("Content-based Pretraining completed.")

                # Pretrain Collaborative
                logging.info("Starting Collaborative Pretraining...")
                pretrain_collaborative(
                    arg_parse_collaborative(),
                    q_drant_end_point="http://103.155.161.100:6333",
                    q_drant_collection_name="recommendation_system",
                    minio_bucket_name="models"
                )
                logging.info("Collaborative Pretraining completed.")

                # Pretrain Cold-start
                logging.info("Starting Cold-start Pretraining...")
                train_cold_start_clusters(
                    args=arg_parse_coldstart(),
                    q_drant_end_point="http://103.155.161.100:6333",
                    q_drant_collection_name="recommendation_system",
                    bucket_name="models"
                )
                logging.info("Cold-start Pretraining completed.")

                # Send message after all pretraining steps are complete
                send_message(message="Pretrain complete!")
                logging.info("-" * 50)
            else:
                logging.info("📢 Message rỗng hoặc không có dữ liệu.")

    except Exception as e:
        logging.info(f"❌ Lỗi khi kết nối hoặc nhận message: {e}")

if __name__ == "__main__":
    kafka_consumer(
        topic_name="model_pretrain_to_client_event",
        bootstrap_servers="kafka.d2f.io.vn:9092"
    )
