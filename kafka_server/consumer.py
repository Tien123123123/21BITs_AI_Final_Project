import logging
import requests
import threading
import time
import json
from kafka import KafkaConsumer
from kafka_server.producer import send_message
from requests.exceptions import RequestException
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from evaluation_pretrain.pretrain_coldstart import train_cold_start_clusters
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from arg_parse.arg_parse_coldstart import arg_parse_coldstart
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to sys.path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Qdrant Configuration
QDRANT_END_POINT = "http://103.155.161.100:6333"
QDRANT_COLLECTION_NAME = "recommendation_system"

# Flask Configuration
FLASK_HOST = os.getenv('FLASK_HOST', 'localhost')
FLASK_PORT = os.getenv('FLASK_PORT', '5000')
FLASK_REFRESH_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/refresh_models"

# Initialize Qdrant client
client = connect_qdrant(end_point=QDRANT_END_POINT, collection_name=QDRANT_COLLECTION_NAME)

def notify_flask(url, retries=3, delay=5):
    """Helper function to notify Flask with retries."""
    for attempt in range(retries):
        try:
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                logging.info(f"✅ Flask models refreshed successfully: {response.json()}")
                return True
            else:
                logging.error(f"❌ Failed to refresh Flask models: {response.status_code} - {response.text}")
        except RequestException as e:
            logging.error(f"❌ Attempt {attempt + 1}/{retries} failed: {str(e)}")
        if attempt < retries - 1:
            time.sleep(delay)
    logging.error("❌ Max retries reached. Could not notify Flask.")
    return False

def perform_pretraining():
    """Helper function to perform pretraining for all models."""
    try:
        # Load data from Qdrant
        logging.info("Loading data from Qdrant...")
        df  = load_to_df(client=client, collection_name=QDRANT_COLLECTION_NAME)
        df1 = df.copy()
        df2 = df.copy()
        df3 = df.copy()
        if df.empty:
            logging.warning("No data loaded from Qdrant, skipping pretraining.")
            return False

        # Pretrain Content-based
        logging.info("Starting Content-based Pretraining...")
        pretrain_contentbase(
            arg_parse_contentbase(),
            df=df1,
            minio_bucket_name="models",
            k=5
        )
        logging.info("Content-based Pretraining completed.")

        # Pretrain Collaborative
        logging.info("Starting Collaborative Pretraining...")
        pretrain_collaborative(
            arg_parse_collaborative(),
            df=df2,
            minio_bucket_name="models"
        )
        logging.info("Collaborative Pretraining completed.")

        # Pretrain Cold-start
        logging.info("Starting Cold-start Pretraining...")
        train_cold_start_clusters(
            args=arg_parse_coldstart(),
            df=df3,
            bucket_name="models"
        )
        logging.info("Cold-start Pretraining completed.")

        # Send "Pretrain complete!" message
        logging.info("Sending 'Pretrain complete!' message to model_pretrain_to_client_event")
        send_message(
            server="kafka.d2f.io.vn:9092",
            topic="model_pretrain_to_client_event",
            message="Pretrain complete!".encode('utf-8')
        )
        return True

    except Exception as e:
        logging.error(f"❌ Error during pretraining: {str(e)}")
        return False

def kafka_consumer(topic_name, bootstrap_servers='kafka.d2f.io.vn:9092'):
    """Kafka consumer to listen for messages on the specified topic."""
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id=f'my-consumer-group-{topic_name}',
            value_deserializer=lambda v: v.decode('utf-8') if v else None,
            max_poll_interval_ms=600000,  # 10 minutes
            max_poll_records=1  # Process one message at a time
        )

        logging.info(f"📡 Đang lắng nghe topic: {topic_name}")

        while True:
            # Poll with a timeout to ensure frequent heartbeats
            message_dict = consumer.poll(timeout_ms=1000)  # 1-second timeout
            if not message_dict:
                logging.debug(f"No messages received from {topic_name} in last poll.")
                continue

            for topic_partition, messages in message_dict.items():
                for message in messages:
                    logging.info(f"Processing message from topic {topic_name}")
                    if message.value is not None:
                        logging.info(f"📢 Nhận được message: {message.value}")
                        logging.info(f"  - Topic: {message.topic}")
                        logging.info(f"  - Partition: {message.partition}")
                        logging.info(f"  - Offset: {message.offset}")
                        logging.info(f"  - Key: {message.key}")

                        if topic_name == "model_retrain_event":
                            logging.info(f"  - Status: Starting pretraining in a separate thread for topic {topic_name}")
                            pretrain_thread = threading.Thread(target=perform_pretraining, daemon=True)
                            pretrain_thread.start()

                        elif topic_name == "model_pretrain_to_client_event":
                            if message.value == "Pretrain complete!":
                                logging.info(f"  - Status: Received pretrain completion message, notifying Flask in a thread.")
                                notify_thread = threading.Thread(
                                    target=notify_flask,
                                    args=(FLASK_REFRESH_URL,),
                                    daemon=True
                                )
                                notify_thread.start()
                            else:
                                logging.info(f"  - Status: Ignoring message on {topic_name}: {message.value}")

                        logging.info("-" * 50)
                    else:
                        logging.info("📢 Message rỗng hoặc không có dữ liệu.")

    except Exception as e:
        logging.error(f"❌ Lỗi khi kết nối hoặc nhận message on topic {topic_name}: {str(e)}")
        raise

if __name__ == "__main__":
    # Start Kafka consumers for both topics in separate threads
    retrain_thread = threading.Thread(target=kafka_consumer, args=("model_retrain_event",), daemon=True)
    pretrain_complete_thread = threading.Thread(target=kafka_consumer, args=("model_pretrain_to_client_event",), daemon=True)

    retrain_thread.start()
    pretrain_complete_thread.start()

    logging.info("✅ Started Kafka consumers for topics 'model_retrain_event' and 'model_pretrain_to_client_event'.")

    # Keep the main thread alive
    try:
        retrain_thread.join()
        pretrain_complete_thread.join()
    except KeyboardInterrupt:
        logging.info("Shutting down Kafka consumers...")