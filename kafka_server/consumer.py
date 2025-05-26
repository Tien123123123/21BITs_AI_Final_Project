import sys
import os
import json
import threading
import requests
from datetime import datetime
from kafka import KafkaConsumer, TopicPartition
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from evaluation_pretrain.pretrain_coldstart import train_cold_start_clusters
from evaluation_pretrain.pretrain_association import pretrain_association
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from arg_parse.arg_parse_coldstart import arg_parse_coldstart
import logging
from qdrant_server.load_data import load_to_df
from process_data.preprocessing import preprocess_data
from qdrant_server.server import connect_qdrant
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

KAFKA_END_POINT = os.getenv('KAFKA_END_POINT', 'kafka.d2f.io.vn:9092')
FLASK_END_POINT = os.getenv('FLASK_END_POINT', 'http://localhost:5000')
QDRANT_END_POINT = os.getenv('QDRANT_END_POINT', 'http://103.155.161.100:6333')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'test_collection')

def kafka_consumer(topic_name, bootstrap_servers=KAFKA_END_POINT, flask_url=FLASK_END_POINT):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Command-line arguments: {sys.argv}")

    q_drant_end_point = QDRANT_END_POINT
    q_drant_collection_name = QDRANT_COLLECTION_NAME
    try:
        client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
        logging.info(f"✅ Đã kết nối đến Qdrant tại {q_drant_end_point}")
    except Exception as e:
        logging.error(f"❌ Lỗi khi kết nối Qdrant: {str(e)}", exc_info=True)
        return

    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=False,
            group_id=f'my-consumer-group-{datetime.utcnow().timestamp()}',
            value_deserializer=lambda v: v.decode('utf-8') if v else None,
            max_poll_interval_ms=600000,
            max_poll_records=1,
            session_timeout_ms=60000,
            heartbeat_interval_ms=20000
        )
        logging.info(f"📡 Đang lắng nghe topic: {topic_name} để đợi message mới...")
    except Exception as e:
        logging.error(f"❌ Lỗi khi khởi tạo Kafka consumer: {str(e)}", exc_info=True)
        return

    try:
        consumer.poll(timeout_ms=1000)
        initial_offsets = {TopicPartition(topic_name, p): consumer.position(TopicPartition(topic_name, p))
                          for p in consumer.partitions_for_topic(topic_name)}
        for tp, offset in initial_offsets.items():
            logging.info(f"Partition {tp.partition} - Offset cuối cùng khi khởi động: {offset}")
    except Exception as e:
        logging.error(f"❌ Lỗi khi lấy offset ban đầu: {str(e)}", exc_info=True)
        return

    while True:
        try:
            message_dict = consumer.poll(timeout_ms=1000)
            if not message_dict:
                logging.debug(f"No messages received from {topic_name} in last poll.")
                continue

            for topic_partition, messages in message_dict.items():
                for message in messages:
                    if message.value is None:
                        logging.info("📢 Message rỗng hoặc không có dữ liệu.")
                        consumer.commit()
                        continue

                    current_offset = message.offset
                    tp = TopicPartition(message.topic, message.partition)

                    if current_offset >= initial_offsets[tp]:
                        logging.info(f"📢 Nhận được message mới: {message.value}")
                        logging.info(f"  - Topic: {message.topic}")
                        logging.info(f"  - Partition: {message.partition}")
                        logging.info(f"  - Offset: {message.offset}")
                        logging.info(f"  - Key: {message.key}")
                        logging.info(f"  - Status: Ready to Pretrain!")

                        try:
                            msg_data = json.loads(message.value)
                            event_time = msg_data.get('event_time')
                            if event_time:
                                event_dt = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S UTC")
                                now = datetime.utcnow()
                                if (now - event_dt).total_seconds() > 3600:
                                    logging.info(f"Message tại offset {current_offset} quá cũ ({event_time}), bỏ qua.")
                                    consumer.commit()
                                    initial_offsets[tp] = current_offset + 1
                                    continue
                        except (json.JSONDecodeError, ValueError) as e:
                            logging.warning(f"Không thể parse event_time từ message: {str(e)}")

                        def process_message():
                            start_time = datetime.utcnow()
                            logging.info("Starting pretraining process...")

                            try:
                                df = load_to_df(client=client, collection_name=q_drant_collection_name)
                                logging.info(f"Data validated successfully: {len(df)} records")
                                unique_users= df['user_id'].nunique()
                                logging.info(f"unique user: {unique_users}")
                                unique_products= df['product_id'].nunique()
                                logging.info(f"unique product: {unique_products}")

                                df = preprocess_data(df, is_encoded=False, nrows=None)
                         
                                start_pretrain = datetime.utcnow()
                                pretrain_contentbase(args=arg_parse_contentbase(), df=df)
                                logging.info(f"pretrain_contentbase took {(datetime.utcnow() - start_pretrain).total_seconds()} seconds")

                                start_pretrain = datetime.utcnow()
                                pretrain_collaborative(args=arg_parse_collaborative(), df=df)
                                logging.info(f"pretrain_collaborative took {(datetime.utcnow() - start_pretrain).total_seconds()} seconds")

                                start_pretrain = datetime.utcnow()
                                train_cold_start_clusters(args=arg_parse_coldstart(), df=df)
                                logging.info(f"pretrain_contentbase took {(datetime.utcnow() - start_pretrain).total_seconds()} seconds")

                                # start_pretrain = datetime.utcnow()
                                # df_electronics = df[df["category_code"].str.startswith("electronics")]
                                # logging.info(f"Filtered {len(df_electronics)} records in 'electronics' category for association pretraining.")
                                # pretrain_association(df=df_electronics)
                                # logging.info(f"pretrain_association took {(datetime.utcnow() - start_pretrain).total_seconds()} seconds")

                                logging.info(f"Total pretraining process took {(datetime.utcnow() - start_time).total_seconds()} seconds")

                                response = requests.post(f"{flask_url}/refresh_models")
                                if response.status_code == 200:
                                    logging.info(f"✅ Successfully refreshed models via {flask_url}/refresh_models")
                                else:
                                    logging.warning(f"⚠️ Failed to refresh models. Status code: {response.status_code}, Response: {response.text}")
                            except Exception as e:
                                logging.error(f"❌ Error during pretraining or model refresh: {str(e)}", exc_info=True)

                        pretrain_thread = threading.Thread(target=process_message, daemon=True)
                        pretrain_thread.start()

                        logging.info("-" * 50)
                        consumer.commit()
                        initial_offsets[tp] = current_offset + 1
                    else:
                        logging.debug(f"Offset {current_offset} không phải message mới, bỏ qua.")
                        consumer.commit()

        except Exception as e:
            logging.error(f"❌ Lỗi trong vòng lặp consumer: {str(e)}", exc_info=True)
            consumer.commit()
            continue

if __name__ == "__main__":
    kafka_consumer(
        topic_name="model_retrain_event",
        bootstrap_servers="kafka:29092",
        flask_url="https://ai.d2f.io.vn"
    )
