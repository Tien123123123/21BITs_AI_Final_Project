import sys, os
from kafka import KafkaConsumer
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from kafka_server.producer import send_message
import logging
logging.info("Command-line arguments: " + str(sys.argv))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

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
                pretrain_contentbase(arg_parse_contentbase, q_drant_end_point="http://103.155.161.100:6333", q_drant_collection_name="recommendation_system", minio_bucket_name = "models", k=5)
                pretrain_collaborative(arg_parse_collaborative(), q_drant_end_point="http://103.155.161.100:6333", q_drant_collection_name="recommendation_system", minio_bucket_name = "models")
                send_message(message="Pretrain complete !")
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