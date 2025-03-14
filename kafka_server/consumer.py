import sys, os
from kafka import KafkaConsumer
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from kafka_server.producer import send_message
import logging
from qdrant_server.load_data import load_to_df
from process_data.preprocessing import preprocess_data
from qdrant_server.server import connect_qdrant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


def kafka_consumer(topic_name, bootstrap_servers='kafka.d2f.io.vn:9092'):
    try:
        consumer = KafkaConsumer(
            topic_name,  # Đăng ký topic trực tiếp
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',  # Chỉ lấy message mới từ thời điểm bắt đầu chạy
            enable_auto_commit=True,
            group_id='my-consumer-group',
            value_deserializer=lambda v: v.decode('utf-8') if v else None
        )

        logging.info(f"📡 Đang lắng nghe topic: {topic_name} để đợi message mới nhất...")

        # Lắng nghe message mới liên tục
        for message in consumer:
            if message.value is not None:
                logging.info(f"📢 Nhận được message mới nhất: {message.value}")
                logging.info(f"  - Topic: {message.topic}")
                logging.info(f"  - Partition: {message.partition}")
                logging.info(f"  - Offset: {message.offset}")
                logging.info(f"  - Key: {message.key}")
                logging.info(f"  - Status: Ready to Pretrain!")

                # Kết nối Qdrant và xử lý dữ liệu
                q_drant_end_point = "http://103.155.161.100:6333"
                q_drant_collection_name = "recommendation_system"
                client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
                df = load_to_df(client=client, collection_name=q_drant_collection_name)
                df = preprocess_data(df, is_encoded=False, nrows=None)
                pretrain_collaborative(args=arg_parse_collaborative(), df=df)
                pretrain_contentbase(args=arg_parse_contentbase(), df=df)

                logging.info("-" * 50)
            else:
                logging.info("📢 Message rỗng hoặc không có dữ liệu.")

        # Không đóng consumer ở đây vì cần lắng nghe liên tục

    except Exception as e:
        logging.error(f"❌ Lỗi khi kết nối hoặc nhận message: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Command-line arguments: {sys.argv}")
    kafka_consumer(
        topic_name="model_retrain_event",
        bootstrap_servers="kafka.d2f.io.vn:9092"
    )