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
import requests  # Thêm thư viện requests để gọi API

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def kafka_consumer(topic_name, bootstrap_servers='kafka.d2f.io.vn:9092'):
    # Khởi tạo logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Command-line arguments: {sys.argv}")

    # Kết nối Qdrant một lần duy nhất
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"
    try:
        client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
        logging.info(f"✅ Đã kết nối đến Qdrant tại {q_drant_end_point}")
    except Exception as e:
        logging.error(f"❌ Lỗi khi kết nối Qdrant: {e}")
        return

    # Khởi tạo Kafka consumer
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='my-consumer-group',
            value_deserializer=lambda v: v.decode('utf-8') if v else None
        )
        logging.info(f"📡 Đang lắng nghe topic: {topic_name} để đợi message mới nhất...")
    except Exception as e:
        logging.error(f"❌ Lỗi khi khởi tạo Kafka consumer: {e}")
        return

    # URL của Flask API (thay đổi nếu Flask chạy trên host/port khác)
    flask_api_url = "http://localhost:5000/refresh_models"  # Điều chỉnh URL nếu cần

    # Lắng nghe message mới liên tục
    while True:
        try:
            for message in consumer:
                if message.value is not None:
                    logging.info(f"📢 Nhận được message mới nhất: {message.value}")
                    logging.info(f"  - Topic: {message.topic}")
                    logging.info(f"  - Partition: {message.partition}")
                    logging.info(f"  - Offset: {message.offset}")
                    logging.info(f"  - Key: {message.key}")
                    logging.info(f"  - Status: Processing...")

                    # Xử lý dữ liệu từ Qdrant
                    try:
                        df = load_to_df(client=client, collection_name=q_drant_collection_name)
                        df = preprocess_data(df, is_encoded=False, nrows=None)
                        logging.info("✅ Dữ liệu đã được tải và tiền xử lý thành công")
                    except Exception as e:
                        logging.error(f"❌ Lỗi khi tải hoặc tiền xử lý dữ liệu: {e}")
                        continue

                    # Chạy pretrain collaborative
                    try:
                        pretrain_collaborative(args=arg_parse_collaborative(), df=df)
                        logging.info("✅ Đã chạy pretrain_collaborative thành công")
                    except Exception as e:
                        logging.error(f"❌ Lỗi khi chạy pretrain_collaborative: {e}")

                    # Chạy pretrain contentbase
                    try:
                        pretrain_contentbase(args=arg_parse_contentbase(), df=df)
                        logging.info("✅ Đã chạy pretrain_contentbase thành công")
                    except Exception as e:
                        logging.error(f"❌ Lỗi khi chạy pretrain_contentbase: {e}")

                    # Gọi API /refresh_models sau khi pretrain xong
                    try:
                        response = requests.post(flask_api_url)
                        if response.status_code == 200:
                            logging.info(f"✅ Đã gọi API /refresh_models thành công: {response.json()}")
                        else:
                            logging.error(f"❌ Lỗi khi gọi API /refresh_models: {response.status_code} - {response.text}")
                    except Exception as e:
                        logging.error(f"❌ Lỗi khi gửi yêu cầu đến API /refresh_models: {e}")

                    logging.info("-" * 50)
                else:
                    logging.info("📢 Message rỗng hoặc không có dữ liệu.")

        except Exception as e:
            logging.error(f"❌ Lỗi trong vòng lặp consumer: {e}")
            continue

if __name__ == "__main__":
    kafka_consumer(
        topic_name="model_retrain_event",
        bootstrap_servers="kafka.d2f.io.vn:9092"
    )