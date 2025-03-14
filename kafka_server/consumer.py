import sys, os
from kafka import KafkaConsumer, TopicPartition
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
import logging
from qdrant_server.load_data import load_to_df
from process_data.preprocessing import preprocess_data
from qdrant_server.server import connect_qdrant
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def kafka_consumer(topic_name, bootstrap_servers='kafka.d2f.io.vn:9092'):
    # Khởi tạo logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Command-line arguments: {sys.argv}")

    try:
        consumer = KafkaConsumer(
            topic_name,  # Subscribe topic trực tiếp như code cũ
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',  # Bắt đầu từ cuối nếu không có offset commit
            enable_auto_commit=False,  # Commit thủ công để kiểm soát
            group_id='my-consumer-group-' + str(datetime.utcnow().timestamp()),  # Group ID duy nhất mỗi lần chạy
            value_deserializer=lambda v: v.decode('utf-8') if v else None
        )

        logging.info(f"📡 Đang lắng nghe topic: {topic_name} để đợi message mới...")

        # Lấy offset cuối cùng khi khởi động
        consumer.poll(timeout_ms=1000)  # Poll đầu tiên để lấy metadata
        initial_offsets = {TopicPartition(topic_name, p): consumer.position(TopicPartition(topic_name, p))
                          for p in consumer.partitions_for_topic(topic_name)}
        for tp, offset in initial_offsets.items():
            logging.info(f"Partition {tp.partition} - Offset cuối cùng khi khởi động: {offset}")

        # Lắng nghe message mới
        for message in consumer:
            if message.value is not None:
                current_offset = message.offset
                tp = TopicPartition(message.topic, message.partition)

                # Chỉ xử lý nếu offset lớn hơn offset khởi đầu
                if current_offset >= initial_offsets[tp]:
                    logging.info(f"📢 Nhận được message mới: {message.value}")
                    logging.info(f"  - Topic: {message.topic}")
                    logging.info(f"  - Partition: {message.partition}")
                    logging.info(f"  - Offset: {message.offset}")
                    logging.info(f"  - Key: {message.key}")
                    logging.info(f"  - Status: Ready to Pretrain!")

                    # Kiểm tra thời gian message (tùy chọn)
                    try:
                        msg_data = json.loads(message.value)
                        event_time = msg_data.get('event_time')
                        if event_time:
                            event_dt = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S UTC")
                            now = datetime.utcnow()
                            if (now - event_dt).total_seconds() > 3600:  # Bỏ qua message cũ hơn 1 giờ
                                logging.info(f"Message tại offset {current_offset} quá cũ ({event_time}), bỏ qua.")
                                consumer.commit()
                                continue
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.warning(f"Không thể parse event_time từ message: {e}")

                    # Kết nối Qdrant và xử lý dữ liệu
                    q_drant_end_point = "http://103.155.161.100:6333"
                    q_drant_collection_name = "recommendation_system"
                    client = connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)
                    df = load_to_df(client=client, collection_name=q_drant_collection_name)
                    df = preprocess_data(df, is_encoded=False, nrows=None)
                    pretrain_collaborative(args=arg_parse_collaborative(), df=df)
                    pretrain_contentbase(args=arg_parse_contentbase(), df=df)

                    logging.info("-" * 50)
                    consumer.commit()  # Commit offset sau khi xử lý
                    initial_offsets[tp] = current_offset + 1  # Cập nhật offset khởi đầu
                else:
                    logging.debug(f"Offset {current_offset} không phải message mới, bỏ qua.")
                    consumer.commit()
            else:
                logging.info("📢 Message rỗng hoặc không có dữ liệu.")
                consumer.commit()

    except Exception as e:
        logging.error(f"❌ Lỗi khi kết nối hoặc nhận message: {e}")

if __name__ == "__main__":
    kafka_consumer(
        topic_name="model_retrain_event",
        bootstrap_servers="kafka.d2f.io.vn:9092"
    )