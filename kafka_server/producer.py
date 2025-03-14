import sys, os
from kafka import KafkaProducer
import logging
logging.info("Command-line arguments: " + str(sys.argv))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

def send_message(server="kafka.d2f.io.vn:9092", topic="model_pretrain_to_client_event", message=b"This is a message about pretrain event !"):
    # Khởi tạo Kafka Producer
    producer = KafkaProducer(bootstrap_servers=server)

    # Gửi tin nhắn vào topic 'my_topic'
    topic = topic
    message = message
    producer.send(topic, message)

    # Đảm bảo tất cả các tin nhắn đã được gửi
    producer.flush()

    # Đóng kết nối với Kafka
    producer.close()

    print("Message sent successfully!")

if __name__ == '__main__':
    send_message()
