from kafka import KafkaAdminClient
from kafka.admin import NewTopic


admin_client = KafkaAdminClient(bootstrap_servers="kafka_server.d2f.io.vn:9092", client_id='test-client')

# Create topic
topic = "model_pretrain_to_client_event"
topic = NewTopic(name=topic, num_partitions=1, replication_factor=1)

admin_client.create_topics(new_topics=[topic], validate_only=False)

print(f"Create topic {topic} successfully !")
