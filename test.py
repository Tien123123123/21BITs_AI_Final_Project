from minio_server.push import push_object

model_name = "models/model_1.pkl"

minio_bucket_name = "models"
object_name = "test_model.pkl"
push_object(bucket_name=minio_bucket_name, file_path=model_name, object_name=object_name)
