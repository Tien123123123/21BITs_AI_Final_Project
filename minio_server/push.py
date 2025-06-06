from minio_server.server import load_server
from minio.error import S3Error
from datetime import datetime
import pytz
import os
import logging
logging.basicConfig(level=logging.INFO)

def push_object(bucket_name, file_path, object_name):
    minio_server = load_server()

    bucket_name = bucket_name  # bucket name
    file_path = file_path # file path to upload data
    object_name = object_name  # name of data after upload

    try:
        if not minio_server.bucket_exists(bucket_name=bucket_name):
            minio_server.make_bucket(bucket_name=bucket_name)
            logging.info(f"bucket name {bucket_name} was created successfully !")

        minio_server.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path
        )
        logging.info(f"{file_path} was uploaded successfully in Minio Server at [ {bucket_name} -> {object_name} ]")

    except S3Error as err:
        logging.info(err)

if __name__ == '__main__':
    timezone = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(timezone)
    formatted_time = now.strftime("%d_%m_%y_%H:%M")

    bucket_name = "recommendation"
    file_path = "models/test_model.pkl"
    object_name = f"test_model_{formatted_time}.pkl"
    push_object(bucket_name, file_path, object_name)
