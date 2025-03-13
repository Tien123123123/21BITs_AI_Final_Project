from minio_server.server import load_server
from minio.error import S3Error
import logging
logging.basicConfig(level=logging.INFO)

def remove_object(bucket_name, object_name):
    minio_server = load_server()

    bucket_name = bucket_name  # bucket name
    object_name = object_name  # name of data after upload

    try:
        minio_server.remove_object(
            bucket_name=bucket_name,
            object_name=object_name,
        )
        logging.info(f"{object_name} was deleted successfully in Minio Server at [ {bucket_name} -> {object_name} ]")

    except S3Error as err:
        logging.info(err)

if __name__ == '__main__':
    bucket_name = "recommendation"
    object_name = "uploaded_hung_2.csv"
    remove_object(bucket_name, object_name)