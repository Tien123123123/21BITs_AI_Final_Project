import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from minio import Minio
from minio.error import S3Error
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def load_server():
    minio_server = Minio(
        "minio.d2f.io.vn",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=True
    )
    return minio_server

def load_data(bucket_name, file_name, nrows=None):
    minio_server = load_server()

    bucket_name = bucket_name
    file_name = file_name

    try:
        response = minio_server.get_object(bucket_name, file_name)

        df = pd.read_csv(response, nrows=nrows)

    except S3Error as err:
        print(err)
    finally:
        response.close()
        response.release_conn()

    return df

def create_bucket(bucket_name):
    minio_server = load_server()
    if minio_server.bucket_exists(bucket_name=bucket_name):
        logging.info(f"Bucket {bucket_name} exists !")
    else:
        try:
            minio_server.make_bucket(bucket_name=bucket_name)
            logging.info(f"Create {bucket_name} successfully !")
        except S3Error as err:
            logging.error(err)


if __name__ == '__main__':
    df= load_data("datasets", "chatbot_dataset.csv", nrows=5)
    create_bucket("duy123")