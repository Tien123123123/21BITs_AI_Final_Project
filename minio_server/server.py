from minio import Minio
from minio.error import S3Error
import pandas as pd

def load_server():
    minio_server = Minio(
        "103.155.161.94:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
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

