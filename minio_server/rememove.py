from minio_server.server import load_server
from minio.error import S3Error

def remove_object(bucket_name, object_name):
    minio_server = load_server()

    bucket_name = bucket_name  # bucket name
    object_name = object_name  # name of data after upload

    try:
        if not minio_server.bucket_exists(bucket_name=bucket_name):
            minio_server.make_bucket(bucket_name=bucket_name)
            print(f"bucket name {bucket_name} was created successfully !")

        minio_server.remove_object(
            bucket_name=bucket_name,
            object_name=object_name,
        )
        print(f"{object_name} was deleted successfully in Minio Server at [ {bucket_name} -> {object_name} ]")

    except S3Error as err:
        print(err)

if __name__ == '__main__':
    bucket_name = "recommendation"
    object_name = "uploaded_hung_2.csv"
    remove_object(bucket_name, object_name)
