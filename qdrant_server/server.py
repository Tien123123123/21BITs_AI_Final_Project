<<<<<<< HEAD
﻿import sys, os
=======
import sys, os
>>>>>>> 770a7c253170202e338aadc3c408d3854456e8e1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from qdrant_client import QdrantClient, models
import logging
logging.basicConfig(level=logging.INFO)


def connect_qdrant(end_point, collection_name="midjourney"):
    client = QdrantClient(url=end_point, timeout=60)
    collection_name = collection_name

    status = client.collection_exists(collection_name=collection_name)

    if status == True:
        logging.info(f"Connect Complete")
        return client
    else:
        logging.info("Fail Connection")

if __name__ == '__main__':
    # Key connection
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "product_embeddings"

    connect_qdrant(end_point=q_drant_end_point, collection_name=q_drant_collection_name)