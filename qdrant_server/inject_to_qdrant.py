import pandas as pd
from minio import Minio
from minio.error import S3Error
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
import os
from dotenv import load_dotenv
import logging
import time
from io import BytesIO
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get configuration from environment variables
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio.d2f.io.vn')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
BUCKET_NAME = os.getenv('null','dataset')
MINIO_FILE = os.getenv('null','output.csv')

QDRANT_END_POINT = os.getenv('QDRANT_END_POINT', 'http://qdrant:6333')
QDRANT_COLLECTION_NAME = os.getenv('null','test_v3')

# Parse QDRANT_END_POINT
try:
    parsed_qdrant_url = urlparse(QDRANT_END_POINT)
    QDRANT_HOST = parsed_qdrant_url.hostname or 'qdrant'
    QDRANT_PORT = parsed_qdrant_url.port or 6333
except Exception as e:
    logger.error(f"Failed to parse QDRANT_END_POINT: {e}")
    QDRANT_HOST = 'qdrant'
    QDRANT_PORT = 6333

# Initialize MinIO client
try:
    logger.info(f"Connecting to MinIO at {MINIO_ENDPOINT}...")
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True
    )
    logger.info("Successfully connected to MinIO")
except Exception as e:
    logger.error(f"Failed to connect to MinIO: {e}")
    raise

# Initialize FastEmbed with BAAI/bge-small-en-v1.5
try:
    logger.info("Initializing FastEmbed model...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    logger.info("FastEmbed model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FastEmbed model: {e}")
    raise

# Initialize Qdrant client
try:
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=True
    )
    logger.info("Successfully connected to Qdrant")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    raise

def create_collection():
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        if QDRANT_COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection {QDRANT_COLLECTION_NAME}...")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} created successfully")
        else:
            logger.info(f"Collection {QDRANT_COLLECTION_NAME} already exists")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise

def vectorize_text(text: str) -> list:
    """Convert text to vector using FastEmbed and truncate to 128 dimensions"""
    try:
        embeddings = list(embedding_model.embed([text]))
        vector = embeddings[0][:128].tolist()
        if len(vector) != 128:
            logger.error(f"Truncated vector has incorrect dimension: {len(vector)}")
            raise ValueError("Vector dimension error after truncation")
        logger.info(f"Generated vector with dimension: {len(vector)}")
        return vector
    except Exception as e:
        logger.error(f"Error in vectorization: {e}")
        raise

def fetch_csv_from_minio():
    """Fetch chatbotdataset.csv from MinIO and return as DataFrame"""
    try:
        logger.info(f"Fetching {MINIO_FILE} from bucket {BUCKET_NAME}...")
        response = minio_client.get_object(BUCKET_NAME, MINIO_FILE)
        data = response.read()
        df = pd.read_csv(BytesIO(data))
        logger.info(f"Successfully fetched CSV with {len(df)} rows")
        return df
    except S3Error as e:
        logger.error(f"MinIO error fetching CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise
    finally:
        if 'response' in locals():
            response.close()
            response.release_conn()

def process_row(row):
    """Process a single CSV row and store it in Qdrant"""
    try:
        # Create text for embedding
        text_to_embed = f"{row.get('Name', '')} {row.get('Category', '')} {row.get('Type', '')} {row.get('Description', '')}"
        
        # Generate vector
        vector = vectorize_text(text_to_embed)
        
        # Create payload
        payload = {
            "product_id": str(row.get('product_id')),
            "name": row.get('name'),
            "price": float(row.get('price', 0)),
            "category": row.get('category'),
            "type": row.get('type'),
            "brand": row.get('brand'),
            "description": row.get('description')
        }
        
        # Create point
        point = models.PointStruct(
            id=int(time.time() * 1000 + row.name),  # Unique ID based on timestamp and row index
            vector=vector,
            payload=payload
        )
        
        # Upload point to Qdrant
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point]
        )
        
        logger.info(f"Successfully processed row with ID: {row.get('ID')}")
    except Exception as e:
        logger.error(f"Error processing row {row.get('ID')}: {e}")
        raise

def main():
    logger.info("Starting Qdrant data injection...")
    logger.info(f"Configuration:")
    logger.info(f"- MinIO: {MINIO_ENDPOINT} (Bucket: {BUCKET_NAME}, File: {MINIO_FILE})")
    logger.info(f"- Qdrant: {QDRANT_HOST}:{QDRANT_PORT} (Collection: {QDRANT_COLLECTION_NAME})")
    
    # Create Qdrant collection
    create_collection()
    
    # Fetch and process CSV
    try:
        df = fetch_csv_from_minio()
        logger.info(f"Processing {len(df)} rows from CSV...")
        
        for index, row in df.iterrows():
            try:
                process_row(row)
                time.sleep(0.01)  # Small delay to avoid overwhelming Qdrant
            except Exception as e:
                logger.error(f"Skipping row {index} due to error: {e}")
                continue
        
        logger.info("Data injection completed successfully")
    except Exception as e:
        logger.error(f"Failed to process CSV data: {e}")
        raise

if __name__ == "__main__":
    main()