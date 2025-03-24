import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from content_base.hybrid_recommendation import recommendation
from collaborative.session_based_recommend import recommend_products
from collaborative.session_based_recommend import recommend_products_anonymous
import pandas as pd
import os
from datetime import datetime
import pickle
import re
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_coldstart import train_cold_start_clusters
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from arg_parse.arg_parse_coldstart import arg_parse_coldstart
from process_data.preprocessing import preprocess_data
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print("Command-line arguments:", sys.argv)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

# MinIO Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio.d2f.io.vn')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = True
BUCKET_NAME = 'models'

# Default models
DEFAULT_SESSION_MODEL = 'collaborative.pkl'
DEFAULT_CONTENT_MODEL = 'content_base.pkl'
DEFAULT_COLDSTART_MODEL = 'coldstart.pkl'

# Qdrant Configuration
QDRANT_END_POINT = "http://103.155.161.100:6333"
QDRANT_COLLECTION_NAME = "recommendation_system"
MINIO_BUCKET_NAME = "models"

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Threading lock for model updates
model_lock = threading.Lock()

# Global model variables
session_model = None
content_model = None
cold_start = None

def extract_timestamp(filename, prefix):
    pattern = rf"{prefix}_(\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}})\.pkl"
    match = re.search(pattern, filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%d_%m_%y_%H_%M")
        except ValueError:
            return None
    return None

def get_latest_model(bucket_name, prefix, default_model):
    try:
        objects = list(minio_client.list_objects(bucket_name))
        model_files = [obj.object_name for obj in objects if obj.object_name.startswith(prefix)]
        logging.info(f"📂 All models with prefix '{prefix}': {model_files}")
        if not model_files:
            logging.warning(f"⚠️ No models found with prefix '{prefix}' in MinIO.")
            return default_model

        timestamped_models = []
        for file in model_files:
            timestamp = extract_timestamp(file, prefix)
            logging.info(f"🔍 Checking {file}, Extracted Timestamp: {timestamp}")
            if timestamp:
                timestamped_models.append((file, timestamp))
                logging.info(f"✅ Valid timestamp found for {file}: {timestamp}")
            else:
                logging.warning(f"⚠️ No valid timestamp found in file: {file}")

        if timestamped_models:
            latest_model = max(timestamped_models, key=lambda x: x[1])[0]
            logging.info(f"🎯 Selected latest model: {latest_model}")
            return latest_model

        latest_model = sorted(model_files)[-1]
        logging.warning(f"⚠️ No valid timestamps found. Using last lexicographical file: {latest_model}")
        return latest_model
    except Exception as e:
        logging.error(f"❌ Error finding latest model for {prefix}: {str(e)}")
        return default_model

def load_model_from_minio(bucket_name, object_name):
    try:
        response = minio_client.get_object(bucket_name, object_name)
        model_data = response.read()
        response.close()
        response.release_conn()
        model = pickle.load(BytesIO(model_data))
        logging.info(f"Loaded model '{object_name}' from MinIO successfully.")
        logging.info(f"Model '{object_name}' structure: {type(model).__name__}, sample: {str(model)[:200]}")
        return model
    except S3Error as e:
        logging.error(f"Model '{object_name}' not found in MinIO: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error loading model '{object_name}' from MinIO: {str(e)}")
        return None

# Load initial models (on startup only)
with model_lock:
    latest_session_model = get_latest_model(BUCKET_NAME, "collaborative", DEFAULT_SESSION_MODEL)
    latest_content_model = get_latest_model(BUCKET_NAME, "content_base", DEFAULT_CONTENT_MODEL)
    latest_cold_start_model = get_latest_model(BUCKET_NAME, "coldstart", DEFAULT_COLDSTART_MODEL)

    session_model = load_model_from_minio(BUCKET_NAME, latest_session_model)
    content_model = load_model_from_minio(BUCKET_NAME, latest_content_model)
    cold_start = load_model_from_minio(BUCKET_NAME, latest_cold_start_model)

if session_model is None:
    logging.warning("Session model is None.")
if content_model is None:
    logging.warning("Content model is None.")
if cold_start is None:
    logging.warning("Cold-start model is None.")

# Load dataset from Qdrant
client = connect_qdrant(end_point=QDRANT_END_POINT, collection_name=QDRANT_COLLECTION_NAME)
df = load_to_df(client=client, collection_name=QDRANT_COLLECTION_NAME)
df = preprocess_data(df, is_encoded=False, nrows=None)
logging.info(f"Data validated after preprocessing successfully: {len(df)} records")
unique_users= df['user_id'].nunique()
logging.info(f"unique user after preprocessing: {unique_users}")
unique_products= df['product_id'].nunique()
logging.info(f"unique product after preprocessing: {unique_products}")
# Optional: Add a manual refresh endpoint to reload models
@app.route('/refresh_models', methods=['POST'])
def refresh_models():
    try:
        with model_lock:
            global session_model, content_model, cold_start
            latest_session_model = get_latest_model(BUCKET_NAME, "collaborative", DEFAULT_SESSION_MODEL)
            latest_content_model = get_latest_model(BUCKET_NAME, "content_base", DEFAULT_CONTENT_MODEL)
            latest_cold_start_model = get_latest_model(BUCKET_NAME, "coldstart", DEFAULT_COLDSTART_MODEL)

            session_model = load_model_from_minio(BUCKET_NAME, latest_session_model)
            content_model = load_model_from_minio(BUCKET_NAME, latest_content_model)
            cold_start = load_model_from_minio(BUCKET_NAME, latest_cold_start_model)

            if session_model is None:
                logging.warning("Failed to update session model.")
            else:
                logging.info(f"✅ Session model updated successfully. Model file: {latest_session_model}")
            if content_model is None:
                logging.warning("Failed to update content model.")
            else:
                logging.info(f"✅ Content model updated successfully. Model file: {latest_content_model}")
            if cold_start is None:
                logging.warning("Failed to update cold-start model.")
            else:
                logging.info(f"✅ Cold-start model updated successfully. Model file: {latest_cold_start_model}")

        return jsonify({"status": "Models refreshed successfully"}), 200
    except Exception as e:
        logging.error(f"Error refreshing models: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

# Endpoints
@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        if product_id is None:
            return jsonify({"error": "Missing product_id"}), 400

        # Keep product_id as string, no conversion to int
        logging.info(f"Using raw product_id: {product_id}")

        if content_model is None:
            return jsonify({"error": "Content model not loaded"}), 503

        logging.info(f"Using content model: {get_latest_model(BUCKET_NAME, 'content_base', DEFAULT_CONTENT_MODEL)}")
        recommendation_result = recommendation(content_model, product_id, top_k=10)
        logging.info(f"Raw recommendations: {recommendation_result}")

        if isinstance(recommendation_result, dict) and "error" in recommendation_result:
            return jsonify(recommendation_result), 404

        if not isinstance(recommendation_result, dict) or "recommendations" not in recommendation_result:
            return jsonify({"error": "Invalid response format from recommendation function"}), 500

        recommended_product_ids = [item["product_id"] for item in recommendation_result["recommendations"]]
        logging.info(f"Recommendations: {recommended_product_ids}")

        return jsonify({
            'product_id': product_id,
            'recommendations': [{"product_id": prod_id} for prod_id in recommended_product_ids]
        }), 200
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/session-recommend', methods=['POST'])
def session_recommend_api():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        user_id = data.get('user_id')
        product_id = data.get('product_id')
        event_type = data.get('event_type')

        if user_id is None or product_id is None:
            return jsonify({"error": "Missing user_id or product_id"}), 400

        # Keep user_id and product_id as strings
        logging.info(f"user_id: {user_id}, product_id: {product_id}, event_type: {event_type}")

        if session_model is None or cold_start is None:
            return jsonify({"error": "Session or cold-start model not loaded"}), 503

        logging.info(f"Using session model: {get_latest_model(BUCKET_NAME, 'collaborative', DEFAULT_SESSION_MODEL)}")
        logging.info(f"Using cold-start model: {get_latest_model(BUCKET_NAME, 'coldstart', DEFAULT_COLDSTART_MODEL)}")
        recommendations_df = recommend_products(session_model, cold_start, df, user_id, product_id, top_n=10)
        if recommendations_df is None or recommendations_df.empty:
            return jsonify({"error": "No recommendations generated"}), 404

        recommendations = recommendations_df.to_dict(orient="records")
        logging.info(f"Session-based Recommendations: {recommendations}")

        return jsonify({
            "user_id": user_id,
            "product_id": product_id,
            "recommendations": recommendations
        }), 200
    except Exception as e:
        logging.error(f"Error in session-recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/anonymous-recommend', methods=['POST'])
def anonymous_recommend_api():
    try:
        data = request.get_json()
        logging.info(f"Received data for anonymous recommend: {data}")

        product_id = data.get('product_id')
        if product_id is None:
            return jsonify({"error": "Missing product_id"}), 400

        # Keep product_id as string
        logging.info(f"product_id: {product_id}")

        if cold_start is None:
            return jsonify({"error": "Cold-start model not loaded"}), 503

        logging.info(f"Using cold-start model: {get_latest_model(BUCKET_NAME, 'coldstart', DEFAULT_COLDSTART_MODEL)}")
        recommendations_df = recommend_products_anonymous(cold_start, df, product_id)
        if recommendations_df is None or recommendations_df.empty:
            return jsonify({"error": "No recommendations generated"}), 404

        recommendations = recommendations_df.to_dict(orient="records")
        logging.info(f"Anonymous Recommendations: {recommendations}")

        return jsonify({
            "user_id": None,
            "product_id": product_id,
            "recommendations": recommendations
        }), 200
    except Exception as e:
        logging.error(f"Error in anonymous-recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

# Pretrain Endpoints
@app.route('/pretrain_contentbase', methods=['POST'])
def pretrain_contentbase_api():
    try:
        data = request.get_json()
        k = data["k_out"]

        pretrain = pretrain_contentbase(arg_parse_contentbase(), df, minio_bucket_name=MINIO_BUCKET_NAME, k=k)
        global content_model
        content_model = load_model_from_minio(BUCKET_NAME, pretrain[1])
        return jsonify({"result": pretrain})
    except Exception as e:
        logging.error(f"Error in pretrain_contentbase_api: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/pretrain_collaborative', methods=['POST'])
def pretrain_collaborative_api():
    try:

        pretrain = pretrain_collaborative(arg_parse_collaborative(), df, minio_bucket_name=MINIO_BUCKET_NAME)
        global session_model
        session_model = load_model_from_minio(BUCKET_NAME, pretrain[1])
        return jsonify({"result": pretrain})
    except Exception as e:
        logging.error(f"Error in pretrain_collaborative_api: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/pretrain_coldstart', methods=['POST'])
def pretrain_coldstart_api():
    try:
        # Parse JSON payload (for consistency, though not used)
        data = request.get_json()

        # Call the pretraining function
        pretrain = train_cold_start_clusters(
            arg_parse_coldstart(),
            df,
            minio_bucket_name=MINIO_BUCKET_NAME
        )

        # Update the global cold_start model
        global cold_start
        if pretrain:  # Only update if pretrain returned a result (i.e., args.save=True)
            cold_start = load_model_from_minio(BUCKET_NAME, pretrain[1])

        return jsonify({"result": pretrain})
    except Exception as e:
        logging.error(f"Error in pretrain_coldstart_api: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)