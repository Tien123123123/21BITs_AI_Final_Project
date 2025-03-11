from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from content_base.hybrid_recommendation import recommendation
from collaborative.session_based_recommend import recommend_products
from collaborative.session_based_recommend import recommend_products_anonymous
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import re
from io import BytesIO
from minio import Minio
from evaluation_pretrain.pretrain_collaborative import pretrain_collaborative
from evaluation_pretrain.pretrain_contentbase import pretrain_contentbase
from evaluation_pretrain.pretrain_coldstart import train_cold_start_clusters
from arg_parse.arg_parse_contentbase import arg_parse_contentbase
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from arg_parse.arg_parse_coldstart import arg_parse_coldstart
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
print("Command-line arguments:", sys.argv)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

# =========================
# MinIO Configuration
# =========================
# MinIO Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', '103.155.161.94:9000')  # Replace with your endpoint
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_SECURE = False  # Set to True if your MinIO uses HTTPS

BUCKET_NAME = 'models'  # Replace with your bucket name

# Default models if no latest timestamped model is found
DEFAULT_SESSION_MODEL = 'model.pkl'
DEFAULT_CONTENT_MODEL = 'content_base.pkl'
DEFAULT_COLDSTART_MODEL = 'cold_start.pkl'

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)
import re
import logging


def extract_timestamp(filename, prefix):
    """
    Extracts the timestamp from the filename using regex.
    Expected format: prefix_DD_MM_YY_HH_MM.pkl
    """
    pattern = rf"{prefix}_(\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}})\.pkl"
    match = re.search(pattern, filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%d_%m_%y_%H_%M")
        except ValueError:
            return None
    return None

def get_latest_model(bucket_name, prefix, default_model):
    """
    Get the latest model based on timestamp sorting.
    """
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

        # If no valid timestamps were found, fallback to lexicographical sorting
        latest_model = sorted(model_files)[-1]
        logging.warning(f"⚠️ No valid timestamps found. Using last lexicographical file: {latest_model}")
        return latest_model

    except Exception as e:
        logging.error(f"❌ Error finding latest model for {prefix}: {str(e)}")
        return default_model

def load_model_from_minio(bucket_name, object_name):
    """
    Retrieve and load a pickle model from MinIO.
    """
    try:
        response = minio_client.get_object(bucket_name, object_name)
        model_data = response.read()
        response.close()
        response.release_conn()

        model = pickle.load(BytesIO(model_data))
        logging.info(f"Loaded model '{object_name}' from MinIO successfully.")
        return model
    except S3Error as e:
        logging.error(f"Model '{object_name}' not found. Using default model.")
        return None
    except Exception as e:
        logging.error(f"Error loading model '{object_name}' from MinIO: {str(e)}")
        raise

# Determine the latest models
latest_session_model = get_latest_model(BUCKET_NAME, "model", DEFAULT_SESSION_MODEL)
latest_content_model = get_latest_model(BUCKET_NAME, "content_base", DEFAULT_CONTENT_MODEL)
lastest_cold_start_model=get_latest_model(BUCKET_NAME,"cold_start", DEFAULT_COLDSTART_MODEL)
# Load the models

try:
    session_model = load_model_from_minio(BUCKET_NAME, latest_session_model)
    content_model = load_model_from_minio(BUCKET_NAME, latest_content_model)
    cold_start = load_model_from_minio(BUCKET_NAME,lastest_cold_start_model)
except Exception as e:
    logging.error("Failed to load models from MinIO. Exiting application.")
    raise
# =========================
# Load Dataset from MinIO
# =========================

def load_dataset_from_minio(bucket_name, object_name):
    """
    Retrieve and load a CSV dataset from a MinIO bucket.
    """
    try:
        # Get the object from MinIO
        response = minio_client.get_object(bucket_name, object_name)
        # Read the entire object data into memory
        dataset_data = response.read()
        response.close()
        response.release_conn()
        # Convert the byte data into a pandas DataFrame
        dataset_df = pd.read_csv(BytesIO(dataset_data))
        logging.info("loading dataset from minio")
        logging.info(f"Loaded dataset '{object_name}' from MinIO successfully.")
        return dataset_df
    except Exception as e:
        logging.error(f"Error loading dataset '{object_name}' from MinIO: {str(e)}")
        raise

# Specify the correct bucket name for the dataset
dataset_bucket_name = 'datasets'  # Correct bucket name
csv_file_path = "dataset.csv"  # Path to the dataset in the 'datasets' bucket

# Load dataset from MinIO
df = load_dataset_from_minio(dataset_bucket_name, csv_file_path)

# =========================
# LabelEncoders Setup
# =========================
enc_product_id = LabelEncoder()
enc_user_id = LabelEncoder()
enc_user_session = LabelEncoder()

# Fit and transform each column independently
df["product_id"] = enc_product_id.fit_transform(df["product_id"])
df["user_id"] = enc_user_id.fit_transform(df["user_id"])
df["user_session"] = enc_user_session.fit_transform(df["user_session"])

def update_label_encoder(encoder, value):
    """Dynamically update LabelEncoder with new values."""
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

def update_label_encoder_if_exists(encoder, value):
    """Encode a value only if it exists in the LabelEncoder; otherwise return None."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return None

# =========================
# Endpoints
# =========================

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

        try:
            product_id = int(product_id)
        except ValueError:
            return jsonify({"error": "product_id must be an integer"}), 400

        # Dynamically update the product_id encoder if necessary
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"Encoded product_id: {encoded_product_id}")
        except Exception as e:
            logging.error(f"Error during encoding: {str(e)}")
            return jsonify({"error": "Invalid product_id after encoding"}), 400

        # Call the recommendation function using the content-based model loaded from MinIO.
        # (Ensure your recommendation() function is updated to accept a model object.)
        recommendation_result = recommendation(content_model, encoded_product_id)
        logging.info(f"Raw recommendations before decoding: {recommendation_result}")

        if not isinstance(recommendation_result, dict) or "recommendations" not in recommendation_result:
            return jsonify({"error": "Invalid response format from recommendation function"}), 500

        # Extract recommended product IDs and decode them
        recommended_product_ids = [item["product_id"] for item in recommendation_result["recommendations"]]
        decoded_recommendations = enc_product_id.inverse_transform(np.array(recommended_product_ids))
        logging.info(f"Decoded recommendations: {decoded_recommendations}")

        return jsonify({
            'product_id': product_id,
            'recommendations': [{"product_id": int(prod_id)} for prod_id in decoded_recommendations]
        }), 200

    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/session-recommend', methods=['POST'])
def session_recommend_api():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        # Extract user and product IDs along with the event type
        user_id = data.get('user_id')
        product_id = data.get('product_id')
        event_type = data.get('event_type')

        if user_id is None or product_id is None:
            return jsonify({"error": "Missing user_id or product_id"}), 400

        try:
            user_id = int(user_id)
            product_id = int(product_id)
        except ValueError:
            return jsonify({"error": "user_id and product_id must be integers"}), 400

        # Encode user_id: if not already in the encoder, update it dynamically
        if user_id in enc_user_id.classes_:
            encoded_user_id = update_label_encoder(enc_user_id, user_id)
            logging.info(f"Encoded user_id: {encoded_user_id}")
        else:
            encoded_user_id = user_id
            logging.info(f"User ID not in encoder, using directly: {encoded_user_id}")

        # Encode product_id dynamically
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"user_id: {encoded_user_id}, product_id: {encoded_product_id}, event_type: {event_type}")
        except Exception as e:
            logging.error(f"Error during encoding: {str(e)}")
            return jsonify({"error": "Invalid user_id or product_id after encoding"}), 400

        # Call the session-based recommendation function using the session model loaded from MinIO.
        # (Ensure your recommend_products() function is updated to accept a model object.)
        recommendations_df = recommend_products(session_model, cold_start, df, encoded_user_id, encoded_product_id, top_n=10)
        logging.info(f"Recommendations DataFrame columns: {recommendations_df.columns}")

        # Decode the product IDs in the recommendations
        recommendations_df['product_id'] = recommendations_df['product_id'].astype(int)
        recommendations_df['product_id'] = enc_product_id.inverse_transform(recommendations_df['product_id'])

        # Convert the DataFrame to a JSON-serializable list of dictionaries
        if isinstance(recommendations_df, pd.DataFrame):
            recommendations = recommendations_df.to_dict(orient="records")
        else:
            recommendations = recommendations_df

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
    """
    Recommend products for anonymous (guest) users based on product_id (cold start).
    """
    try:
        data = request.get_json()
        logging.info(f"Received data for anonymous recommend: {data}")

        product_id = data.get('product_id')

        if product_id is None:
            return jsonify({"error": "Missing product_id"}), 400

        try:
            product_id = int(product_id)
        except ValueError:
            return jsonify({"error": "product_id must be an integer"}), 400

        # Encode product_id dynamically (same as session recommend)
        try:
            encoded_product_id = update_label_encoder(enc_product_id, product_id)
            logging.info(f"Encoded product_id: {encoded_product_id}")
        except Exception as e:
            logging.error(f"Error during product_id encoding: {str(e)}")
            return jsonify({"error": "Invalid product_id after encoding"}), 400

        # Generate recommendations using the cold-start cluster-based logic
        recommendations_df = recommend_products_anonymous(cold_start, df, encoded_product_id)

        # Decode product_id back to original before returning
        recommendations_df['product_id'] = recommendations_df['product_id'].astype(int)
        recommendations_df['product_id'] = enc_product_id.inverse_transform(recommendations_df['product_id'])

        # Convert DataFrame to a JSON-serializable list
        recommendations = recommendations_df.to_dict(orient="records")

        return jsonify({
            "user_id": None,  # Since it's anonymous
            "product_id": product_id,
            "recommendations": recommendations
        }), 200

    except Exception as e:
        logging.error(f"Error in anonymous-recommend endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500



# Pretrain Process
# @app.route('/pretrain_contentbase', methods=['POST'])
# def pretrain_contentbase_api():
#     try:
#         data = request.get_json()
#
#         bucket_name = data["bucket_name"]
#         dataset = data["dataset"]
#         k = data["k_out"]
#
#         print(f"Obtain data successfully !")
#         print(f"bucket_name: {bucket_name}")
#         print(f"dataset: {dataset}")
#
#         pretrain = pretrain_contentbase(arg_parse_contentbase(), bucket_name=bucket_name, dataset=dataset, k=k)
#         return jsonify({
#             "result": pretrain
#         })
#     except Exception as e:
#         return jsonify({
#             "error": str(e)
#         }), 500
#
# @app.route('/pretrain_collaborative', methods=['POST'])
# def pretrain_collaborative_api():
#     try:
#         data = request.get_json()
#
#         bucket_name = data["bucket_name"]
#         dataset = data["dataset"]
#
#         print(f"Obtain data successfully !")
#         print(f"bucket_name: {bucket_name}")
#         print(f"dataset: {dataset}")
#         pretrain = pretrain_collaborative(arg_parse_collaborative(), bucket_name=bucket_name, dataset=dataset)
#         return jsonify({
#             "result": pretrain
#         })
#     except Exception as e:
#         return jsonify({
#             "error": str(e)
#         }), 500

@app.route('/pretrain_coldstart', methods=['POST'])
def pretrain_coldstart_api():
    """
    Endpoint to pretrain cold-start recommendation clusters.
    Expects JSON with bucket_name and dataset path.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        bucket_name = data.get("bucket_name")
        dataset_path = data.get("dataset")

        if not bucket_name or not dataset_path:
            return jsonify({"error": "Missing bucket_name or dataset"}), 400

        logging.info(f"Starting cold-start pretraining with bucket: {bucket_name}, dataset: {dataset_path}")

        # Load dataset from MinIO
        dataset_df = load_dataset_from_minio(bucket_name, dataset_path)

        # Ensure required columns exist
        if "product_id" not in dataset_df.columns or "user_session" not in dataset_df.columns:
            return jsonify({"error": "Dataset must contain product_id and user_session columns"}), 400

        # Apply label encoding if not already encoded
        if not pd.api.types.is_numeric_dtype(dataset_df["product_id"]):
            dataset_df["product_id"] = enc_product_id.fit_transform(dataset_df["product_id"])
        if not pd.api.types.is_numeric_dtype(dataset_df["user_session"]):
            dataset_df["user_session"] = enc_user_session.fit_transform(dataset_df["user_session"])

        # Parse arguments and override defaults with request data
        args = arg_parse_coldstart()
        args.bucket = bucket_name
        args.data = dataset_path  # Not used directly since df is passed, but kept for consistency
        # Optionally, allow these to be overridden via JSON if desired
        args.save = data.get("save", args.save)
        args.model = data.get("model", args.model)
        args.top_n = data.get("top_n", args.top_n)
        args.random_n = data.get("random_n", args.random_n)

        # Train and save the cold-start model to MinIO
        model_path = train_cold_start_clusters(
            args,
            df=dataset_df,
            bucket_name=BUCKET_NAME  # Use the 'models' bucket
        )

        # Refresh the cold_start model in memory if saved
        if model_path:
            global cold_start
            latest_cold_start_model = get_latest_model(BUCKET_NAME, "cold_start", DEFAULT_COLDSTART_MODEL)
            cold_start = load_model_from_minio(BUCKET_NAME, latest_cold_start_model)
        else:
            latest_cold_start_model = None  # No model saved

        return jsonify({
            "status": "success",
            "message": "Cold-start clusters pretrained successfully" + (" (not saved)" if not model_path else ""),
            "model_file": latest_cold_start_model if model_path else None
        }), 200

    except S3Error as e:
        logging.error(f"MinIO error during cold-start pretraining: {str(e)}")
        return jsonify({"error": "MinIO storage error", "details": str(e)}), 500
    except Exception as e:
        logging.error(f"Error in pretrain_coldstart_api: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)