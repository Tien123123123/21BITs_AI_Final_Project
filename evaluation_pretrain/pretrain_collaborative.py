import os, sys, pickle, ast, pytz
from datetime import datetime
from minio_server.server import create_bucket
from qdrant_server.load_data import load_to_df
from qdrant_server.server import connect_qdrant
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from minio_server.push import push_object
from arg_parse.arg_parse_collaborative import arg_parse_collaborative
from process_data.optimize_weight import optimize_and_train
import logging
logging.basicConfig(level=logging.INFO)

def pretrain_collaborative(args, q_drant_end_point="http://103.155.161.100:6333", q_drant_collection_name="recommendation_system", minio_bucket_name = "models"):
    # Load and Split data
    end_point = q_drant_end_point
    client = connect_qdrant(end_point=end_point, collection_name=q_drant_collection_name)
    df = load_to_df(client=client, collection_name=q_drant_collection_name)

    # Train model with train data
    params_dict = ast.literal_eval(args.param) if args.param != False else False
    logging.info(params_dict)
    best_weights, best_f1_score, model_filename = optimize_and_train(df)
    logging.info("Training Model Complete")

    # Save model
    timezone = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(timezone)
    formatted_time = now.strftime("%d_%m_%y_%H_%M")
    model_name = f"collaborative_{formatted_time}.pkl"
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_name}"))
    with open(save_path, 'wb') as f:
        pickle.dump(model_filename, f)
    if os.path.exists(model_filename):
        logging.info(f"Model was saved at {model_filename}")
        bucket_models = minio_bucket_name
        create_bucket(bucket_models)
        push_object(bucket_name=bucket_models, file_path=model_filename, object_name=model_name)
        return bucket_models, model_name


if __name__ == '__main__':
    args = arg_parse_collaborative()
    logging.info("Collaborative is running...")
    q_drant_end_point = "http://103.155.161.100:6333"
    q_drant_collection_name = "recommendation_system"
    pretrain_collaborative(args, q_drant_end_point, q_drant_collection_name, minio_bucket_name = "models")