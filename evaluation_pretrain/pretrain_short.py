import os
import pickle
import sys
import ast
from datetime import datetime
import pytz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from process_data.preprocessing import preprocess_data
from process_data.train_test_split import train_test_split
from collaborative.train_model import train_model
from evaluation_pretrain.evaluate_data import evaluate_model
import argparse
from minio_server.push import push_object

def arg_parse():
    parser = argparse.ArgumentParser("Collaborative Pretrain Process!")
    parser.add_argument("--save", "-s", type=bool, default="True", help="Save model after training and evaluating", required=False)
    parser.add_argument("--bucket", "-b", type=str, default="recommendation", help="minio bucket name",
                        required=True)
    parser.add_argument("--data", "-d", type=str, default="dataset.csv", help="file path and name of dataset", required=True)
    parser.add_argument("--param", "-p", type=str, default=False, help="Parameter for SVD model - {'param 1': [1,2,3,4], param 2': [1,2,3,4]}", required=False)
    parser.add_argument("--model", "-m", type=str, default="collaborative",
                        help="model name", required=False)
    args = parser.parse_args()
    return args

def tracking_pretrain(args):
    # Load and Split data
    bucket_name = args.bucket
    file_name = args.data
    df, df_weighted = preprocess_data(bucket_name, file_name, is_encoded=True, nrows=50000)
    df_test, df_weighted, df_GT = train_test_split(df, df_weighted, test_size=0.1)
    # Train model with train data
    params_dict = ast.literal_eval(args.param) if args.param else False
    model, _ = train_model(df_weighted, param_grid=params_dict)

    # Evaluate model with test data
    _, _, eval = evaluate_model(df_test, df_GT, model, top_N=3)
    print(f"eval: {eval}")
    print(f"F1 score: {eval:.4f}")

    # Save model
    if args.save:
        timezone = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(timezone)
        formatted_time = now.strftime("%d_%m_%y_%H_%M")
        model_name = f"{args.model}_{formatted_time}.pkl"
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_name}"))
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        if os.path.exists(save_path):
            print(f"Model was saved at {save_path}")
            push_object(bucket_name=bucket_name, file_path=save_path, object_name=model_name)


if __name__ == '__main__':
    args = arg_parse()
    print("Is running...")
    tracking_pretrain(args)
