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
from process_data.optimize_weight import optimize_and_train

def arg_parse_collaborative():
    parser = argparse.ArgumentParser("Collaborative Pretrain Process!")
    parser.add_argument("--save", "-s", type=bool, default="True", help="Save model after training and evaluating", required=False)
    parser.add_argument("--bucket", "-b", type=str, help="minio bucket name",
                        required=False)
    parser.add_argument("--data", "-d", type=str, help="file path and name of dataset", required=False)
    parser.add_argument("--param", "-p", type=str, default=False, help="Parameter for SVD model - {'param 1': [1,2,3,4], param 2': [1,2,3,4]}", required=False)
    parser.add_argument("--model", "-m", type=str, default="collaborative",
                        help="model name", required=False)
    args = parser.parse_args()
    return args

def pretrain_collaborative(args, bucket_name=False, dataset=False):
    # Load and Split data
    bucket_name = args.bucket if args.bucket else bucket_name
    dataset = args.data if args.data else dataset

    # Train model with train data
    params_dict = ast.literal_eval(args.param) if args.param != False else False
    print(params_dict)

    best_model, best_f1_score, best_weights, best_model_filename = optimize_and_train(bucket_name, dataset, trial_nums=5, nrows=500000)
    print("Training Model Complete")

    # Save model
    # timezone = pytz.timezone("Asia/Ho_Chi_Minh")
    # now = datetime.now(timezone)
    # formatted_time = now.strftime("%d_%m_%y_%H_%M")
    # model_name = f"collaborative_{formatted_time}.pkl"
    # save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../models/{model_name}"))
    # with open(save_path, 'wb') as f:
    #     pickle.dump(best_model, f)
    if os.path.exists(best_model_filename):
        print(f"Model was saved at {best_model_filename}")
        bucket_models = "models"
        obj_name = "collaborative.pkl"
        push_object(bucket_name=bucket_models, file_path=best_model_filename, object_name=obj_name)
        # return bucket_models, model_name


if __name__ == '__main__':
    args = arg_parse_collaborative()
    print("Collaborative is running...")
    pretrain_collaborative(args)
