import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import pandas as pd
from process_data.preprocessing import preprocess_data
from collaborative.train_model import train_model
from surprise import prediction_algorithms as alg
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="Command Console")
    parser.add_argument("--model", type=str, help=f"Name of model ( svd, knnbasic, knnwithmeans, knnbaseline, normalpredictor, baselineonly, nmf, slopeone, coclustering", required=True)
    parser.add_argument("--data", "-d", type=str, default="data/one_data.csv", help="Load dataset", required=True)
    args = parser.parse_args()
    return args

def evaluate_model(args):
    model_mapping = {
        "svd": alg.SVD(),
        "knnbasic": alg.KNNBasic(),
        "knnwithmeans": alg.KNNWithMeans(),
        "knnbaseline": alg.KNNBaseline(),
        "normalpredictor": alg.NormalPredictor(),
        "baselineonly": alg.BaselineOnly(),
        "nmf": alg.NMF(),
        "slopeone": alg.SlopeOne(),
        "coclustering": alg.CoClustering(),
    }
    if args.model.lower() not in model_mapping:
        raise ValueError(f"Invalid name, name must exist in {[name for name, _ in model_mapping.items()]}")
    model = model_mapping[args.model.lower()]
    model = model

    # Load data
    root = args.data
    df = pd.read_csv(root, nrows=1000)

    # Data Preprocessing
    _, df_weighted = preprocess_data(df)
    model, results = train_model(df_weighted, model=model)
    # Training Model
    # while True:
    #     inp = input("Choose Command [Train/Save/Exit]: ").lower()
    #     if inp == "save":
    #         save = input("Model name for saving: ").lower()
    #         model_file = f"models/{save}.pkl"
    #         with open(model_file, "wb") as f:
    #             pickle.dump(model, f)
    #         print(f'Model save successfully at {model_file}')
    #         break
    #     elif inp == "train":
    #         model, results = train_model(df_weighted, model=model)
    #     elif inp == "exit":
    #         break

if __name__ == '__main__':
    args = arg_parse()
    evaluate_model(args)
