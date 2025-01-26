import os.path
import pickle
from process_data.preprocessing import preprocess_data
from process_data.train_test_split import train_test_split
from collaborative.train_model import train_model
from evaluation_pretrain.evaluate_data import evaluate_model
import argparse

def arg_parse():
    parser = argparse.ArgumentParser("Collaborative Pretrain Process!")
    parser.add_argument("--save", "-s", type=bool, default="True", help="Save model after training and evaluating", required=False)
    parser.add_argument("--data", "-d", type=str, default="data/[data name].csv", help="file path and name of dataset", required=True)
    parser.add_argument("--param", "-p", type=str, default="{'param 1': [1,2,3,4], param 2': [1,2,3,4]}", help="Parameter for SVD model", required=False)
    args = parser.parse_args()
    return args

def tracking_pretrain(args):
    # Load and Split data
    root = args.data
    df, df_weighted = preprocess_data(root, is_encoded=True)
    df_test, df_weighted, df_GT = train_test_split(df, df_weighted, test_size=0.1)
    # Train model with train data
    model, _ = train_model(df_weighted, param_grid=args.param if args.param else False)

    # Evaluate model with test data
    eval = evaluate_model(df_test, model, top_N=3)
    print(f"F1 score: {eval:.4f}")

    # Save model
    if args.save:
        save_path = "models/collaborative.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        if os.path.exists(save_path): print(f"Model save successfully at {save_path}")


if __name__ == '__main__':
    args = arg_parse()
    tracking_pretrain(args)
