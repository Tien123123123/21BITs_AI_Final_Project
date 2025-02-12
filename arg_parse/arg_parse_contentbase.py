import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

def arg_parse_contentbase():
    parser = argparse.ArgumentParser("Collaborative Pretrain Process!")
    parser.add_argument("--save", "-s", type=bool, default="True", help="Save model after training and evaluating", required=False)
    parser.add_argument("--bucket", "-b", type=str, default="recommendation", help="minio bucket name",
                        required=False)
    parser.add_argument("--data", "-d", type=str, default="dataset.csv", help="file path and name of dataset on bucket", required=False)
    parser.add_argument("--model", "-m", type=str, default="contentbase", help="Name of model after saving",
                        required=False)
    parser.add_argument("-k", type=int, default=10, help="Total recommended items for a input item",
                        required=False)
    args = parser.parse_args(args=[])
    return args