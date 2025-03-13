# arg_parse/arg_parse_coldstart.py
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))

def arg_parse_coldstart():
    parser = argparse.ArgumentParser("Cold-start Pretrain Process!")
    parser.add_argument("--save", "-s", type=bool, default=True, help="Save model after training", required=False)
    parser.add_argument("--bucket", "-b", type=str, default="datasets", help="MinIO bucket name", required=False)
    parser.add_argument("--data", "-d", type=str, default="dataset.csv", help="File path and name of dataset on bucket", required=False)
    parser.add_argument("--model", "-m", type=str, default="coldstart", help="Name of model after saving", required=False)
    parser.add_argument("--top_n", "-t", type=int, default=8, help="Number of top candidates for clusters", required=False)
    parser.add_argument("--random_n", "-r", type=int, default=5, help="Number of random candidates for clusters", required=False)
    args = parser.parse_args(args=[])  # Use args=[] for programmatic default usage
    return args

if __name__ == "__main__":
    args = arg_parse_coldstart()
    print(args)