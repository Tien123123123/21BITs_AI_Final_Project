import argparse

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