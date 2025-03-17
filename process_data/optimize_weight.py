import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from bayes_opt import BayesianOptimization
from process_data.calc_score import calc_score
from collaborative.train_model import train_model
from process_data.train_test_split import train_test_split
from evaluation_pretrain.evaluate_data import evaluate_model
import pickle
import logging

logging.info("Command-line arguments: " + str(sys.argv))

def evaluate_weights(df, w3, r2, r4, r1):
    df = df
    # Ensure weight order: w1 < w4 < w2 < w3
    w2 = w3 * r2
    w4 = w2 * r4
    w1 = w4 * r1

    if not (w1 < w4 < w2 < w3):
        logging.warning(f"Invalid weights order: w1={w1}, w4={w4}, w2={w2}, w3={w3}")
        return -1.0

    df_weighted = calc_score(df, w1, w2, w3, w4)
    df_test, df_weighted_split, df_GT = train_test_split(df, df_weighted)
    model_filename = "model.pkl"
    train_model(df_weighted_split, model_file=model_filename)

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    _, f1_score = evaluate_model(df_test, df_GT, model, top_N=3)
    logging.info(f"Evaluating weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f} -> F1 Score: {f1_score:.4f}")
    return f1_score


def optimize_and_train(df):
    # Define bounded ratios w1 < w4 < w2 < w3
    df= df
    pbounds = {
        'w3': (0.6, 0.9),  # w3 is the largest value
        'r2': (0.2, 1.0),  # ratio w2/w3, ensures w2 < w3
        'r4': (0.1, 1.0),  # ratio w4/w2, ensures w4 < w2
        'r1': (0.1, 1.0)   # ratio w1/w4, ensures w1 < w4
    }


    optimizer = BayesianOptimization(f=lambda w3, r2, r4, r1: evaluate_weights(df=df, w3=w3, r2=r2, r4=r4, r1=r1), pbounds=pbounds, random_state=5)
    optimizer.maximize(init_points=1, n_iter=1)  # 10 random + 20 optimized trials

    best_params = optimizer.max['params']
    best_f1_score = optimizer.max['target']

    w3_opt = best_params['w3']
    w2_opt = w3_opt * best_params['r2']
    w4_opt = w2_opt * best_params['r4']
    w1_opt = w4_opt * best_params['r1']

    best_weights = {
        'w1': w1_opt,
        'w2': w2_opt,
        'w3': w3_opt,
        'w4': w4_opt
    }

    model_filename = "model.pkl"

    return best_weights, best_f1_score, model_filename