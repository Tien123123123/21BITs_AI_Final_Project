import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from bayes_opt import BayesianOptimization
from process_data.preprocessing import preprocess_data
from process_data.calc_score import calc_score
from collaborative.train_model import train_model
from process_data.train_test_split import train_test_split
from evaluation_pretrain.evaluate_data import evaluate_model
import pickle
import logging
logging.info("Command-line arguments: " + str(sys.argv))

def evaluate_weights(df, w1, w2, w3, w4):
    df_weighted = calc_score(df, w1, w2, w3, w4)
    df_test, df_weighted_split, df_GT = train_test_split(df, df_weighted)
    model_filename = "model.pkl"
    train_model(df_weighted_split, model_file=model_filename)
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    _, _, mean_f1 = evaluate_model(df_test, df_GT, model, top_N=3)
    logging.info(f"Evaluating weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f} -> F1 Score: {mean_f1:.4f}")
    return mean_f1

def optimize_and_train(df):
    # Define weights
    sets = {'w1': (0.1, 0.9), 'w2': (0.1, 0.9), 'w3': (0.1, 0.9), 'w4': (0.1, 0.9)}

    # Preprocess Data
    df = preprocess_data(df, is_encoded=True, nrows=None)
    # Optimize weights
    optimizer = BayesianOptimization(f=lambda w1, w2, w3, w4: evaluate_weights(df=df, w1=w1, w2=w2, w3=w3, w4=w4), pbounds=sets, random_state=1)
    optimizer.maximize(init_points=1, n_iter=1)

    best_weights = optimizer.max['params']
    best_f1_score = optimizer.max['target']
    model_filename = "model.pkl"

    return best_weights, best_f1_score, model_filename