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

def optimize_and_train(df):
    # Preprocess Data
    df = preprocess_data(df, is_encoded=True, nrows=None)

    # Define evaluate_weights as a closure to capture df
    def evaluate_weights(w3, r2, r4, r1):
        # Convert ratios to actual values ensuring w1 < w4 < w2 < w3
        w2 = w3 * r2  # ensures w2 < w3 since r2 < 1
        w4 = w2 * r4  # ensures w4 < w2 since r4 < 1
        w1 = w4 * r1  # ensures w1 < w4 since r1 < 1

        # Safety check
        if not (w1 < w4 < w2 < w3):
            logging.warning("Invalid weight order encountered.")
            return -1.0  # Penalize if the order is incorrect (rare case)

        df_weighted = calc_score(df, w1, w2, w3, w4)
        df_test, df_weighted_split, df_GT = train_test_split(df, df_weighted)
        model_filename = "model.pkl"
        train_model(df_weighted_split, model_file=model_filename)
        with open(model_filename, "rb") as f:
            model = pickle.load(f)
        _, _, mean_f1 = evaluate_model(df_test, df_GT, model, top_N=3)
        logging.info(f"Evaluating weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}, w4={w4:.4f} -> F1 Score: {mean_f1:.4f}")
        return mean_f1

    # Define pbounds with the ratio
    pbounds = {
        'w3': (0.6, 0.9),  # w3 is largest
        'r2': (0.2, 1.0),  # ensures w2 < w3
        'r4': (0.1, 1.0),  # ensures w4 < w2
        'r1': (0.1, 1.0)   # ensures w1 < w4
    }

    # Optimize weights
    optimizer = BayesianOptimization(f=evaluate_weights, pbounds=pbounds, random_state=5)
    optimizer.maximize(init_points=10, n_iter=20)  # 10 random + 20 optimized trials

    # Get best parameters
    best_params = optimizer.max['params']
    best_f1_score = optimizer.max['target']

    # Calculate actual best weights
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