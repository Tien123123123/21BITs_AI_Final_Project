import os
import random
import pickle
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), f"../")))
from process_data.preprocessing import preprocess_data
from process_data.calc_score import calc_score
from collaborative.train_model import train_model
from process_data.train_test_split import train_test_split
from evaluation_pretrain.evaluate_data import evaluate_model

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

# Keep track of used weight combinations so we don't generate duplicates
used_weights = set()


def get_unique_weights():
    """
    Generate a unique set of weights for each trial.
    """
    possible_values = [round(x, 1) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    while True:
        w1 = random.choice(possible_values)
        w4 = random.choice(possible_values)
        w2 = random.choice(possible_values)
        w3 = random.choice(possible_values)

        if w1 < w4 < w2 < w3:
            if (w1, w4, w2, w3) not in used_weights:
                used_weights.add((w1, w4, w2, w3))
                return w1, w2, w3, w4


def optimize_and_train(bucket_name, file_name, trial_nums=5, nrows=None):
    """
    Optimize weights and train the best model.

    Args:
        csv_file_path (str): Path to the CSV dataset.
        trial_nums (int): Number of trials (weight sets) to try.

    Returns:
        best_model: The best trained model.
        best_f1_score (float): The best F1 score achieved.
        best_weights (tuple): The weights that achieved the best F1 score.
        best_model_filename (str): The filename of the best model.
    """

    # Preprocess the dataset once
    logging.info("Running preprocessing...")
    df = preprocess_data(bucket_name, file_name, nrows=nrows)
    logging.info(f"Preprocessing complete. Rows in df = {len(df)}")

    best_f1_score = -1.0
    best_weights = None
    best_model = None
    best_model_filename = None

    for trial in range(trial_nums):
        # 1. Generate a unique set of weights for this trial
        w1, w2, w3, w4 = get_unique_weights()
        logging.info(f"Trial #{trial + 1}/{trial_nums}")
        logging.info(f"w1={w1}, w2={w2}, w3={w3}, w4={w4}")

        # 2. Calculate df_weighted with these weights
        df ,df_weighted = calc_score(df, w1, w2, w3, w4)

        # 3. Train-test split
        df_test, df_weighted_split, df_GT = train_test_split(df, df_weighted)

        # 4. Train model
        model_filename = f"models/model_{trial + 1}.pkl"
        model, results = train_model(df_weighted=df_weighted_split)

        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
        if os.path.exists(model_filename):
            logging.info(f"Model saved to {model_filename}")

        logging.info("Finished training model.")

        # 5. Evaluate the model
        with open(model_filename, "rb") as f:
            model_trial = pickle.load(f)
        df_test_eval, df_metrics, mean_f1 = evaluate_model(df_test, df_GT, model_trial, top_N=3)
        logging.info(f"F1 Score for this set: {mean_f1}")

        # 6. Check if it's the best so far
        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_weights = (w1, w2, w3, w4)
            best_model = model_trial
            best_model_filename = model_filename

    # Final summary
    logging.info("Optimization Complete")
    logging.info(f"Best F1 Score: {best_f1_score}")
    logging.info(f"Best Weights (w1, w2, w3, w4): {best_weights}")
    if best_model_filename:
        logging.info(f"Best Model: {best_model_filename}")

    return best_model, best_f1_score, best_weights, best_model_filename


# Example usage
if __name__ == "__main__":
    csv_file_path = r"D:\Download\pyCharmpPro\recomemend_test\3_session_dataset_5M.csv"
    best_model, best_f1_score, best_weights, best_model_filename = optimize_and_train(csv_file_path, trial_nums=5)