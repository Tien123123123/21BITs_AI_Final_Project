import time
import pickle
import pandas as pd
from process_data.preprocessing import preprocess_data
from collaborative.train_model import train_model

# This will load full data or combined 2 data and load
def load_data(root):
    df = pd.read_csv(root)
    df, df_weighted = preprocess_data(df)
    return df_weighted

def tracking_pretrain(root=""):
    while True: # 1. Load data 2. Train data 3. Save data
        df = load_data(root)
        model, _ = train_model(df)
        model_save = "models/collaborative.pkl"
        with open(model_save, "wb") as f:
            pickle.dump(model, f)
            # print(f"Model save successfully")

        # Repeat pretrain after 1hr
        time.sleep(10000)

if __name__ == '__main__':
    root = ""
    train_model(root)
