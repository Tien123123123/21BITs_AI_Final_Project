import os
import kagglehub
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

# Download and Load Dataset
path = kagglehub.dataset_download("waqi786/e-commerce-clickstream-and-transaction-dataset")
file = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".csv")]

df = pd.read_csv(file[0], nrows=10000)

# Data Preprocessing
# Remove and Add Features
selected_event_types = ["product_view", "add_to_cart", "purchase"]
df = df[df["EventType"].isin(selected_event_types)]
df = df.drop("Amount", axis=1)
df["Outcome"] = df["Outcome"].fillna("unpurchase")

# Calculate Total view, cart, purchase
is_view = df[df["EventType"]=="product_view"].groupby(["UserID", "SessionID", "ProductID"]).size().reset_index(name="is_view")
is_cart = df[df["EventType"]=="add_to_cart"].groupby(["UserID", "SessionID", "ProductID"]).size().reset_index(name="is_cart")
is_purchase = df[df["EventType"]=="purchase"].groupby(["UserID", "SessionID", "ProductID"]).size().reset_index(name="is_purchase")

total_view = df[df["EventType"]=="product_view"].groupby(["UserID", "SessionID"]).size().reset_index(name="total_view")
total_cart = df[df["EventType"]=="add_to_cart"].groupby(["UserID", "SessionID"]).size().reset_index(name="total_cart")
total_purchase = df[df["EventType"]=="purchase"].groupby(["UserID", "SessionID"]).size().reset_index(name="total_purchase")

df = (df.merge(is_view, on=["UserID", "SessionID", "ProductID"], how="left")
        .merge(is_cart, on=["UserID", "SessionID", "ProductID"], how="left")
        .merge(is_purchase, on=["UserID", "SessionID", "ProductID"], how="left")
        .merge(total_view, on=["UserID", "SessionID"], how="left")
        .merge(total_cart, on=["UserID", "SessionID"], how="left")
        .merge(total_purchase, on=["UserID", "SessionID"], how="left"))
df.fillna(0, inplace=True)

# Calculate Duration and Total Duration by seconds
df = df.sort_values(by=["UserID", "SessionID", "Timestamp"], ascending=[True, True, True])
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["duration"] = df.groupby(["UserID", "SessionID"])["Timestamp"].diff().fillna(pd.Timedelta(seconds=0))
single_item_sessions = df.groupby(["UserID", "SessionID"]).size() == 1
sessions_to_modify = df[df[["UserID", "SessionID"]].apply(tuple, axis=1).isin(single_item_sessions[single_item_sessions].index)]
df.loc[sessions_to_modify.index, "duration"] = pd.Timedelta(seconds=900)

df["duration"] = df["duration"].dt.total_seconds()
df["total_duration"] = df.groupby(["UserID", "SessionID"])["duration"].transform("sum")

# F1: View - F2: Cart - F3: Purchase - F4: Duration
df['F1'] = df.apply(
    lambda row: row['is_view'] / row['total_view'] if row['EventType'] == 'product_view' else 0, axis=1
)
df['F2'] = df.apply(
    lambda row: row['is_cart'] / row['total_cart'] if row['EventType'] == 'add_to_cart' else 0, axis=1
)
df['F3'] = df.apply(
    lambda row: row['is_purchase'] / row['total_purchase'] if row['EventType'] == 'purchase' else 0, axis=1
)
df['F4'] = df.apply(
    lambda row: row['duration'] / row['total_duration'], axis=1
)
print(df.isnull().sum())

# Calculate Interesting Score of User in each Item
initial_weights = {"F1": 0.1, "F2": 0.25, "F3": 0.45, "F4": 0.2}
df["score"] = (
        initial_weights["F1"] * df["F1"] +
        initial_weights["F2"] * df["F2"] +
        initial_weights["F3"] * df["F3"] +
        initial_weights["F4"] * df["F4"]
)
df_weighted = df.groupby(["UserID", "ProductID"])["score"].sum().reset_index(name="score")

# Model and Evaluation
# Preparing Data
selected_features = ["UserID", "ProductID", "score"]
df_weighted = df[selected_features]

min_score = df_weighted["score"].min()
max_score = df_weighted["score"].max()
reader = Reader(rating_scale=(min_score, max_score))
data = Dataset.load_from_df(df_weighted, reader)

# trainset, testset = train_test_split(data, test_size=0.2)

# Deploy Model with Hyperparameter Tuning Method
param_grid = {
    'n_factors': [10, 20, 30],  # Số lượng latent factors
    'n_epochs': [5, 10, 20],    # Số vòng lặp huấn luyện
    'lr_all': [0.002, 0.005],   # Learning rate cho các tham số
    'reg_all': [0.02, 0.05],    # Regularization term
}

grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
grid_search.fit(data)

# Result
print(f"Best RMSE: {grid_search.best_score['rmse']}")
print(f"Best MAE: {grid_search.best_score['mae']}")
print("Best parameters:", grid_search.best_params)

# Model after Tuning
best_model = grid_search.best_estimator['rmse']