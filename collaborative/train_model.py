import pickle

from surprise.model_selection import cross_validate
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import SVD


def train_model(df_weighted, model_file, model=SVD(), param_grid=None):
    # Define rating scale
    min_score = df_weighted["score"].min()
    max_score = df_weighted["score"].max()
    reader = Reader(rating_scale=(min_score, max_score))

    # Load data for Surprise
    data = Dataset.load_from_df(df_weighted, reader)
    trainset = data.build_full_trainset()

    if param_grid:
        # Perform grid search to find best parameters
        model = GridSearchCV(SVD, param_grid=param_grid, measures=["rmse", "mae"], cv=5, refit=True)
        model.fit(data)
        best_rmse = model.best_score["rmse"]
        best_params_rmse = model.best_params["rmse"]
        best_mae = model.best_score["mae"]
        best_params_mae = model.best_params["mae"]
        results = [best_rmse, best_params_rmse, best_mae, best_params_mae]
    else:
        # Fit the model without grid search
        model.fit(trainset)
        results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

    # Save model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print("Save Successfully!")