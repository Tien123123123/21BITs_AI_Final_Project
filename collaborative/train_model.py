from surprise.model_selection import cross_validate
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import SVD


def train_model(df_weighted, model=SVD(), param_grid=False):
    # Define rating scale
    min_score = df_weighted["score"].min()
    max_score = df_weighted["score"].max()
    reader = Reader(rating_scale=(min_score, max_score))

    # Load data for Surprise
    data = Dataset.load_from_df(df_weighted, reader)
    trainset = data.build_full_trainset()

    if param_grid == True:
        model = GridSearchCV(SVD, param_grid=param_grid, measures=["rmse", "mae"], cv=5)
        model.fit(trainset)
        results = [model.best_score["rmse"], model.best_params["rmse"], model.best_score["mae"], model.best_params["mae"]]
    elif param_grid == False:
        model.fit(trainset)  # Sử dụng trainset ở đây
        results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    return model, results



