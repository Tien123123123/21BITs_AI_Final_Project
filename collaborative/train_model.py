from surprise.model_selection import cross_validate
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import SVD


def train_model(df_weighted, model=SVD, param_grid=False):
    # Define rating scale
    min_score = df_weighted["score"].min()
    max_score = df_weighted["score"].max()
    reader = Reader(rating_scale=(min_score, max_score))

    # Load data for Surprise
    data = Dataset.load_from_df(df_weighted, reader)
    # trainset, testset = train_test_split(data, test_size=0.2, random_state=123)

    # Deploy and Train Model
    # model = model
    # model.fit(data)

    # Metrics to evaluate
    # predictions = model.test(testset)
    # results = [rmse(predictions), mae(predictions)]

    if param_grid:
        model = GridSearchCV(model, param_grid=param_grid, measures=["rmse", "mae"], cv=5)
        model.fit(data)
        results = [model.best_score["rmse"], model.best_params["rmse"], model.best_score["mae"], model.best_params["mae"]]
    else:
        model = model
        results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    return model, results


