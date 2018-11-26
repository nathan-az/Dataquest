import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

cars = pd.read_csv("imports-85.data")
cols = ["symboling",
        "normalized-losses",
        "make",
        "fuel-type",
        "aspiration",
        "num-of-doors",
        "body-style",
        "drive-wheels",
        "engine-location",
        "wheel-base",
        "length",
        "width",
        "height",
        "curb-weight",
        "engine-type",
        "num-of-cylinders",
        "engine-size",
        "fuel-system",
        "bore",
        "stroke",
        "compression-ratio",
        "horsepower",
        "peak-rpm",
        "city-mpg",
        "highway-mpg",
        "price"]
cars.columns = cols
cars.replace("?", np.nan, inplace=True)

numeric_cols = ["normalized-losses",
                "wheel-base",
                "length",
                "width",
                "height",
                "curb-weight",
                "bore",
                "stroke",
                "compression-ratio",
                "horsepower",
                "peak-rpm",
                "city-mpg",
                "highway-mpg",
                "price"]

cars = cars[numeric_cols].apply(pd.to_numeric, errors='coerce')
numeric_cars = cars[numeric_cols]
numeric_cars = cars.dropna(subset=["price"])
numeric_cars = numeric_cars.fillna(numeric_cars.mean())

norm_cars = (numeric_cars - numeric_cars.min()) / (numeric_cars.max() - numeric_cars.min())
norm_cars["price"] = numeric_cars["price"]


def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
    shuffle = np.random.permutation(df.shape[0])
    shuffled_df = df.copy().reset_index().reindex(shuffle)
    split_point = int(len(shuffled_df)*0.8)
    if type(train_col)==list:
        feature_data = shuffled_df[train_col]
    else:
        feature_data = shuffled_df[[train_col]]
    target_data = shuffled_df[[target_col]]
    train_feature = feature_data.iloc[0:split_point]
    train_target = target_data.iloc[0:split_point]
    test_feature = feature_data.iloc[split_point:]
    test_target = target_data.iloc[split_point:]

    knn.fit(train_feature, train_target)

    pred = knn.predict(test_feature)
    rmse = mean_squared_error(test_target, pred) ** 0.5
    return rmse


car_cols = list(norm_cars.columns)
car_cols.remove("price")
col_errors = {}
for col in car_cols:
    col_errors[col] = knn_train_test(col, "price", norm_cars)
errors = pd.Series(col_errors).sort_values()
lowest_two = errors.index[:2].tolist()

