import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

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


def knn_train_test(train_col, target_col, df, k):
    knn = KNeighborsRegressor(n_neighbors = k)
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
    col_errors[col] = knn_train_test(col, "price", norm_cars, 5)
univar_errors_k5 = pd.DataFrame(pd.Series(col_errors).sort_values())
print("Univariate RMSE values with 5 neighbors:\n{}\n".format(univar_errors_k5.to_string()))


##################################################
#this data is separated because it isn't used
col_errors = {}
for col in car_cols:
    col_errors[col] = {}
    for i in (1, 3, 5, 7, 9):
        col_errors[col][i] = knn_train_test(col, "price", norm_cars, i)
univar_errors = pd.DataFrame.from_dict(col_errors)
univar_errors.plot.line()
plt.show()
print("Univariate RMSE values with variable neighbors:\n{}\n".format(univar_errors.to_string()))
##################################################


multivar_errors_d = {}
features = [univar_errors_k5.index[:i].tolist() for i in range(2, 6)]
num_features = list(range(2, 6))
rmses = [knn_train_test(feature, "price", norm_cars, 5) for feature in features]
multivar_errors = pd.DataFrame({"features": features, "rmse":rmses, "num_features":num_features})\
    .sort_values(by="rmse").reset_index(drop=True)
print("")


best_multivar_errors = [(cols, {k:knn_train_test(cols, "price", norm_cars, k) for k in range(1, 26)}) \
                        for cols in multivar_errors.features[:3]]
best_multivar_errors = pd.DataFrame(best_multivar_errors)
best_multivar_errors = pd.concat([best_multivar_errors.drop(1, axis=1), \
                                  pd.DataFrame(best_multivar_errors[1].tolist())], axis=1).T
col_names = [str(col) for col in best_multivar_errors.iloc[0]]
best_multivar_errors.columns = col_names
best_multivar_errors.drop(0, inplace=True)
best_multivar_errors = best_multivar_errors.apply(pd.to_numeric, errors="coerce")
best_multivar_errors.reset_index(inplace=True)
best_multivar_errors.rename(index=str, columns={"index": "k"}, inplace=True)
best_multivar_errors.loc[25] = np.nan
for col in best_multivar_errors.columns[1:]:
    min_k = best_multivar_errors.loc[best_multivar_errors[col]==best_multivar_errors[col].min(), "k"]
    best_multivar_errors.loc[25, col] = np.float64(min_k)
print(best_multivar_errors.to_string())

x = best_multivar_errors["k"][:-1]
y = best_multivar_errors[best_multivar_errors.columns[1:]][:-1]
best_multivar_errors.iloc[:-1].plot.line(x="k", y=best_multivar_errors.columns[1:])