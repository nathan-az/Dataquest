import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")
plt.interactive(True)

# cleaning data step 1: replace empty "?" datapoints with NaN
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

# step 2: identify numeric (and truly quantitative) columns for use in model
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

# step 3: convert datatype to numeric, fill NaN. Chosen method was to replace with column mean. Alternatively, to drop rows
cars = cars[numeric_cols].apply(pd.to_numeric, errors='coerce')
numeric_cars = cars[numeric_cols]
numeric_cars = cars.dropna(subset=["price"])
numeric_cars = numeric_cars.fillna(numeric_cars.mean())

# feature scaling (min-max normalisation). target "price" is left alone
norm_cars = (numeric_cars - numeric_cars.min()) / (numeric_cars.max() - numeric_cars.min())
norm_cars["price"] = numeric_cars["price"]


# below function returns the error of a predictive model. can take list inputs for multiple features. default k=5
def knn_train_test(train_col, target_col, df, k=5):
    knn = KNeighborsRegressor(n_neighbors=k)
    np.random.seed(1)
    shuffle = np.random.permutation(df.shape[0])
    shuffled_df = df.copy().reset_index().reindex(shuffle)
    split_point = int(len(shuffled_df) * 0.8)
    if type(train_col) == list:
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


# simple univariate test with k=5
car_cols = list(norm_cars.columns)
car_cols.remove("price")
col_errors = {}
for col in car_cols:
    col_errors[col] = knn_train_test(col, "price", norm_cars, 5)
univar_errors_k5 = pd.DataFrame(pd.Series(col_errors).sort_values())
print("\n{}\nUnivariate RMSE values with 5 neighbors:\n{}\n".format("-" * 30, univar_errors_k5.to_string()))

''' the below code is written for the sake of learning but is commented out because it is not used and its insights
are not especially interesting
col_errors = {}
for col in car_cols:
    col_errors[col] = {}
    for i in (1, 3, 5, 7, 9):
        col_errors[col][i] = knn_train_test(col, "price", norm_cars, i)
univar_errors = pd.DataFrame.from_dict(col_errors)
univar_errors.plot.line()
print("Univariate RMSE values with variable neighbors:\n{}\n".format(univar_errors.to_string()))
'''

# multivariate test: takes the best 2 to best 4 features from univariate test
multivar_errors_d = {}
features = [univar_errors_k5.index[:i].tolist() for i in range(2, 6)]
num_features = list(range(2, 6))
rmses = [knn_train_test(feature, "price", norm_cars, 5) for feature in features]
multivar_errors = pd.DataFrame({"features": features, "rmse": rmses, "num_features": num_features}) \
    .sort_values(by="rmse").reset_index(drop=True)
print("")

# below, we take the best 3 of the 4 multivariate tests and vary the number of neighbors in the model
# there must be a more elegant method than the double dict comprehension, but this functions perfectly
best_multivar_errors = [(cols, {k: knn_train_test(cols, "price", norm_cars, k) for k in range(1, 26)}) \
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

# we also create a row which identifies the num_neighbours value which minimised rmse (bias not considered yet)
best_multivar_errors.loc[25] = np.nan
for col in best_multivar_errors.columns[1:]:
    min_k = best_multivar_errors.loc[best_multivar_errors[col] == best_multivar_errors[col].min(), "k"]
    best_multivar_errors.loc[25, col] = np.float64(min_k)
best_multivar_errors.loc[25, "k"] = "MIN"
print(best_multivar_errors.to_string())

x = best_multivar_errors["k"][:-1]
y = best_multivar_errors[best_multivar_errors.columns[1:]][:-1]
best_multivar_errors.iloc[:-1].plot.line(x="k", y=best_multivar_errors.columns[1:], title= \
    "Performance of top feature combinations:\nRMSE vs k neighbors (num_neighbors)")


# below function is much simpler and allows sklearn to manage the data separation, shuffling, and folding
# also allows us to identify bias which occurs in varying fold counts (over/undertraining)
def knn_folding(train_cols, target_col, df, min_folds, max_folds, fold_step, k=5):
    if type(train_cols) == list:
        feature_data = df[train_cols]
    elif type(train_cols) == str:
        feature_data = df[[train_cols]]
    else:
        print("train_col type must be list, not other arraylike. Use list() or .tolist() please")
        return
    target_data = df[[target_col]]
    rmses_desc = {}
    rmses_raw = {}
    for fold in range(min_folds, max_folds + 1, fold_step):
        rmses_desc[fold] = {}
        kf = KFold(n_splits=fold, shuffle=True, random_state=1)
        knn = KNeighborsRegressor(n_neighbors=k)
        mses = cross_val_score(knn, X=feature_data, y=target_data, scoring="neg_mean_squared_error", cv=kf)
        rmses = np.sqrt(np.absolute(mses))
        rmses_raw[fold] = rmses
        rmses_desc[fold]["mean"] = np.mean(rmses)
        rmses_desc[fold]["std"] = np.std(rmses)
    return rmses_desc, rmses_raw


target_col = "price"
train_cols = norm_cars.columns.tolist()
train_cols.remove(target_col)
desc, raw = knn_folding(train_cols, target_col, norm_cars, 2, 10, 1, 5)
folding_errors = pd.DataFrame.from_dict(desc).T.reset_index()
folding_errors.rename(index=str, columns={"index": "Number of folds"}, inplace=True)
folding_errors.plot(x="Number of folds", y=["mean", "std"], kind="line", \
                    title="Cross validation:\nMean and std of error by fold count")
