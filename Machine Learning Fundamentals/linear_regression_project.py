import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def transform_features(df):
    transformed = df.copy()
    missing_num = transformed.select_dtypes(include=["integer", "float"]).isnull().sum()
    missing_obj = transformed.select_dtypes(include=["object"]).isnull().sum()

    drop_num = missing_num[missing_num>int(0.05*len(transformed))].index
    drop_obj = missing_obj[missing_obj>1].index

    transformed = transformed.drop(columns = list(drop_num)+list(drop_obj))
    keep_num = transformed.select_dtypes(include=["integer", "float"]).columns

    for col in keep_num:
        col_max = transformed[col].value_counts().sort_values(ascending=False).index[0]
        transformed[col].fillna(col_max)

    transformed = transformed.drop(columns = ["PID", "Order"])
    transformed = transformed.drop(columns = ["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"])

    return transformed


def select_features(df):
    return df[["Gr Liv Area", "SalePrice"]]


def train_and_test(df):
    train = df.iloc[:1460]
    test = df.iloc[1460:]
    is_num = df.select_dtypes(include=["float", "integer"])

    features = is_num.columns.drop("SalePrice")
    num_train = train[features]
    num_test = test[features]
    lr = LinearRegression()
    lr.fit(num_train, train[["SalePrice"]])
    pred = lr.predict(num_test)

    rmse = np.sqrt(mean_squared_error(pred, test["SalePrice"]))
    return rmse


housing = pd.read_csv("AmesHousing_1.tsv", delimiter="\t")
transformed = transform_features(housing)
#filtered = select_features(transformed)
#rmse = train_and_test(filtered)
