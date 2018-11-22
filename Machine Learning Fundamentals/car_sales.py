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

norm_cars = (numeric_cars-numeric_cars.min())/(numeric_cars.max()-numeric_cars.min())