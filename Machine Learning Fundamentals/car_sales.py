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
change_type = ["normalized-losses",
"bore",
"stroke",
"horsepower",
"peak-rpm",
"price"]
cars[change_type] = cars[change_type].apply(pd.to_numeric, errors='coerce')
cars = cars.dropna(subset=["price"])