import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score


data_frame = pd.read_csv('FuelConsumption.csv')
selected_data_frame = data_frame[
    [
        'ENGINESIZE',
        'CYLINDERS',
        'FUELCONSUMPTION_COMB',
        'CO2EMISSIONS'
    ]
]

mask = np.random.rand(len(data_frame)) < 0.8
train = selected_data_frame[mask]
test = selected_data_frame[~mask]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

regression = linear_model.LinearRegression()
train_y_ = regression.fit(train_x_poly, train_y)

test_x_poly = poly.fit_transform(test_x)
test_y_ = regression.predict(test_x_poly)
print(r2_score(test_y, test_y_))


