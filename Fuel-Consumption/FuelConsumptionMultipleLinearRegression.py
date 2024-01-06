import pandas as pd
import numpy as np
from sklearn import linear_model


data_frame = pd.read_csv('FuelConsumption.csv')
selected_data_frame = data_frame[
    [
        'ENGINESIZE',
        'CYLINDERS',
        'FUELCONSUMPTION_CITY',
        'FUELCONSUMPTION_HWY',
        'FUELCONSUMPTION_COMB',
        'CO2EMISSIONS'
    ]
]

mask = np.random.rand(len(data_frame)) < 0.8
train = selected_data_frame[mask]
test = selected_data_frame[~mask]

regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(train_x, train_y)

test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regression.predict(test_x)
print(regression.score(test_x, test_y))
