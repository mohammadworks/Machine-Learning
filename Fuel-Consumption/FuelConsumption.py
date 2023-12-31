import pandas as pd
import numpy as np
from sklearn import linear_model


df = pd.read_csv('FuelConsumption.csv')
cdf = df[
    [
        'ENGINESIZE',
        'CYLINDERS',
        'FUELCONSUMPTION_CITY',
        'FUELCONSUMPTION_HWY',
        'FUELCONSUMPTION_COMB',
        'CO2EMISSIONS'
    ]
]


msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print(regr.score(test_x, test_y))
