import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit


data_frame = pd.read_csv('HousePrice.csv')
data_frame['Area'] = pd.to_numeric(data_frame['Area'], errors='coerce')
data_frame = data_frame.dropna(subset=['Area', 'Address'])
label_encoder = LabelEncoder()
data_frame['Address'] = label_encoder.fit_transform(data_frame['Address'])

data_frame = data_frame.sort_values('Area', axis=0)
data_frame = data_frame.iloc[:-75]

mask = np.random.rand(len(data_frame)) < 0.8
train = data_frame[mask]
test = data_frame[~mask]

regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']])
train_y = np.asanyarray(train[['Price']])
regression.fit(train_x, train_y)

test_x = np.asanyarray(test[['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']])
test_y = np.asanyarray(test[['Price']])
print("Predict", regression.score(test_x, test_y))
xdata = train['Area']/max(train['Area'])
ydata = train['Price']/max(train['Price'])


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

popt, pcov = curve_fit(sigmoid, xdata, ydata)

x = np.linspace(0, 300, 55)
x = x/max(x)
plt.figure()
y = sigmoid(x, *popt)
plt.scatter(xdata, ydata, color='blue')
plt.plot(x, y, color='red')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
