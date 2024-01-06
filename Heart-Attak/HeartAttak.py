import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv('Heart.csv')
df.rename(columns={'output': 'heartattak'}, inplace=True)

X = df[
    [
        'age',
        'sex',
        'cp',
        'trtbps',
        'chol',
        'fbs',
        'restecg',
        'thalachh',
        'exng',
        'oldpeak',
        'slp',
        'caa',
        'thall'
    ]
]

X = np.asanyarray(X)
y = np.asanyarray(df['heartattak'])

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X.astype(float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

k = 5
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
y_hat = neigh.predict(X_test)
print("Predict: ", metrics.accuracy_score(y_test, y_hat))

x = np.asanyarray(df[['age']])
y = np.asanyarray(df[['chol']])
plt.scatter(x, y, color='blue')
plt.xlabel('Age')
plt.ylabel('Heart attak')
plt.show()
