import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from datetime import datetime

fields = ['Required age', 'Price', 'DLC count', 'Metacritic score', 'Recommendations', 'Positive',
          'Negative', 'Peak CCU']
df = pd.read_csv('games.csv', skipinitialspace=True, usecols=fields)

X = df.drop('Price', axis=1)
y = df['Price']

print(df.head())
print(df.columns)

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
