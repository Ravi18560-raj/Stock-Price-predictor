import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import yfinance as yf

data = yf.download(
    "AAPL",
    start = "2020-01-01",
    end = "2025-01-01",
    auto_adjust = False
    )
print(data.head())
print(data.isnull().sum())
data.dropna(inplace=True)
x = data[['Open','High','Low','Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MAE:",mae)
print("RMSE:",rmse)
print("R2 SCORE:",r2)

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.show()

