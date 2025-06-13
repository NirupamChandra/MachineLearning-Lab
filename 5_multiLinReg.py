import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

data = {
    'Size'      : [s for s in range(1500, 2500, 100)],
    'Bedrooms'  : [2, 2, 3, 4, 3, 4, 4, 5, 3, 5],
    'Age'       : [1, 4, 4, 2, 1, 3, 2, 3, 2, 1],
    'Price'     : [p for p in range(15000, 25000, 1000)]
    
}

df = pd.DataFrame(data)
# print(df.head())

X = df.drop('Price', axis=1, inplace=False)
y = df[['Price']]
def test():
    # print(df.columns)
    # print(X.columns)
    # print(X['Bedrooms'].dtype)
    return False


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('R2 Score : ', r2_score(y_test, y_pred))
print(f'MAE Score : {mean_absolute_error(y_test, y_pred):.2f}')

plt.figure(figsize=(8, 12))
plt.scatter(y_test, y_pred, color='green', label='Predicted vs Actual')
plt.plot([y.min(), y.max()] , [y.min(), y.max()] , 'r--' , label = 'Perfect prediction Line')
plt.xlabel('ACtual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()


