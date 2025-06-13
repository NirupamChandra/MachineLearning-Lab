import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

datas = load_diabetes()

X = datas.data # type: ignore
y = datas.target # type: ignore
print(datas.feature_names) # type: ignore

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=2, random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'R2 Score : {r2_score(y_test, y_pred):.2f}')
print(f'MAE : {mean_absolute_error(y_test, y_pred):.2f}')

plt.figure(figsize=(15, 10))
plt.title('Decision Tree Regressor')
plot_tree(model, filled=True, max_depth=2)
plt.show()
