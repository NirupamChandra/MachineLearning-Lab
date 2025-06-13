import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

digits = load_digits()
x = digits.data # type: ignore
# y = digits.target
y = (digits.target == 9).astype(int) # type: ignore

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Accuracy score : {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))

