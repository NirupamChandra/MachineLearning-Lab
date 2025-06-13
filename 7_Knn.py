import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data  # type: ignore
y = digits.target # type: ignore


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

conf = confusion_matrix(y_test, y_pred)
print(conf)

print(f'Accuracy score : {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report : ')
print(classification_report(y_test, y_pred))



