import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x=np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y=np.array([10,15,20,25,30,40,50,60])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

x_in = [5, 12]
y_out = model.predict(np.array(x_in).reshape(-1, 1))
print(y_out)

y_pred = model.predict(x_test)

print(y_pred)


plt.scatter(x_train, y_train, color='green')
y_line = model.predict(x_train)
plt.plot(x_train, y_line, color='red', label='Regression Line')

plt.scatter(x_in, y_out, color='blue', label='predicted', edgecolors='black')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.title('Experience vs Salary')
plt.show()


