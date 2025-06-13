#1_MeanMode
import statistics as st

n = int(input('Enter no.of items : '))

def takeInput():
    print('Enter data : ')
    data = []
    for x in range(n):
        data.append(int(input()))
    return data

def getMean(data):
    
    return sum(x for x in data)/n

def getMode(data):

    return st.mode(data)

def getMedian(data):

    return st.median(data)

def calcTendencyMeasure(data):

    mean = getMean(data)
    mode = getMode(data)
    median = getMedian(data)

    return mean, mode, median

def getVariance(data, mean):

    diffSum = sum((x - mean)**2 for x in data)
    return diffSum / n-1

def calcDispersion(data):

    var = getVariance(data, getMean(data))
    stdDev = var**0.5

    return var, stdDev

data = takeInput()

mean, mode, median = calcTendencyMeasure(data)
var, stdDev = calcDispersion(data)

print(data)

print('Mean : ', mean)
print('Mode : ', mode)
print('Median : ', median)
print('Variance : ', var)
print('Std.Deviation : ', stdDev)




#4_linearReg
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
y_out = model.predict(np.array([x_in]).reshape(-1, 1))
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






#5_multiLinReg
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






#6_DecsnTreeReg
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




#7_Knn
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







#8_LogisticReg
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





#9_KMeansCluster
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=100, centers =4, random_state=2) # type: ignore

model = KMeans(n_clusters=4, random_state=3)
model.fit(X)

plt.figure(figsize=(10, 10))
plt.title('Clustering')
plt.scatter(X[:, 0], X[:, 1], c = model.labels_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, marker='X', color = 'red', label='Centers' )
plt.show()



