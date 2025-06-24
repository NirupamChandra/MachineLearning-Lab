# import statistics as st

# n = int(input('Enter no.of inputs : '))
# data = []

# def getData(n):
#     print('Enter inputs : ')
    
#     for x in range(n):
#         data.append(int(input()))

# def getCentralTendencies():

#     mean = st.mean(data)
#     mode = st.mode(data)
#     median = st.median(data)

#     return mean, mode, median

# def getDispersions():

#     stdev = st.stdev(data)
#     var = st.variance(data)

#     return stdev, var

# def printResults():
    
#     print(getCentralTendencies())
#     print(getDispersions())

# getData(n)
# printResults()


# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from matplotlib import pyplot as plt

# x = np.array([1, 2, 3, 4, 5, 6,7 ,8 ,9, 10]).reshape(-1, 1)
# y = np.array([temp*10 for temp in x])

# print(x)
# print(y)
# print(x.shape)
# print(y.shape)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# plt.figure(figsize=(10, 8))
# plt.title('Experience vs salary')
# plt.scatter(x_train, y_train, color='blue', label='Actual value')
# plt.plot(x_train, model.predict(x_train), 'r--', color='red', label='Predicted')
# plt.ylabel('Salary')
# plt.xlabel('Experience')
# plt.legend()
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error

# datas = {
#     'Area' : [1500, 2000, 2500, 2000, 1400, 1500, 1700, 1800, 1900, 2000],
#     'Age' : [5, 4, 4, 4, 3, 3, 2, 2, 1, 1],
#     'Price' : [x for x in range(15000, 25000, 1000)],
#     'Bedrooms'  : [2, 2, 3, 4, 3, 4, 4, 5, 3, 5],
# }

# print(datas)

# df = pd.DataFrame(datas)

# print(df)
# x = df.drop('Price', axis=1, inplace=False)
# y = df[['Price']]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# model = LinearRegression()
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# print(f'r2_SCORE : {r2_score(y_test, y_pred)}')
# print(f'MAE : {mean_absolute_error(y_test, y_pred)}')

# plt.figure(figsize=(8, 8))
# plt.title('Actual vs Predicted')
# plt.scatter(y_test, y_pred, color='green', label='(Actual, Predicted)')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', color='red', label='Actual Line')
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.legend()
# plt.grid(True)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor, plot_tree
# from sklearn.metrics import r2_score, mean_absolute_error
# from sklearn.datasets import load_diabetes

# datas = load_diabetes()
# x = datas.data # type: ignore
# y = datas.target # type: ignore

# print(datas.feature_names) # type: ignore

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model = DecisionTreeRegressor(max_depth = 2, random_state=42)
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# print(f'R2_score : {r2_score(y_test,y_pred):.3f}')
# print(f'MAE : {mean_absolute_error(y_test, y_pred):.3f}')

# plt.figure(figsize=(10, 10))
# plot_tree(model, max_depth=2, filled=True)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier

# digits = load_digits()
# x = digits.data
# y = digits.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(f'Accuracy score : {accuracy_score(y_test, y_pred)}')

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_digits
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# digits = load_digits()
# x = digits.data
# y = (digits.target==9).astype(int)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# model = LogisticRegression(max_iter = 1000, random_state=42)

# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(f'Accuracy score : {accuracy_score(y_test, y_pred):.3f}')

# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans

# datas, _= make_blobs(n_samples = 400, centers=4, random_state=4)

# model = KMeans(n_clusters=4, random_state=4)
# model.fit(datas)

# plt.figure()
# plt.scatter(datas[:, 0], datas[:, 1], c = model.labels_)
# plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, marker='X', color='red', label='Centers')
# plt.legend()
# plt.grid(True)
# plt.show()