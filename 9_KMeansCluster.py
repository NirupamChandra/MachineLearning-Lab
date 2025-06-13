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