import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# We will use the first two features of the iris dataset, namely Sepal length and Sepal width
X = iris.data[:, :2]

# Apply k-means algorithm to the data with 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=0).fit(X)

# Apply k-means algorithm to the data with 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=0).fit(X)

# Plot the clusters using different colors for the different clusters
plt.figure(figsize=(10, 4))

# Plot the results for two clusters
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_2.labels_, cmap='viridis')
plt.scatter(kmeans_2.cluster_centers_[:, 0], kmeans_2.cluster_centers_[:, 1], marker='*', s=300, c='r')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-means clustering with 2 clusters')

# Plot the results for three clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_3.labels_, cmap='viridis')
plt.scatter(kmeans_3.cluster_centers_[:, 0], kmeans_3.cluster_centers_[:, 1], marker='*', s=300, c='r')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-means clustering with 3 clusters')

plt.show()
#There is an error or a problem with my code the output works but
# there's a warning which shows up and also I don't know why "cluster_centers_" and "labels_"
# showing me error can you please help add a comment when marking and tell me how I can change it or do
# is it just like that??