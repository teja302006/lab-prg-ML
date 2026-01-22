import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X = np.array([[1],[2],[3],[4],[5],[10],[11],[12],[13],[14]])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=2,max_iter=200)
gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)
probabilities = gmm.predict_proba(X_scaled)

print("Cluster Labels:",labels)
print("Cluster Means:",gmm.means_)
print("Cluster Weights:",gmm.weights_)
print("Silhouette Score:",silhouette_score(X_scaled,labels))
print("Membership Probabilities:")
print(probabilities)
