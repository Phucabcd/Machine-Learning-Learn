from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

# centers of the clusters
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
# N points cluster
N = 100
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# X input 
X = np.concatenate((X0, X1, X2), axis=0)
# K số lượng cluster cần tìm
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :] 
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
    
    plt.axis('equal')
    plt.plot()
    plt.show()
    
# run 1 with none sklearn 
# kmeans_display(X, original_label)

# random choice K centers
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

# calculate distance and return the index of the closest center
def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1)

# loop, find update centers
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

# check if centers have converged
def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
 
# call all the above functions to implement kmeans
def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

## run 2 with none sklearn 
# (centers, labels, it) = kmeans(X, K)
# print('Centers found by our algorithm:')
# print(centers[-1])

# kmeans_display(X, labels[-1])

#using sklearn to check the result
kmeans = KMeans(n_clusters=K, random_state=11).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_labels = kmeans.predict(X)
kmeans_display(X, pred_labels)