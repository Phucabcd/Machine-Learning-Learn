import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.cluster import KMeans

mndata = MNIST('./data')
mndata.load_testing()
X = np.array(mndata.test_images)
n_samples = 11 
K = 10

kmeans = KMeans(n_clusters=K, random_state=11).fit(X)
pred = kmeans.predict(X)

fig, axes = plt.subplots(K, n_samples, figsize=(n_samples, K))

for cluster_id in range(K):
    idx = np.where(pred == cluster_id)[0]  # index các ảnh thuộc cluster này
    for col in range(n_samples):
        ax = axes[cluster_id, col]
        img = X[idx[col]].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('image/cluster_grid.png', bbox_inches='tight', facecolor='black')
plt.show()