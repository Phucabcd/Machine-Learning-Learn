import os
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from display_network import *

print("Current directory:", os.getcwd())
print("Data exists:", os.path.exists("./data"))
print(
    "Label exists:",
    os.path.exists("./data/t10k-labels-idx1-ubyte")
)

mndata = MNIST('./data')
mndata.load_testing()
X = mndata.test_images
K = 10

kmeans = KMeans(n_clusters=K, random_state=11).fit(X)
pred = kmeans.predict(X)