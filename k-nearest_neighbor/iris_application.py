import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import time

# iris dataset có sẵn trong sklearn và nhiều dataset khác
iris = datasets.load_iris()
iris_X = iris.data          # feature
iris_y = iris.target        # lable/class

# df = pd.DataFrame(iris_y)
# print(df)

print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))

# Gán nhãn các class 0, 1, 2
X0 = iris_X[iris_y == 0, :]
print('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1, :]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2, :]
print('\nSamples from class 2:\n', X2[:5,:])

X_train, x_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)
print(f"\nTraining test:  {len(y_train)}")
print(f"Test size: {len(y_test)}")

#K = 1 
start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors= 1, p = 2) # p = norm 
clf.fit(X_train, y_train)
y_pred = clf.predict(x_test)

# Method: ground truth 
# nhãn/label/đầu ra thực sự của các điểm trong test data
print("\nPrint results for 20 test data points:")
print("Result predict: ", y_pred[20:40])  
print("Result test    : ", y_test[20:40]) 


# Evaluation method
print(f"\nAccuracy of 1NN:  {100*accuracy_score(y_test, y_pred)}")


# Major voting                                              # weights mặc định: uniform các value = nhau
clf2 = neighbors.KNeighborsClassifier(n_neighbors= 10, p = 2, weights= 'distance') 
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(x_test)

print(f"\nAccuracy of 10NN with major voting: {100*accuracy_score(y_test, y_pred2)}")
end_time = time.time()
print("Running time: %.2f (s)" % (end_time - start_time))

    

