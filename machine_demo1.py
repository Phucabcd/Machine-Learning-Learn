from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#import to 75% training and 25% testing data
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

#load_iris datasets of sklearn (150 values in flower iris)
iris_data = load_iris()
print(iris_data)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, random_state=10)
# print(X_test)
# print(type(X_train))

model = DecisionTreeClassifier()
train_model = model.fit(X_train, y_train)

X_new = np.array([[6.0, 3.23, 4.5, 2.25]])
# print(train_model.predict(X_new))

#score dung de du doan
# print(train_model.score(X_test, y_test))
