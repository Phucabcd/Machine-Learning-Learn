import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For nearest neighbor this is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - y_pred: A numpy array of shape (num_test,) containing predicted labels
          for the test data, where y_pred[i] is the predicted label for the test
          point X[i].  
        """
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        # Loop over all test rows
        for i in range(num_test):
            # Find the nearest training sample to the i-th test sample
            distances = np.linalg.norm(self.X_train - X[i], axis=1)
            closest_index = np.argmin(distances)
            y_pred[i] = self.y_train[closest_index]

        return y_pred