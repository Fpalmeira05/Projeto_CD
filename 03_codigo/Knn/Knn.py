import numpy as np
from collections import Counter


class Knn:
    """
    K-Nearest Neighbors Classifier implemented from scratch using highly
    optimized NumPy vectorization.

    Attributes:
        k -- Number of neighbors to use.

    Methods:
        fit -- Fit the classifier to the training data.
        predict -- Predict the class labels of the test data.
        score -- Calculate the accuracy of the classifier.
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # We ensure they are NumPy arrays just in case a Pandas DataFrame is passed
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        # Predict the label for each sample in the test set
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # np.linalg.norm calculates the Euclidean distance
        distances = np.linalg.norm(self.X_train - x, axis=1)

        #Get k nearest samples' indices
        k_indices = np.argsort(distances)[:self.k]

        #Extract the actual labels of those neighbors
        k_nearest_labels = self.y_train[k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]#returns a tuple of the most common and how many times it appears. The [0][0] means only the feature

    def score(self, X_test, y_test):
        """
        method to calculate the accuracy of the model.
        """
        y_test = np.array(y_test)
        predictions = self.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        return accuracy