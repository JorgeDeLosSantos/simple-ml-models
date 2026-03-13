import numpy as np


class KNNClassifier:
    """
    A simple implementation of the k-Nearest Neighbors (KNN) classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for classification.

    metric : str, default="euclidean"
        Distance metric to use. Currently supports:
        - "euclidean"

    weights : str, default="uniform"
        Weighting strategy. Currently supports:
        - "uniform"
        - "distance" (future implementation)
    """

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform", p=2):
        # Configuration parameters
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.p = p  # For Minkowski distance (p=2 is Euclidean)

        # Attributes set during fitting
        self.X_train = None
        self.y_train = None
        self.n_samples = None
        self.n_features = None
        self.classes_ = None

        self._is_fitted = False

    # ============================================================
    # Public API
    # ============================================================

    def fit(self, X, y):
        """
        Fit the KNN classifier by storing the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Validate dimensions
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        n_samples, n_features = X.shape

        if self.n_neighbors > n_samples:
            raise ValueError(
                "n_neighbors cannot be greater than number of training samples."
            )

        # Store training data
        self.X_train = X
        self.y_train = y

        # Store metadata
        self.n_samples = n_samples
        self.n_features = n_features
        self.classes_ = np.unique(y)

        # Mark as fitted
        self._is_fitted = True

        return self


    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        ------- 
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input data must be 2D.")

        if X.shape[1] != self.n_features:
            raise ValueError("Feature mismatch.")

        # # ||X||^2
        # X_sq = np.sum(X**2, axis=1, keepdims=True)

        # # ||X_train||^2
        # train_sq = np.sum(self.X_train**2, axis=1)

        # # Cross term: X @ X_train^T
        # cross = X @ self.X_train.T

        # # Squared distances: ||X - X_train||^2 = ||X||^2 + ||X_train||^2 - 2 * (X @ X_train^T)
        # distances = X_sq + train_sq - 2 * cross
        # Compute pairwise distances
        if self.metric == "euclidean":
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            train_sq = np.sum(self.X_train**2, axis=1)
            cross = X @ self.X_train.T
            distances = X_sq + train_sq - 2 * cross
            distances = np.maximum(distances, 0)

        elif self.metric == "manhattan":
            distances = np.sum(
                np.abs(X[:, None, :] - self.X_train[None, :, :]),
                axis=2
            )

        elif self.metric == "minkowski":
            distances = np.sum(
                np.abs(X[:, None, :] - self.X_train[None, :, :]) ** self.p,
                axis=2
            )

        else:
            raise ValueError("Unsupported metric.")

        # Get indices of the k nearest neighbors
        k = self.n_neighbors
        neighbor_indices = np.argpartition(distances, k-1, axis=1)[:, :k]

        predictions = []

        for i, neighbors in enumerate(neighbor_indices):

            labels = self.y_train[neighbors]

            if self.weights == "uniform":

                classes, counts = np.unique(labels, return_counts=True)
                prediction = classes[np.argmax(counts)]

            elif self.weights == "distance":

                neighbor_distances = distances[i, neighbors]
                weights = 1 / (neighbor_distances + 1e-9)

                class_scores = {}

                for label, w in zip(labels, weights):
                    class_scores[label] = class_scores.get(label, 0) + w

                prediction = max(class_scores, key=class_scores.get)

            else:
                raise ValueError("Unsupported weights type.")

            predictions.append(prediction)

        return np.array(predictions)

    def score(self, X, y):
        """
        Compute the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean classification accuracy.
        """

        self._check_is_fitted()

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("Input data must be 2D.")

        if y.ndim != 1:
            raise ValueError("y must be 1D.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        y_pred = self.predict(X)

        return np.mean(y_pred == y)

    # ============================================================
    # Utility Methods
    # ============================================================

    def _check_is_fitted(self):
        """
        Check if the model has been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("This MyKNNClassifier instance is not fitted yet.")
