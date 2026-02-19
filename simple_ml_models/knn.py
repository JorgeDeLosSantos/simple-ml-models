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

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform"):
        # Configuration parameters
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

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
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        if X.shape[1] != self.n_features:
            raise ValueError(
                "Number of features in input does not match training data."
            )

        n_samples_test = X.shape[0]
        predictions = np.empty(n_samples_test, dtype=self.y_train.dtype)

        for i in range(n_samples_test):
            predictions[i] = self._predict_one(X[i])

        return predictions


    # ============================================================
    # Internal Methods (Private)
    # ============================================================

    def _predict_one(self, x):
        """
        Predict the class label for a single sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)

        Returns
        -------
        label : int or str
            Predicted class label.
        """
        self._check_is_fitted()

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError("Input sample must be a 1D array.")

        # Step 1: Compute distances
        distances = self._compute_distances(x)

        # Step 2: Get k nearest neighbors
        neighbor_indices = self._get_k_neighbors(distances)

        # Step 3: Extract neighbor labels
        neighbor_labels = self.y_train[neighbor_indices]

        # Step 4: Extract neighbor distances (for future weighted voting)
        neighbor_distances = distances[neighbor_indices]

        # Step 5: Vote
        prediction = self._vote(
            neighbor_labels,
            neighbor_distances=neighbor_distances
        )

        return prediction


    def _compute_distances(self, x):
        """
        Compute distances between a single sample and all training samples.

        Parameters
        ----------
        x : ndarray of shape (n_features,)

        Returns
        -------
        distances : ndarray of shape (n_train,)
        """
        if self.metric != "euclidean":
            raise ValueError("Currently only 'euclidean' metric is supported.")

        if x.shape[0] != self.n_features:
            raise ValueError(
                "Input sample has incorrect number of features."
            )

        distances = np.empty(self.n_samples)

        for i in range(self.n_samples):
            diff = x - self.X_train[i]
            distances[i] = np.sqrt(np.sum(diff ** 2))

        return distances



    def _get_k_neighbors(self, distances):
        """
        Get indices of the k nearest neighbors.

        Parameters
        ----------
        distances : ndarray of shape (n_train,)

        Returns
        -------
        indices : ndarray of shape (k,)
            Indices of the nearest neighbors.
        """
        if self.n_neighbors > len(distances):
            raise ValueError(
                "n_neighbors cannot be greater than number of training samples."
            )

        # Sort distances and return first k indices
        sorted_indices = np.argsort(distances)
        return sorted_indices[:self.n_neighbors]


    def _vote(self, neighbor_labels, neighbor_distances=None):
        """
        Perform majority voting among neighbor labels.

        Parameters
        ----------
        neighbor_labels : ndarray of shape (k,)
        neighbor_distances : ndarray of shape (k,), optional
            Distances corresponding to neighbor_labels. Used if weights="distance".

        Returns
        -------
        label : int or str
            Winning class label.
        """
        if self.weights == "uniform":
            # Count occurrences of each class
            classes, counts = np.unique(neighbor_labels, return_counts=True)

            # Select class with highest count
            # np.unique sorts classes, so tie-breaking is deterministic
            winner = classes[np.argmax(counts)]

            return winner

        elif self.weights == "distance":
            raise NotImplementedError("Distance weighting not implemented yet.")

        else:
            raise ValueError("Invalid weights parameter.")


    # ============================================================
    # Utility Methods
    # ============================================================

    def _check_is_fitted(self):
        """
        Check if the model has been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("This MyKNNClassifier instance is not fitted yet.")
