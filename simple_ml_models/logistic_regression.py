from typing import Optional, Union
import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


class LogisticRegression:
    """
    Logistic Regression classifier trained using batch Gradient Descent.

    This implementation is numerically stable and computes the loss
    directly from logits using logaddexp to avoid overflow and log(0) issues.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used during gradient descent.

    n_iterations : int, default=1000
        Number of gradient descent iterations.

    fit_intercept : bool, default=True
        Whether to include an intercept term.

    random_state : Optional[int], default=None
        Seed for weight initialization.

    verbose : bool, default=False
        If True, prints loss during training.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.loss_history_: list = []

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters.
        """
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(loc=0.0, scale=0.01, size=n_features)

        self.intercept_ = 0.0 if self.fit_intercept else None

    def _linear_combination(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the linear combination (logits).

        z = Xw + b
        """
        z = X @ self.coef_
        if self.fit_intercept:
            z += self.intercept_
        return z

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.
        """
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z)),
        )

    def _compute_loss(self, z: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss using logits.

        Loss:
            L = mean( log(1 + exp(z)) - y*z )

        Implemented using logaddexp for numerical stability.
        """
        loss = np.mean(np.logaddexp(0, z) - y * z)
        return loss

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n_samples, n_features = X.shape

        self._initialize_parameters(n_features)
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            # Forward pass
            z = self._linear_combination(X)
            p = self._sigmoid(z)

            # Compute gradients
            error = p - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # Update parameters
            self.coef_ -= self.learning_rate * dw
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * db

            # Store loss
            loss = self._compute_loss(z, y)
            self.loss_history_.append(loss)

            if self.verbose:
                print(f"Loss: {loss:.6f}")

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class probabilities.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, 2)
            Column 0 contains P(y=0)
            Column 1 contains P(y=1)
        """
        X = np.asarray(X)
        z = self._linear_combination(X)
        p = self._sigmoid(z)
        return np.column_stack((1 - p, p))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict binary class labels.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Compute accuracy of the classifier.
        """
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)