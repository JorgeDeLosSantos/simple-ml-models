from typing import Union
import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression (closed-form solution).

    This implementation follows a didactic approach using the normal equation:

        beta = (X^T X)^(-1) X^T y

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        If False, no intercept will be used in calculations.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    n_features_in_ : int
        Number of features seen during fit.

    _is_fitted : bool
        Internal flag indicating whether the model has been fitted.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_features_in_: int | None = None
        self._is_fitted: bool = False

    def _validate_inputs(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert input data to numpy arrays.
        """

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array or pandas DataFrame.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array or pandas Series.")

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,).")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        return X.astype(float), y.astype(float)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit linear model using the Moore-Penrose pseudo-inverse.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X, y = self._validate_inputs(X, y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.hstack((ones, X))

        # Moore-Penrose pseudo-inverse solution
        X_pinv = np.linalg.pinv(X)
        beta = X_pinv @ y

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self._is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """

        if not self._is_fitted:
            raise RuntimeError("You must fit the model before calling predict().")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array or pandas DataFrame.")

        if X.ndim != 2:
            raise ValueError("X must be 2D.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with {self.n_features_in_} features."
            )

        X = X.astype(float)

        y_pred = X @ self.coef_ + self.intercept_

        return y_pred

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        R^2 = 1 - (SS_res / SS_tot)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            R^2 of self.predict(X) w.r.t. y.
        """

        if not self._is_fitted:
            raise RuntimeError("You must fit the model before calling score().")

        X, y = self._validate_inputs(X, y)

        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Edge case: constant target
        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot
