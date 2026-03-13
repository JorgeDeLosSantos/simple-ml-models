import numpy as np


class LinearSVC:
    
    def __init__(
        self,
        C=1.0,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-4,
        fit_intercept=True,
        verbose=False
    ):
        """
        Linear Support Vector Classifier (from scratch)

        Parameters
        ----------
        C : float
            Regularization strength.
        learning_rate : float
            Step size for gradient descent.
        max_iter : int
            Maximum number of training iterations.
        tol : float
            Convergence tolerance.
        fit_intercept : bool
            Whether to include a bias term.
        verbose : bool
            Whether to print training progress.
        """

        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        # Attributes created after fitting
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.loss_history_ = []

    def _initialize_parameters(self, n_features):
        """
        Initialize model parameters.
        """

        self.coef_ = np.zeros(n_features)

        if self.fit_intercept:
            self.intercept_ = 0.0
        else:
            self.intercept_ = 0.0

    def decision_function(self, X):
        """
        Compute the decision function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Distance of each sample to the decision hyperplane.
        """

        X = np.asarray(X)

        scores = X @ self.coef_

        if self.fit_intercept:
            scores += self.intercept_

        return scores
    
    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray
        """

        scores = self.decision_function(X)

        y_pred = np.sign(scores)

        return y_pred
    
    
    def _compute_loss(self, X, y):
        """
        Compute SVM objective function.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)

        Returns
        -------
        loss : float
        """

        n_samples = X.shape[0]

        # scores
        scores = self.decision_function(X)

        # margins
        margins = 1 - y * scores

        # hinge loss
        hinge_loss = np.maximum(0, margins)

        # regularization term
        reg = 0.5 * np.dot(self.coef_, self.coef_)

        # total loss
        loss = reg + self.C * np.sum(hinge_loss)

        return loss
    
    def fit(self, X, y):
        """
        Train the Linear SVM using Batch Gradient Descent

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (must be -1 or 1)

        Returns
        -------
        self : object
            Returns self.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Validate that y contains only -1 and 1
        unique_labels = np.unique(y)
        if not np.array_equal(np.sort(unique_labels), [-1, 1]):
            raise ValueError("y must contain only -1 and 1 labels")

        n_samples, n_features = X.shape

        # initialize parameters
        self._initialize_parameters(n_features)

        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # compute scores
            scores = self.decision_function(X)

            # margins
            margins = y * scores

            # mask for hinge loss
            mask = margins < 1

            # gradient w
            grad_w = self.coef_ - self.C * (X[mask].T @ y[mask])

            # gradient b
            grad_b = -self.C * np.sum(y[mask]) if self.fit_intercept else 0

            # update parameters
            self.coef_ -= self.learning_rate * grad_w

            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_b

            # compute loss for convergence check
            current_loss = self._compute_loss(X, y)
            self.loss_history_.append(current_loss)

            # check convergence
            if abs(prev_loss - current_loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                self.n_iter_ = iteration + 1
                break

            prev_loss = current_loss

            # verbose output
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Loss: {current_loss:.6f}")

        else:
            # didn't converge
            self.n_iter_ = self.max_iter
            if self.verbose:
                print(f"Maximum iterations ({self.max_iter}) reached")

        return self
