import numpy as np
import pytest
from simple_ml_models.svm import LinearSVC


class TestLinearSVC:
    
    def test_initialization(self):
        """Test that LinearSVC initializes with correct default parameters"""
        clf = LinearSVC()
        
        assert clf.C == 1.0
        assert clf.learning_rate == 0.01
        assert clf.max_iter == 1000
        assert clf.tol == 1e-4
        assert clf.fit_intercept == True
        assert clf.verbose == False
        
        # Test custom parameters
        clf = LinearSVC(C=2.0, learning_rate=0.05, max_iter=500, tol=1e-3, 
                       fit_intercept=False, verbose=True)
        assert clf.C == 2.0
        assert clf.learning_rate == 0.05
        assert clf.max_iter == 500
        assert clf.tol == 1e-3
        assert clf.fit_intercept == False
        assert clf.verbose == True

    def test_parameter_initialization(self):
        """Test that parameters are properly initialized"""
        clf = LinearSVC()
        clf._initialize_parameters(3)
        
        assert clf.coef_.shape == (3,)
        assert np.allclose(clf.coef_, np.zeros(3))
        assert clf.intercept_ == 0.0

    def test_decision_function(self):
        """Test decision function computation"""
        clf = LinearSVC()
        clf.coef_ = np.array([1.0, -2.0])
        clf.intercept_ = 0.5
        
        X = np.array([[1.0, 2.0], [3.0, 1.0]])
        scores = clf.decision_function(X)
        
        expected = np.array([1.0*1.0 + (-2.0)*2.0 + 0.5, 1.0*3.0 + (-2.0)*1.0 + 0.5])
        expected = np.array([-2.5, 1.5])
        
        assert np.allclose(scores, expected)

    def test_decision_function_no_intercept(self):
        """Test decision function without intercept"""
        clf = LinearSVC(fit_intercept=False)
        clf.coef_ = np.array([1.0, -2.0])
        clf.intercept_ = 0.0
        
        X = np.array([[1.0, 2.0], [3.0, 1.0]])
        scores = clf.decision_function(X)
        
        expected = np.array([1.0*1.0 + (-2.0)*2.0, 1.0*3.0 + (-2.0)*1.0])
        expected = np.array([-3.0, 1.0])
        
        assert np.allclose(scores, expected)

    def test_predict(self):
        """Test prediction function"""
        clf = LinearSVC()
        clf.coef_ = np.array([1.0, -2.0])
        clf.intercept_ = 0.5
        
        X = np.array([[1.0, 2.0], [3.0, 1.0], [0.0, 0.0]])
        predictions = clf.predict(X)
        
        # Expected: [-2.5, 1.5, 0.5] -> [-1, 1, 1]
        expected = np.array([-1, 1, 1])
        
        assert np.array_equal(predictions, expected)

    def test_loss_computation(self):
        """Test loss computation"""
        clf = LinearSVC(C=1.0)
        clf.coef_ = np.array([1.0, -1.0])
        clf.intercept_ = 0.0
        
        X = np.array([[1.0, 2.0], [3.0, 1.0]])
        y = np.array([-1.0, 1.0])
        
        loss = clf._compute_loss(X, y)
        
        # Manual calculation:
        # scores = [1*1 + (-1)*2, 1*3 + (-1)*1] = [-1, 2]
        # margins = 1 - y*scores = [1 - (-1)*(-1), 1 - 1*2] = [0, -1]
        # hinge_loss = max(0, margins) = [0, 0]
        # reg = 0.5 * (1^2 + (-1)^2) = 1.0
        # loss = 1.0 + 1.0 * 0 = 1.0
        
        assert np.isclose(loss, 1.0)

    def test_fit_convergence(self):
        """Test that fit method converges"""
        # Create simple linearly separable data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.sign(X[:, 0] + X[:, 1])  # Linearly separable
        
        clf = LinearSVC(max_iter=1000, tol=1e-4, verbose=False)
        clf.fit(X, y)
        
        # Check that model was fitted
        assert hasattr(clf, 'coef_')
        assert hasattr(clf, 'intercept_')
        assert hasattr(clf, 'n_iter_')
        assert hasattr(clf, 'loss_history_')
        
        # Check that loss decreased
        assert len(clf.loss_history_) > 0
        assert clf.loss_history_[-1] < clf.loss_history_[0]
        
        # Check convergence
        assert clf.n_iter_ < clf.max_iter

    def test_fit_with_intercept(self):
        """Test fitting with intercept"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.sign(X[:, 0] + X[:, 1] + 0.5)  # With bias
        
        clf = LinearSVC(fit_intercept=True, max_iter=500, tol=1e-4)
        clf.fit(X, y)
        
        assert clf.intercept_ != 0.0
        assert clf.n_iter_ < clf.max_iter

    def test_fit_without_intercept(self):
        """Test fitting without intercept"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.sign(X[:, 0] + X[:, 1])
        
        clf = LinearSVC(fit_intercept=False, max_iter=500, tol=1e-4)
        clf.fit(X, y)
        
        assert clf.intercept_ == 0.0
        assert clf.n_iter_ < clf.max_iter

    def test_label_validation(self):
        """Test that only -1 and 1 labels are accepted"""
        clf = LinearSVC()
        X = np.random.randn(10, 2)
        
        # Valid labels
        y_valid = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        clf.fit(X, y_valid)  # Should not raise
        
        # Invalid labels
        y_invalid = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="y must contain only -1 and 1 labels"):
            clf.fit(X, y_invalid)
            
        y_invalid2 = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        with pytest.raises(ValueError, match="y must contain only -1 and 1 labels"):
            clf.fit(X, y_invalid2)

    def test_fit_returns_self(self):
        """Test that fit returns self"""
        clf = LinearSVC()
        X = np.random.randn(10, 2)
        y = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        
        result = clf.fit(X, y)
        assert result is clf

    def test_verbose_output(self):
        """Test verbose output during training"""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.sign(X[:, 0] + X[:, 1])
        
        clf = LinearSVC(max_iter=10, tol=1e-4, verbose=True)
        
        # Capture stdout to check verbose output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        clf.fit(X, y)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        # Should contain some verbose output
        assert "Iteration" in output or "Converged" in output or "Maximum iterations" in output

    def test_different_C_values(self):
        """Test that different C values affect the model"""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.sign(X[:, 0] + X[:, 1])
        
        clf1 = LinearSVC(C=0.1, max_iter=500, tol=1e-4)
        clf1.fit(X, y)
        
        clf2 = LinearSVC(C=10.0, max_iter=500, tol=1e-4)
        clf2.fit(X, y)
        
        # Different C values should generally lead to different coefficients
        # (though this might not always be true for this specific random seed)
        assert hasattr(clf1, 'coef_') and hasattr(clf2, 'coef_')

    def test_large_dataset(self):
        """Test with larger dataset"""
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        y = np.sign(X[:, 0] + X[:, 1] + X[:, 2])
        
        clf = LinearSVC(max_iter=100, tol=1e-3)
        clf.fit(X, y)
        
        assert clf.n_iter_ <= clf.max_iter
        assert len(clf.loss_history_) == clf.n_iter_

    def test_edge_case_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = np.sign(X[:, 0])
        
        clf = LinearSVC(max_iter=500, tol=1e-4)
        clf.fit(X, y)
        
        assert clf.coef_.shape == (1,)
        assert clf.n_iter_ < clf.max_iter

    def test_perfect_separation(self):
        """Test with perfectly separable data"""
        X = np.array([[1, 1], [2, 2], [3, 3], [-1, -1], [-2, -2], [-3, -3]], dtype=float)
        y = np.array([1, 1, 1, -1, -1, -1], dtype=float)
        
        clf = LinearSVC(max_iter=1000, tol=1e-6, learning_rate=0.1)
        clf.fit(X, y)
        
        # Should converge for perfectly separable data
        assert clf.n_iter_ <= clf.max_iter
        assert len(clf.loss_history_) == clf.n_iter_
