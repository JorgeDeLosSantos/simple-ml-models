"""
Example demonstrating the LinearSVC implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from simple_ml_models.svm import LinearSVC


def create_linearly_separable_data(n_samples=100, noise=0.1, random_state=42):
    """Create a simple linearly separable 2D dataset"""
    np.random.seed(random_state)
    
    # Generate data
    X = np.random.randn(n_samples, 2)
    
    # Create linearly separable labels
    # Decision boundary: x1 + x2 = 0
    y = np.sign(X[:, 0] + X[:, 1] + np.random.normal(0, noise, n_samples))
    
    return X, y


def plot_decision_boundary(X, y, clf, title="SVM Decision Boundary"):
    """Plot the decision boundary of the trained SVM"""
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class +1', alpha=0.7)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='s', label='Class -1', alpha=0.7)
    
    # Create a mesh to plot the decision boundary
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='-', linewidths=2)
    
    # Plot margins
    if hasattr(clf, 'coef_') and hasattr(clf, 'intercept_'):
        # Calculate margin lines
        w = clf.coef_
        b = clf.intercept_
        
        # Margin lines: w·x + b = ±1
        # x2 = (-b ± 1 - w1*x1) / w2
        x1_range = np.linspace(x_min, x_max, 100)
        if abs(w[1]) > 1e-10:  # Avoid division by zero
            x2_margin_plus = (-b + 1 - w[0] * x1_range) / w[1]
            x2_margin_minus = (-b - 1 - w[0] * x1_range) / w[1]
            
            plt.plot(x1_range, x2_margin_plus, 'k--', alpha=0.5, label='Margin +1')
            plt.plot(x1_range, x2_margin_minus, 'k--', alpha=0.5, label='Margin -1')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def main():
    print("LinearSVC Implementation Demo")
    print("=" * 40)
    
    # Create dataset
    print("Creating linearly separable dataset...")
    X, y = create_linearly_separable_data(n_samples=200, noise=0.2)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount((y + 1).astype(int))}")
    
    # Train SVM with different regularization strengths
    print("\nTraining SVM with different C values...")
    
    # High regularization (C = 0.1)
    print("\n1. High regularization (C = 0.1):")
    svm_high_reg = LinearSVC(C=0.1, max_iter=1000, tol=1e-4, verbose=True)
    svm_high_reg.fit(X, y)
    print(f"   Converged in {svm_high_reg.n_iter_} iterations")
    print(f"   Final loss: {svm_high_reg.loss_history_[-1]:.6f}")
    print(f"   Coefficients: {svm_high_reg.coef_}")
    print(f"   Intercept: {svm_high_reg.intercept_}")
    
    # Low regularization (C = 10.0)
    print("\n2. Low regularization (C = 10.0):")
    svm_low_reg = LinearSVC(C=10.0, max_iter=1000, tol=1e-4, verbose=True)
    svm_low_reg.fit(X, y)
    print(f"   Converged in {svm_low_reg.n_iter_} iterations")
    print(f"   Final loss: {svm_low_reg.loss_history_[-1]:.6f}")
    print(f"   Coefficients: {svm_low_reg.coef_}")
    print(f"   Intercept: {svm_low_reg.intercept_}")
    
    # Compare with and without intercept
    print("\n3. Without intercept:")
    svm_no_intercept = LinearSVC(C=1.0, fit_intercept=False, max_iter=1000, tol=1e-4)
    svm_no_intercept.fit(X, y)
    print(f"   Converged in {svm_no_intercept.n_iter_} iterations")
    print(f"   Coefficients: {svm_no_intercept.coef_}")
    print(f"   Intercept: {svm_no_intercept.intercept_}")
    
    # Plot results
    print("\nGenerating plots...")
    
    plot_decision_boundary(X, y, svm_high_reg, "SVM with High Regularization (C=0.1)")
    plot_decision_boundary(X, y, svm_low_reg, "SVM with Low Regularization (C=10.0)")
    plot_decision_boundary(X, y, svm_no_intercept, "SVM without Intercept")
    
    # Plot loss history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(svm_high_reg.loss_history_, 'b-', linewidth=2)
    plt.title('Loss History (C=0.1)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(svm_low_reg.loss_history_, 'r-', linewidth=2)
    plt.title('Loss History (C=10.0)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(svm_no_intercept.loss_history_, 'g-', linewidth=2)
    plt.title('Loss History (No Intercept)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test predictions
    print("\nTesting predictions...")
    test_points = np.array([[0, 0], [1, 1], [-1, -1], [2, -1]])
    for i, point in enumerate(test_points):
        pred_high = svm_high_reg.predict([point])[0]
        pred_low = svm_low_reg.predict([point])[0]
        print(f"Point {point}: High reg prediction = {pred_high}, Low reg prediction = {pred_low}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()