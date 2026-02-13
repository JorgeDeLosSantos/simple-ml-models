import numpy as np
import matplotlib.pyplot as plt
from models.linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

X = np.linspace(1, 10, 10).reshape(-1, 2)
print(X)
y = X[:, 0] + X[:, 1] + np.random.normal(0, 1, size=X.shape[0])
print(y)

model = LinearRegression()
model_sklearn = SklearnLinearRegression()

model.fit(X, y)
model_sklearn.fit(X, y)

print("Coef:", model.coef_)
print("Intercept:", model.intercept_)

print("Sklearn Coef:", model_sklearn.coef_)
print("Sklearn Intercept:", model_sklearn.intercept_)

y_pred = model.predict(X)
y_pred_sklearn = model_sklearn.predict(X)
print("Predictions:", y_pred)
print("Sklearn Predictions:", y_pred_sklearn)

print("R² Score (Custom):", model.score(X, y))
print("R² Score (Sklearn):", model_sklearn.score(X, y))

plt.scatter(X[:, 0], y, color='blue', label='Data')
plt.plot(X[:, 0], y_pred, color='red', label='Custom Linear Regression')
plt.plot(X[:, 0], y_pred_sklearn, color='green', label='Sklearn Linear Regression', linestyle='dashed')
plt.xlabel('Feature 1') 
plt.ylabel('Target')
plt.legend()
plt.title('Linear Regression Comparison')


plt.show()