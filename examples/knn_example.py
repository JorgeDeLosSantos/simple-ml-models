import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from models.knn import KNNClassifier

# Dataset de entrenamiento
X_train = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [8, 8],
    [8, 9],
    [9, 8],
    [9, 9]
])

y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Dataset de prueba
X_test = np.array([
    [1.5, 1.5],
    [8.5, 8.5],
    [5, 5]
])

model = KNNClassifier(n_neighbors=5)
model.fit(X_train, y_train)

sk_model = KNeighborsClassifier(n_neighbors=5)
sk_model.fit(X_train, y_train)

y_pred_custom = model.predict(X_test)
print("Custom KNN predictions:", y_pred_custom)

y_pred_sklearn = sk_model.predict(X_test)
print("Sklearn KNN predictions:", y_pred_sklearn)

print("Are predictions equal?", np.array_equal(y_pred_custom, y_pred_sklearn))


# Crear una malla de puntos
h = 0.1  # tamaño del paso
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Predecir para cada punto de la malla
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Graficar frontera
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Graficar puntos de entrenamiento
plt.scatter(X_train[:, 0], X_train[:, 1], 
            c=y_train, cmap='coolwarm', edgecolor='k', s=100)

# Graficar puntos de prueba
plt.scatter(X_test[:, 0], X_test[:, 1], 
            c='green', marker='X', s=200, label='Test Points')

plt.title('KNN Decision Boundary')
plt.legend()
plt.show()
