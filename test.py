import numpy as np
from sklearn.preprocessing import normalize

# Example data (features)
x_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=float)
x_test = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)

# Normalize the data
# 'axis=1' normalizes each sample independently
x_train = normalize(x_train, axis=1)
x_test_normalized = normalize(x_train, axis=1)

print("Original x_train:\n", x_train)
print("Normalized x_train:\n", x_test)

print("Original x_test:\n", x_test)
print("Normalized x_test:\n", x_test_normalized)
