import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Array + 2:", arr + 2)
print("Array * 2:", arr * 2)
print("Matrix Multiplication:", np.dot(arr, arr.T))  # 2x2 square matrix
