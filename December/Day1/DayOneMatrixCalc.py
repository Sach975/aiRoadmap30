import numpy as np

array1 = np.array([[1, 2],
                   [3, 4]])

# The identity matrix (I)
print(f"Initial Array (I):\n{array1}")
print(f"Shape: {array1.shape}\n") # (2, 2)

### 1. Vector/Matrix Addition (Element-wise)
print("#--- Vector/Matrix Addition ---")
# I + I = [[1+1, 0+0], [0+0, 1+1]]
addition_result = array1 + array1
print(addition_result) # Should be [[2, 0], [0, 2]]
print("\n")

### 2. Scalar Multiplication (Element-wise)
print("#--- Scalar Multiplication ---")
# 2 * I = [[2*1, 2*0], [2*0, 2*1]]
scalar_result = 2 * array1
print(scalar_result) # Should be [[2, 0], [0, 2]]
print("\n")

### 3. Element-wise Multiplication (The operation you wrote)
print("#--- Element-wise Multiplication (A * B) ---")
# I * I = [[1*1, 0*0], [0*0, 1*1]]
elementwise_result = array1 * array1
print(elementwise_result) # Should be [[1, 0], [0, 1]] (I)
print("\n")

### 4. Matrix Multiplication (The correct operation for A x B)
print("#--- Matrix Multiplication (A @ B or np.matmul(A, B)) ---")
# I @ I = I (Multiplying the identity matrix by itself gives the identity matrix)
matrix_multiplication_result = array1 @ array1
print(matrix_multiplication_result) # Should be [[1, 0], [0, 1]] (I)
print("\n")

### 5. Dot Product (Correctly implemented)
print("#--- Dot Product (Equivalent to Matrix Multiplication for 2D arrays) ---")
# For 2D arrays, np.dot() is synonymous with matrix multiplication.
dot_product_result_A = array1 @ array1       # Pythonic
dot_product_result_B = np.dot(array1, array1) # Dedicated function
print(f"Using @: \n{dot_product_result_A}")
print(f"Using np.dot():\n{dot_product_result_B}")