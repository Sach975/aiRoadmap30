import numpy as np

# Define a 3x3 matrix
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

print(f"--- Original Matrix A ---\n{A}\n")

# 1. Determinant
det_A = np.linalg.det(A)
print(f"1. Determinant (det(A)): {det_A:.2f}")

# 2. Transpose
A_T = A.T
print(f"\n2. Transpose (A.T):\n{A_T}")

# 3. Rank
rank_A = np.linalg.matrix_rank(A)
print(f"\n3. Rank (matrix_rank(A)): {rank_A}")

# 4. Trace
trace_A = np.trace(A)
# Trace = 1 + 1 + 0 = 2
print(f"\n4. Trace (trace(A)): {trace_A}")

# 5. Matrix Inverse
try:
    A_inv = np.linalg.inv(A)
    print(f"\n5. Matrix Inverse (inv(A)): \n{A_inv}")
    # Verification: A @ A_inv should be close to the Identity Matrix (I)
    print(f"\nVerification (A @ inv(A)):\n{A @ A_inv}") 

except np.linalg.LinAlgError as e:
    print(f"\n5. Matrix Inverse Error: {e}")
    print("The matrix is singular (determinant is zero or near zero) and cannot be inverted.")



'''
# This is the chatGPT script 


import numpy as np

def _ensure_same_shape(A, B):
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

def is_square(A):
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def is_symmetric(A, tol=1e-9):
    if not is_square(A): return False
    return np.allclose(A, A.T, atol=tol)

def mat_add(A, B):
    _ensure_same_shape(A, B)
    return A + B

def mat_scalar_mul(c, A):
    return c * A

def mat_mul(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes for @: {A.shape} @ {B.shape}")
    return A @ B

def dot(u, v):
    u, v = np.asarray(u), np.asarray(v)
    if u.ndim != 1 or v.ndim != 1 or u.shape[0] != v.shape[0]:
        raise ValueError("dot expects 1D vectors of equal length")
    return float(u @ v)

def mat_transpose(A):
    return A.T

def mat_det(A):
    if not is_square(A): raise ValueError("determinant requires a square matrix")
    return float(np.linalg.det(A))

def mat_rank(A):
    return int(np.linalg.matrix_rank(A))

def mat_trace(A):
    if not is_square(A): raise ValueError("trace requires a square matrix")
    return float(np.trace(A))

def mat_inverse(A):
    if not is_square(A): raise ValueError("inverse requires a square matrix")
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("matrix is singular (non-invertible)")

if __name__ == "__main__":
    A = np.array([[1., 2.], [3., 4.]])
    B = np.array([[5., 6.], [7., 8.]])
    print("A + B:\n", mat_add(A, B))
    print("A @ B:\n", mat_mul(A, B))
    print("det(A):", mat_det(A))
    print("rank(B):", mat_rank(B))
    print("trace(A):", mat_trace(A))
    print("A^{-1}:\n", mat_inverse(A))

'''