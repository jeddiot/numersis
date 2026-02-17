"""
Examples demonstrating linear algebra and matrix factorization in Numersis.
"""

import numpy as np
from numersis import (
    LUFactorization, CholeskyFactorization, LDLtFactorization,
    GaussElimination, JacobiIterationMethod, GaussSeidelMethod,
    LUSolver, CholeskySolver
)


def example_lu_factorization():
    """Example: LU factorization."""
    print("=" * 60)
    print("LU FACTORIZATION EXAMPLE")
    print("=" * 60)
    
    A = np.array([
        [1, 2, -1],
        [2, 4, 0],
        [0, 1, -1]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    factorizer = LUFactorization()
    factorizer.factorize(A)
    
    print("\nP (permutation):")
    print(factorizer.P)
    print("\nL (lower triangular):")
    print(factorizer.L)
    print("\nU (upper triangular):")
    print(factorizer.U)
    
    # Verify: PA = LU
    print("\nVerification (P*A - L*U):")
    print(np.allclose(factorizer.P @ A, factorizer.L @ factorizer.U))
    print()


def example_gauss_elimination():
    """Example: Gauss elimination."""
    print("=" * 60)
    print("GAUSS ELIMINATION EXAMPLE")
    print("=" * 60)
    
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ], dtype=float)
    
    b = np.array([6, 25, -11, 15], dtype=float)
    
    print("System Ax = b:")
    print("A =")
    print(A)
    print("\nb =")
    print(b)
    
    solver = GaussElimination()
    x = solver.solve(A, b)
    
    print("\nSolution x =")
    print(x)
    
    # Verify
    residual = np.linalg.norm(A @ x - b)
    print(f"\nResidual ||Ax - b|| = {residual:.2e}")
    print()


def example_jacobi_iteration():
    """Example: Jacobi iteration method."""
    print("=" * 60)
    print("JACOBI ITERATION METHOD EXAMPLE")
    print("=" * 60)
    
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ], dtype=float)
    
    b = np.array([6, 25, -11, 15], dtype=float)
    
    print("System Ax = b:")
    print("A =")
    print(A)
    print("\nb =")
    print(b)
    
    solver = JacobiIterationMethod(tolerance=1e-6)
    x = solver.solve(A, b, max_iters=100)
    
    print("\nSolution x =")
    print(x)
    print(f"Iterations: {solver.iterations}")
    
    residual = np.linalg.norm(A @ x - b)
    print(f"Residual ||Ax - b|| = {residual:.2e}")
    print()


def example_gauss_seidel():
    """Example: Gauss-Seidel method."""
    print("=" * 60)
    print("GAUSS-SEIDEL METHOD EXAMPLE")
    print("=" * 60)
    
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ], dtype=float)
    
    b = np.array([6, 25, -11, 15], dtype=float)
    
    print("System Ax = b:")
    print("A =")
    print(A)
    print("\nb =")
    print(b)
    
    solver = GaussSeidelMethod(tolerance=1e-6)
    x = solver.solve(A, b, max_iters=100)
    
    print("\nSolution x =")
    print(x)
    print(f"Iterations: {solver.iterations}")
    
    residual = np.linalg.norm(A @ x - b)
    print(f"Residual ||Ax - b|| = {residual:.2e}")
    print()


def example_lu_solver():
    """Example: Solving using LU factorization."""
    print("=" * 60)
    print("LU SOLVER EXAMPLE")
    print("=" * 60)
    
    A = np.array([
        [4, 3],
        [6, 3]
    ], dtype=float)
    
    b = np.array([10, 12], dtype=float)
    
    print("System Ax = b:")
    print("A =")
    print(A)
    print("\nb =")
    print(b)
    
    solver = LUSolver()
    x = solver.solve(A, b)
    
    print("\nSolution x =")
    print(x)
    
    residual = np.linalg.norm(A @ x - b)
    print(f"Residual ||Ax - b|| = {residual:.2e}")
    print()


def example_cholesky():
    """Example: Cholesky factorization for SPD matrices."""
    print("=" * 60)
    print("CHOLESKY FACTORIZATION EXAMPLE")
    print("=" * 60)
    
    # Create a symmetric positive-definite matrix
    A = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6]
    ], dtype=float)
    
    print("Symmetric positive-definite matrix A:")
    print(A)
    
    factorizer = CholeskyFactorization()
    factorizer.factorize(A)
    
    print("\nCholesky L (lower triangular):")
    print(factorizer.L)
    
    # Verify: A = LL^T
    A_reconstructed = factorizer.L @ factorizer.L.T
    print("\nVerification (L*L^T = A):")
    print(np.allclose(A, A_reconstructed))
    
    # Use it to solve a system
    b = np.array([1, 2, 3], dtype=float)
    solver = CholeskySolver()
    x = solver.solve(A, b)
    
    print("\nSolving Ax = b where b = [1, 2, 3]:")
    print("Solution x =")
    print(x)
    print(f"Residual ||Ax - b|| = {np.linalg.norm(A @ x - b):.2e}")
    print()


if __name__ == "__main__":
    example_lu_factorization()
    example_gauss_elimination()
    example_jacobi_iteration()
    example_gauss_seidel()
    example_lu_solver()
    example_cholesky()
