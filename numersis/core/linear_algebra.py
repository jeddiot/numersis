"""
Linear systems and matrix factorization methods.
"""

from typing import Tuple
import numpy as np
from .base import LinearSolver, MatrixFactorization


# ============================================================================
# MATRIX FACTORIZATION METHODS
# ============================================================================

class LUFactorization(MatrixFactorization):
    """LU factorization of a matrix (PA = LU)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.P = None
        self.L = None
        self.U = None
    
    def factorize(self, A: np.ndarray):
        """
        Compute LU factorization of A.
        
        Args:
            A: Matrix to factorize.
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        
        # Use scipy's lu for stability
        try:
            from scipy.linalg import lu
            self.P, self.L, self.U = lu(A)
        except ImportError:
            # Fallback: simple LU without pivoting
            L = np.eye(n)
            U = A.copy()
            
            for k in range(n - 1):
                for i in range(k + 1, n):
                    factor = U[i, k] / U[k, k]
                    L[i, k] = factor
                    U[i, k:] -= factor * U[k, k:]
            
            self.P = np.eye(n)
            self.L = L
            self.U = U


class CholeskyFactorization(MatrixFactorization):
    """Cholesky factorization for symmetric positive-definite matrices."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.L = None
    
    def factorize(self, A: np.ndarray):
        """
        Compute Cholesky factorization A = LL^T.
        
        Args:
            A: Symmetric positive-definite matrix.
            
        Raises:
            ValueError: If matrix is not positive-definite.
        """
        A = np.array(A, dtype=complex)
        
        try:
            self.L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix must be symmetric positive-definite")


class LDLtFactorization(MatrixFactorization):
    """LDL^T factorization for symmetric matrices."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.L = None
        self.D = None
        self.perm = None
    
    def factorize(self, A: np.ndarray):
        """
        Compute LDL^T factorization.
        
        Args:
            A: Symmetric matrix.
        """
        A = np.array(A, dtype=float)
        
        try:
            from scipy.linalg import ldl
            self.L, self.D, self.perm = ldl(A)
        except ImportError:
            # Fallback: simple implementation
            n = A.shape[0]
            L = np.eye(n)
            D = np.zeros((n, n))
            
            for k in range(n):
                D[k, k] = A[k, k]
                for i in range(k):
                    D[k, k] -= L[k, i] ** 2 * D[i, i]
                
                for i in range(k + 1, n):
                    L[i, k] = A[i, k]
                    for j in range(k):
                        L[i, k] -= L[i, j] * D[j, j] * L[k, j]
                    L[i, k] /= D[k, k]
            
            self.L = L
            self.D = D
            self.perm = np.arange(n)


# ============================================================================
# LINEAR SYSTEM SOLVERS
# ============================================================================

class GaussElimination(LinearSolver):
    """Gaussian elimination with back substitution."""
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b using Gaussian elimination.
        
        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            
        Returns:
            Solution vector x.
        """
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        
        # Forward elimination
        for k in range(n - 1):
            # Partial pivoting
            max_idx = k + np.argmax(np.abs(A[k:, k]))
            A[[k, max_idx]] = A[[max_idx, k]]
            b[[k, max_idx]] = b[[max_idx, k]]
            
            # Eliminate column
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = b[i]
            for j in range(i + 1, n):
                x[i] -= A[i, j] * x[j]
            x[i] /= A[i, i]
        
        return x


class JacobiIterationMethod(LinearSolver):
    """Jacobi iterative method for solving linear systems."""
    
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: np.ndarray = None, max_iters: int = 100) -> np.ndarray:
        """
        Solve Ax = b using Jacobi iteration.
        
        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            x0: Initial guess (default: zero vector).
            max_iters: Maximum iterations.
            
        Returns:
            Solution vector x.
        """
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)
        
        D = np.diag(A)
        R = A - np.diag(D)
        
        for _ in range(max_iters):
            x_new = (b - np.dot(R, x)) / D
            
            # Check convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                break
            
            x = x_new
            self.iterations += 1
        
        return x


class GaussSeidelMethod(LinearSolver):
    """Gauss-Seidel iterative method for solving linear systems."""
    
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: np.ndarray = None, max_iters: int = 100) -> np.ndarray:
        """
        Solve Ax = b using Gauss-Seidel iteration.
        
        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            x0: Initial guess (default: zero vector).
            max_iters: Maximum iterations.
            
        Returns:
            Solution vector x.
        """
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        n = len(b)
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.array(x0, dtype=float)
        
        for iteration in range(max_iters):
            x_old = x.copy()
            
            for i in range(n):
                s = 0.0
                for j in range(n):
                    if i != j:
                        s += A[i, j] * x[j]
                x[i] = (b[i] - s) / A[i, i]
            
            # Check convergence
            if np.linalg.norm(x - x_old) < self.tolerance:
                break
            
            self.iterations += 1
        
        return x


class LUSolver(LinearSolver):
    """Solver using LU factorization."""
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b using LU factorization.
        
        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            
        Returns:
            Solution vector x.
        """
        lu_fact = LUFactorization()
        lu_fact.factorize(A)
        
        # Solve Py = b, Lz = y, Ux = z
        y = np.dot(lu_fact.P.T, b)
        
        # Forward substitution: Lz = y
        z = np.linalg.solve(lu_fact.L, y)
        
        # Back substitution: Ux = z
        x = np.linalg.solve(lu_fact.U, z)
        
        return x


class CholeskySolver(LinearSolver):
    """Solver using Cholesky factorization (for SPD matrices)."""
    
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b using Cholesky factorization.
        Matrix A must be symmetric positive-definite.
        
        Args:
            A: Symmetric positive-definite matrix.
            b: Right-hand side vector.
            
        Returns:
            Solution vector x.
        """
        chol = CholeskyFactorization()
        chol.factorize(A)
        
        # Solve Ly = b, L^T x = y
        y = np.linalg.solve(chol.L, b)
        x = np.linalg.solve(chol.L.T, y)
        
        return x
