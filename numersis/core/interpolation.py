"""
Interpolation methods for approximating functions through known data points.
"""

from typing import Callable
import numpy as np
from .base import Interpolator


class LagrangePolynomial(Interpolator):
    """Lagrange polynomial interpolation."""
    
    def evaluate(self, x: float) -> float:
        """
        Evaluate Lagrange polynomial at x.
        
        Args:
            x: Point to evaluate at.
            
        Returns:
            Interpolated value.
        """
        result = 0.0
        n = len(self.nodes)
        
        for i in range(n):
            # Compute Lagrange basis polynomial L_i(x)
            L = 1.0
            for j in range(n):
                if i != j:
                    L *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
            
            result += self.values[i] * L
        
        return result
    
    def solve(self, x):
        """Alias for evaluate to match interface."""
        return self.evaluate(x)


class NewtonDividedDifference(Interpolator):
    """Newton's divided difference formula for interpolation."""
    
    def __init__(self, nodes: np.ndarray, values: np.ndarray, **kwargs):
        """
        Args:
            nodes: X-coordinates of data points.
            values: Y-coordinates of data points.
        """
        super().__init__(nodes, values, **kwargs)
        self._compute_divided_differences()
    
    def _compute_divided_differences(self):
        """Compute divided difference table."""
        n = len(self.nodes)
        # Create 2D array for divided differences
        self.dd_table = np.zeros((n, n))
        self.dd_table[:, 0] = self.values.copy()
        
        for j in range(1, n):
            for i in range(n - j):
                self.dd_table[i, j] = (
                    (self.dd_table[i + 1, j - 1] - self.dd_table[i, j - 1]) /
                    (self.nodes[i + j] - self.nodes[i])
                )
    
    def evaluate(self, x: float) -> float:
        """
        Evaluate Newton polynomial at x using divided differences.
        
        Args:
            x: Point to evaluate at.
            
        Returns:
            Interpolated value.
        """
        n = len(self.nodes)
        result = self.dd_table[0, 0]
        
        # Product term for Newton polynomial
        product = 1.0
        for i in range(1, n):
            product *= (x - self.nodes[i - 1])
            result += self.dd_table[0, i] * product
        
        return result
    
    def solve(self, x):
        """Alias for evaluate to match interface."""
        return self.evaluate(x)


class ClampedCubicSpline(Interpolator):
    """Clamped cubic spline interpolation."""
    
    def __init__(self, nodes: np.ndarray, values: np.ndarray, 
                 first_deriv_start: float = None, first_deriv_end: float = None, **kwargs):
        """
        Args:
            nodes: X-coordinates of data points.
            values: Y-coordinates of data points.
            first_deriv_start: Derivative at first point (None for natural spline).
            first_deriv_end: Derivative at last point (None for natural spline).
        """
        super().__init__(nodes, values, **kwargs)
        self.first_deriv_start = first_deriv_start
        self.first_deriv_end = first_deriv_end
        self._compute_spline()
    
    def _compute_spline(self):
        """Compute cubic spline coefficients."""
        n = len(self.nodes)
        h = np.diff(self.nodes)
        
        # Set up system of equations for second derivatives
        alpha = np.zeros(n)
        
        if self.first_deriv_start is not None:
            alpha[0] = 3 * (self.values[1] - self.values[0]) / h[0] - 3 * self.first_deriv_start
        
        for i in range(1, n - 1):
            alpha[i] = (3 / h[i]) * (self.values[i + 1] - self.values[i]) - \
                       (3 / h[i - 1]) * (self.values[i] - self.values[i - 1])
        
        if self.first_deriv_end is not None:
            alpha[n - 1] = 3 * self.first_deriv_end - 3 * (self.values[n - 1] - self.values[n - 2]) / h[n - 2]
        
        # Solve tridiagonal system
        l = np.ones(n)
        mu = np.zeros(n - 1)
        z = np.zeros(n)
        
        l[0] = 2 * h[0] if self.first_deriv_start is not None else 1
        mu[0] = h[0] / l[0] if self.first_deriv_start is not None else h[0]
        z[0] = alpha[0] / l[0]
        
        for i in range(1, n - 1):
            l[i] = 2 * (self.nodes[i + 1] - self.nodes[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        
        l[n - 1] = 2 * h[n - 2] if self.first_deriv_end is not None else 1
        z[n - 1] = (alpha[n - 1] - h[n - 2] * z[n - 2]) / l[n - 1]
        
        # Back substitution
        self.c = np.zeros(n)
        self.c[n - 1] = z[n - 1]
        
        for j in range(n - 2, -1, -1):
            self.c[j] = z[j] - mu[j] * self.c[j + 1]
        
        # Compute b and d coefficients
        self.b = np.zeros(n - 1)
        self.d = np.zeros(n - 1)
        
        for j in range(n - 1):
            self.b[j] = (self.values[j + 1] - self.values[j]) / h[j] - h[j] * (2 * self.c[j] + self.c[j + 1]) / 6
            self.d[j] = (self.c[j + 1] - self.c[j]) / (6 * h[j])
    
    def evaluate(self, x: float) -> float:
        """
        Evaluate cubic spline at x.
        
        Args:
            x: Point to evaluate at.
            
        Returns:
            Interpolated value.
        """
        # Find the interval containing x
        if x <= self.nodes[0]:
            j = 0
        elif x >= self.nodes[-1]:
            j = len(self.nodes) - 2
        else:
            j = np.searchsorted(self.nodes, x) - 1
        
        dx = x - self.nodes[j]
        return (self.values[j] + self.b[j] * dx + 
                self.c[j] * dx ** 2 + self.d[j] * dx ** 3)
    
    def solve(self, x):
        """Alias for evaluate to match interface."""
        return self.evaluate(x)
