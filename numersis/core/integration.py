"""
Numerical integration methods for computing approximate integrals.
"""

from typing import Callable
import numpy as np
from .base import Integrator


class CompositeSimpsonRule(Integrator):
    """Composite Simpson's rule for numerical integration."""
    
    def solve(self, f: Callable, a: float, b: float, n: int = 100) -> float:
        """
        Approximate integral of f over [a, b] using composite Simpson's rule.
        
        Args:
            f: Function to integrate.
            a: Lower bound.
            b: Upper bound.
            n: Number of subintervals (must be even).
            
        Returns:
            Approximate integral value.
            
        Raises:
            ValueError: If n is not even.
        """
        if n % 2 != 0:
            raise ValueError("n must be even for Simpson's rule")
        
        h = (b - a) / n
        
        # Sum endpoints
        result = f(a) + f(b)
        
        # Sum odd-indexed points (coefficient 4)
        for i in range(1, n, 2):
            result += 4 * f(a + i * h)
        
        # Sum even-indexed points (coefficient 2)
        for i in range(2, n - 1, 2):
            result += 2 * f(a + i * h)
        
        return result * h / 3


class RombergIntegration(Integrator):
    """Romberg integration for high-precision numerical integration."""
    
    def solve(self, f: Callable, a: float, b: float, n: int = 5) -> float:
        """
        Approximate integral using Romberg integration.
        
        Args:
            f: Function to integrate.
            a: Lower bound.
            b: Upper bound.
            n: Number of rows in Romberg table.
            
        Returns:
            Approximate integral value.
        """
        R = np.zeros((n, n))
        h = b - a
        
        # Compute first column (trapezoidal rule)
        for i in range(n):
            R[i, 0] = self._trapezoidal(f, a, b, 2 ** i)
        
        # Compute Richardson extrapolation
        for k in range(1, n):
            for i in range(k, n):
                R[i, k] = R[i, k - 1] + (R[i, k - 1] - R[i - 1, k - 1]) / (4 ** k - 1)
        
        return R[n - 1, n - 1]
    
    def _trapezoidal(self, f: Callable, a: float, b: float, n: int) -> float:
        """Compute trapezoidal rule approximation."""
        h = (b - a) / n
        s = f(a) / 2.0 + f(b) / 2.0
        
        for i in range(1, n):
            s += f(a + i * h)
        
        return s * h


class GaussianQuadrature(Integrator):
    """Gaussian quadrature for numerical integration."""
    
    def solve(self, f: Callable, a: float, b: float, n: int = 5) -> float:
        """
        Approximate integral using Gaussian quadrature.
        
        Args:
            f: Function to integrate.
            a: Lower bound.
            b: Upper bound.
            n: Degree of Legendre polynomial (number of nodes).
            
        Returns:
            Approximate integral value.
        """
        # Get Legendre nodes and weights
        x, w = np.polynomial.legendre.leggauss(n)
        
        # Transform from [-1, 1] to [a, b]
        t = 0.5 * (b - a) * x + 0.5 * (a + b)
        
        # Apply quadrature formula
        h_half = 0.5 * (b - a)
        result = h_half * np.sum(w * np.array([f(ti) for ti in t]))
        
        return result


class AdaptiveQuadrature(Integrator):
    """Adaptive quadrature for automatic step size control."""
    
    def solve(self, f: Callable, a: float, b: float, tol: float = 1e-4) -> float:
        """
        Approximate integral with adaptive refinement.
        
        Args:
            f: Function to integrate.
            a: Lower bound.
            b: Upper bound.
            tol: Tolerance for error estimation.
            
        Returns:
            Approximate integral value.
        """
        return self._adaptive_simpson(f, a, b, tol)
    
    def _adaptive_simpson(self, f: Callable, a: float, b: float, tol: float) -> float:
        """Recursive adaptive Simpson's rule."""
        c = (a + b) / 2
        h = b - a
        
        fa = f(a)
        fb = f(b)
        fc = f(c)
        
        S = h * (fa + 4 * fc + fb) / 6
        
        # Subdivide
        d = (a + c) / 2
        e = (c + b) / 2
        
        fd = f(d)
        fe = f(e)
        
        S_left = h / 12 * (fa + 4 * fd + fc)
        S_right = h / 12 * (fc + 4 * fe + fb)
        S_fine = S_left + S_right
        
        # Check convergence
        if abs(S_fine - S) <= 15 * tol:
            return S_fine + (S_fine - S) / 15
        
        # Recurse
        return (self._adaptive_simpson(f, a, c, tol / 2) + 
                self._adaptive_simpson(f, c, b, tol / 2))


class SimpsonsDoubleIntegral(Integrator):
    """Simpson's rule for double integrals."""
    
    def solve(self, f, a: float, b: float, c_func: Callable, d_func: Callable, 
              m: int = 14, n: int = 14) -> float:
        """
        Approximate double integral of f over a domain.
        
        ∫∫ f(x, y) dy dx
        
        Args:
            f: Function of two variables f(x, y).
            a: Lower bound for x.
            b: Upper bound for x.
            c_func: Function c(x) giving lower y bound.
            d_func: Function d(x) giving upper y bound.
            m: Number of subintervals in y direction (must be even).
            n: Number of subintervals in x direction (must be even).
            
        Returns:
            Approximate double integral value.
        """
        if m % 2 != 0 or n % 2 != 0:
            raise ValueError("m and n must be even")
        
        h = (b - a) / n
        J = 0
        
        for i in range(n):
            x = a + i * h
            c = c_func(x)
            d = d_func(x)
            
            # Apply Simpson's rule in y direction
            H = (d - c) / m
            K_1 = f(x, c) + f(x, d)
            K_2 = 0
            K_3 = 0
            
            for j in range(1, m):
                y = c + j * H
                Q = f(x, y)
                
                if j % 2 == 0:
                    K_2 += Q
                else:
                    K_3 += Q
            
            L = ((K_1 + 2 * K_2 + 4 * K_3) * H) / 3
            
            # Weight for Simpson's rule in x direction
            if i == 0 or i == n:
                J += L
            elif i % 2 == 0:
                J += 2 * L
            else:
                J += 4 * L
        
        return J * h / 3
