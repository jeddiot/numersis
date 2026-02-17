"""
Root-finding methods for solving f(x) = 0.
"""

from typing import Callable, Tuple
import numpy as np
from .base import RootFinder


class BisectionMethod(RootFinder):
    """Bisection method for root finding."""
    
    def solve(self, f: Callable, a: float, b: float) -> Tuple[float, int]:
        """
        Find root of f in [a, b] using bisection method.
        
        Args:
            f: Function to find root of.
            a: Left endpoint.
            b: Right endpoint.
            
        Returns:
            Tuple of (root, iterations).
            
        Raises:
            ValueError: If f(a) and f(b) have the same sign.
        """
        self.reset()
        
        fa = f(a)
        fb = f(b)
        
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        xl, xr = a, b
        
        while abs(xl - xr) >= self.tolerance and self.iterations < self.max_iterations:
            c = (xl + xr) / 2.0
            fc = f(c)
            
            if fa * fc > 0:
                xl = c
                fa = fc
            else:
                xr = c
                fb = fc
            
            self.iterations += 1
        
        return (xl + xr) / 2.0, self.iterations


class FixedPointMethod(RootFinder):
    """Fixed-point iteration method."""
    
    def solve(self, g: Callable, x0: float) -> Tuple[float, int, list]:
        """
        Find fixed point of g (where g(x) = x) using fixed-point iteration.
        
        Args:
            g: Fixed-point function g(x). Should satisfy x = g(x) at fixed point.
            x0: Initial guess.
            
        Returns:
            Tuple of (fixed_point, iterations, history).
        """
        self.reset()
        
        x = x0
        history = [x0]
        
        while self.iterations < self.max_iterations:
            x_new = g(x)
            error = abs(x_new - x)
            
            if error < self.tolerance:
                break
            
            x = x_new
            history.append(x)
            self.iterations += 1
        
        return x, self.iterations, history


class NewtonRaphsonMethod(RootFinder):
    """Newton-Raphson method for root finding."""
    
    def solve(self, f: Callable, df: Callable, x0: float) -> Tuple[float, int]:
        """
        Find root of f using Newton-Raphson method.
        
        Args:
            f: Function to find root of.
            df: Derivative of f.
            x0: Initial guess.
            
        Returns:
            Tuple of (root, iterations).
            
        Raises:
            ValueError: If derivative is zero at any point.
        """
        self.reset()
        
        x = x0
        
        while self.iterations < self.max_iterations:
            fx = f(x)
            dfx = df(x)
            
            if abs(dfx) < 1e-15:
                raise ValueError(f"Derivative too small at x={x}")
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            if error < self.tolerance:
                break
            
            x = x_new
            self.iterations += 1
        
        return x, self.iterations


class FalsePositionMethod(RootFinder):
    """False position (regula falsi) method for root finding."""
    
    def solve(self, f: Callable, a: float, b: float) -> Tuple[float, int]:
        """
        Find root of f in [a, b] using false position method.
        
        Args:
            f: Function to find root of.
            a: Left endpoint.
            b: Right endpoint.
            
        Returns:
            Tuple of (root, iterations).
            
        Raises:
            ValueError: If f(a) and f(b) have the same sign.
        """
        self.reset()
        
        fa = f(a)
        fb = f(b)
        
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        while self.iterations < self.max_iterations:
            # Linear interpolation formula
            c = a - (fa * (b - a)) / (fb - fa)
            fc = f(c)
            
            if abs(fc) < self.tolerance or abs(b - a) < self.tolerance:
                break
            
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
            
            self.iterations += 1
        
        return c, self.iterations


class MullersMethod(RootFinder):
    """Müller's method for root finding (handles complex roots)."""
    
    def solve(self, f: Callable, p0: float, p1: float, p2: float) -> Tuple[float, int]:
        """
        Find root of f using Müller's method.
        
        Args:
            f: Function to find root of.
            p0, p1, p2: Three initial points.
            
        Returns:
            Tuple of (root, iterations).
        """
        self.reset()
        
        MAX_ITERATIONS = 10000
        
        while self.iterations < min(self.max_iterations, MAX_ITERATIONS):
            f0 = f(p0)
            f1 = f(p1)
            f2 = f(p2)
            
            d1 = f0 - f2
            d2 = f1 - f2
            h1 = p0 - p2
            h2 = p1 - p2
            
            a0 = f2
            a1 = (((h1 ** 2 * d2) - (h2 ** 2 * d1)) / (h1 * h2 * (h1 - h2)))
            a2 = (((h2 * d1) - (h1 * d2)) / (h1 * h2 * (h1 - h2)))
            
            discriminant = a1 * a1 - 4 * a0 * a2
            sqrt_disc = abs(discriminant) ** 0.5
            
            x = (-2 * a0) / (a1 + sqrt_disc) if abs(a1 + sqrt_disc) > abs(a1 - sqrt_disc) else (-2 * a0) / (a1 - sqrt_disc)
            res = x + p2
            
            # Check convergence
            if abs(res - p2) < self.tolerance:
                break
            
            p0, p1, p2 = p1, p2, res
            self.iterations += 1
        
        return res, self.iterations
