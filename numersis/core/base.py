"""
Abstract base classes for all numerical methods in Numersis.
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, List
import numpy as np


class NumericalMethod(ABC):
    """Base class for all numerical methods."""
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Args:
            tolerance: Convergence tolerance.
            max_iterations: Maximum number of iterations.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iterations = 0
    
    @abstractmethod
    def solve(self, *args, **kwargs):
        """Solve the problem."""
        pass
    
    def reset(self):
        """Reset iteration counter."""
        self.iterations = 0


class RootFinder(NumericalMethod):
    """Base class for root-finding methods."""
    
    @abstractmethod
    def solve(self, f: Callable, *args, **kwargs) -> float:
        """
        Find the root of function f.
        
        Args:
            f: Function to find root of.
            
        Returns:
            The approximate root.
        """
        pass


class Interpolator(NumericalMethod):
    """Base class for interpolation methods."""
    
    def __init__(self, nodes: np.ndarray, values: np.ndarray, **kwargs):
        """
        Args:
            nodes: X-coordinates of data points.
            values: Y-coordinates of data points.
        """
        super().__init__(**kwargs)
        self.nodes = np.asarray(nodes)
        self.values = np.asarray(values)
        
        if len(self.nodes) != len(self.values):
            raise ValueError("nodes and values must have the same length")
    
    @abstractmethod
    def evaluate(self, x: float) -> float:
        """
        Evaluate the interpolant at point x.
        
        Args:
            x: Point to evaluate at.
            
        Returns:
            Interpolated value.
        """
        pass
    
    def __call__(self, x):
        """Allow callable interface."""
        return self.evaluate(x)


class Integrator(NumericalMethod):
    """Base class for numerical integration methods."""
    
    @abstractmethod
    def solve(self, f: Callable, a: float, b: float, *args, **kwargs) -> float:
        """
        Integrate function f over [a, b].
        
        Args:
            f: Function to integrate.
            a: Lower bound.
            b: Upper bound.
            
        Returns:
            The approximate integral.
        """
        pass


class Differentiator(NumericalMethod):
    """Base class for numerical differentiation methods."""
    
    @abstractmethod
    def solve(self, f: Callable, x: float, h: float = None) -> float:
        """
        Approximate the derivative of f at x.
        
        Args:
            f: Function to differentiate.
            x: Point at which to compute derivative.
            h: Step size (if None, auto-computed).
            
        Returns:
            The approximate derivative.
        """
        pass


class ODESolver(NumericalMethod):
    """Base class for ODE solvers."""
    
    @abstractmethod
    def solve(self, f: Callable, t0: float, t_final: float, y0: float, 
              num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) with initial condition y(t0) = y0.
        
        Args:
            f: Function f(t, y) representing dy/dt.
            t0: Initial time.
            t_final: Final time.
            y0: Initial condition y(t0).
            num_steps: Number of steps.
            
        Returns:
            Tuple of (t_values, y_values).
        """
        pass


class LinearSolver(NumericalMethod):
    """Base class for linear system solvers."""
    
    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b.
        
        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            
        Returns:
            Solution vector x.
        """
        pass


class MatrixFactorization(NumericalMethod):
    """Base class for matrix factorization methods."""
    
    @abstractmethod
    def factorize(self, A: np.ndarray):
        """
        Factorize matrix A.
        
        Args:
            A: Matrix to factorize.
        """
        pass
