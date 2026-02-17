"""
Numerical differentiation and ODE solver methods.
"""

from typing import Callable, Tuple
import numpy as np
from .base import Differentiator, ODESolver


# ============================================================================
# DIFFERENTIATION METHODS
# ============================================================================

class ThreePointMidpointFormula(Differentiator):
    """Three-point midpoint formula for numerical differentiation."""
    
    def solve(self, f: Callable, x: float, h: float = 0.01) -> float:
        """
        Approximate f'(x) using three-point midpoint formula.
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        
        Args:
            f: Function to differentiate.
            x: Point at which to compute derivative.
            h: Step size.
            
        Returns:
            Approximate derivative.
        """
        return (f(x + h) - f(x - h)) / (2 * h)


class ThreePointEndpointFormula(Differentiator):
    """Three-point endpoint formula for numerical differentiation."""
    
    def solve(self, f: Callable, x: float, h: float = 0.01, forward: bool = True) -> float:
        """
        Approximate f'(x) using three-point endpoint formula.
        
        Args:
            f: Function to differentiate.
            x: Point at which to compute derivative.
            h: Step size.
            forward: If True, use forward formula; else use backward.
            
        Returns:
            Approximate derivative.
        """
        if forward:
            # Forward: f'(x) ≈ [-3f(x) + 4f(x+h) - f(x+2h)] / (2h)
            return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)
        else:
            # Backward: f'(x) ≈ [f(x-2h) - 4f(x-h) + 3f(x)] / (2h)
            return (f(x - 2 * h) - 4 * f(x - h) + 3 * f(x)) / (2 * h)


class FivePointFormula(Differentiator):
    """Five-point formula for higher-order numerical differentiation."""
    
    def solve(self, f: Callable, x: float, h: float = 0.01) -> float:
        """
        Approximate f'(x) using five-point formula (O(h^4) accurate).
        
        Args:
            f: Function to differentiate.
            x: Point at which to compute derivative.
            h: Step size.
            
        Returns:
            Approximate derivative.
        """
        return (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h)) / (12 * h)


# ============================================================================
# ODE SOLVER METHODS
# ============================================================================

class EulerMethod(ODESolver):
    """Euler's method for solving ODEs."""
    
    def solve(self, f: Callable, t0: float, t_final: float, y0: float, 
              num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) with y(t0) = y0 using Euler's method.
        
        Args:
            f: Function f(t, y) representing dy/dt.
            t0: Initial time.
            t_final: Final time.
            y0: Initial condition y(t0).
            num_steps: Number of steps.
            
        Returns:
            Tuple of (t_values, y_values).
        """
        self.reset()
        
        h = (t_final - t0) / num_steps
        t_vals = np.zeros(num_steps + 1)
        y_vals = np.zeros(num_steps + 1)
        
        t_vals[0] = t0
        y_vals[0] = y0
        
        for i in range(1, num_steps + 1):
            y_vals[i] = y_vals[i - 1] + h * f(t_vals[i - 1], y_vals[i - 1])
            t_vals[i] = t0 + i * h
            self.iterations += 1
        
        return t_vals, y_vals


class HigherOrderTaylorMethod(ODESolver):
    """Higher-order Taylor series method for solving ODEs."""
    
    def solve(self, f: Callable, t0: float, t_final: float, y0: float, 
              num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) using higher-order Taylor method.
        
        Args:
            f: Function f(t, y) representing dy/dt. 
               For p-order method, f should return the p-th derivative.
            t0: Initial time.
            t_final: Final time.
            y0: Initial condition.
            num_steps: Number of steps.
            
        Returns:
            Tuple of (t_values, y_values).
        """
        self.reset()
        
        h = (t_final - t0) / num_steps
        t_vals = np.zeros(num_steps + 1)
        y_vals = np.zeros(num_steps + 1)
        
        t_vals[0] = t0
        y_vals[0] = y0
        
        for i in range(1, num_steps + 1):
            # Apply Taylor expansion: y_{i+1} = y_i + h*T(w_i, h)
            y_vals[i] = y_vals[i - 1] + h * f(y_vals[i - 1], h)
            t_vals[i] = t0 + i * h
            self.iterations += 1
        
        return t_vals, y_vals


class RungeKutta4(ODESolver):
    """Fourth-order Runge-Kutta method (RK4) for solving ODEs."""
    
    def solve(self, f: Callable, t0: float, t_final: float, y0: float, 
              num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) with y(t0) = y0 using RK4 method.
        
        Args:
            f: Function f(t, y) representing dy/dt.
            t0: Initial time.
            t_final: Final time.
            y0: Initial condition y(t0).
            num_steps: Number of steps.
            
        Returns:
            Tuple of (t_values, y_values).
        """
        self.reset()
        
        h = (t_final - t0) / num_steps
        t_vals = np.zeros(num_steps + 1)
        y_vals = np.zeros(num_steps + 1)
        
        t_vals[0] = t0
        y_vals[0] = y0
        
        for i in range(1, num_steps + 1):
            k1 = h * f(t_vals[i - 1], y_vals[i - 1])
            k2 = h * f(t_vals[i - 1] + 0.5 * h, y_vals[i - 1] + 0.5 * k1)
            k3 = h * f(t_vals[i - 1] + 0.5 * h, y_vals[i - 1] + 0.5 * k2)
            k4 = h * f(t_vals[i - 1] + h, y_vals[i - 1] + k3)
            
            y_vals[i] = y_vals[i - 1] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            t_vals[i] = t0 + i * h
            self.iterations += 1
        
        return t_vals, y_vals


class AdamsFourthOrderPredictorCorrector(ODESolver):
    """Fourth-order Adams predictor-corrector method for ODEs."""
    
    def solve(self, f: Callable, t0: float, t_final: float, y0: float, 
              num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) using Adams predictor-corrector method.
        Uses RK4 to get first 3 points, then applies Adams method.
        
        Args:
            f: Function f(t, y) representing dy/dt.
            t0: Initial time.
            t_final: Final time.
            y0: Initial condition y(t0).
            num_steps: Number of steps.
            
        Returns:
            Tuple of (t_values, y_values).
        """
        self.reset()
        
        h = (t_final - t0) / num_steps
        t_vals = np.arange(t0, t_final + h, h)
        y_vals = np.zeros(num_steps + 1)
        y_vals[0] = y0
        
        # Use RK4 for first 3 steps
        for i in range(0, min(3, num_steps)):
            k1 = h * f(t_vals[i], y_vals[i])
            k2 = h * f(t_vals[i] + 0.5*h, y_vals[i] + 0.5*k1)
            k3 = h * f(t_vals[i] + 0.5*h, y_vals[i] + 0.5*k2)
            k4 = h * f(t_vals[i] + h, y_vals[i] + k3)
            
            y_vals[i + 1] = y_vals[i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Adams predictor-corrector for remaining steps
        for i in range(3, num_steps):
            # Predictor (Adams-Bashforth)
            y_pred = (y_vals[i] + h * (55.0 * f(t_vals[i], y_vals[i]) - 
                                       59.0 * f(t_vals[i-1], y_vals[i-1]) + 
                                       37.0 * f(t_vals[i-2], y_vals[i-2]) - 
                                       9.0 * f(t_vals[i-3], y_vals[i-3])) / 24.0)
            
            # Corrector (Adams-Moulton)
            y_vals[i + 1] = (y_vals[i] + h * (9.0 * f(t_vals[i+1], y_pred) + 
                                               19.0 * f(t_vals[i], y_vals[i]) - 
                                               5.0 * f(t_vals[i-1], y_vals[i-1]) + 
                                               f(t_vals[i-2], y_vals[i-2])) / 24.0)
            
            self.iterations += 1
        
        return t_vals[:num_steps + 1], y_vals[:num_steps + 1]
