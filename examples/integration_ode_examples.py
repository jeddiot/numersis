"""
Examples demonstrating numerical integration and ODE solving in Numersis.
"""

import numpy as np
import math
from numersis import (
    CompositeSimpsonRule, RombergIntegration, GaussianQuadrature,
    AdaptiveQuadrature, SimpsonsDoubleIntegral,
    EulerMethod, RungeKutta4
)


def example_composite_simpson():
    """Example: Composite Simpson's rule."""
    print("=" * 60)
    print("COMPOSITE SIMPSON'S RULE EXAMPLE")
    print("=" * 60)
    
    f = lambda x: -10 / (x ** 1.5)
    
    integrator = CompositeSimpsonRule()
    integral = integrator.solve(f, 5, 10, n=1000000)
    
    print(f"Function: f(x) = -10 / x^(3/2)")
    print(f"Interval: [5, 10]")
    print(f"Integral: {integral:.10f}")
    print()


def example_romberg():
    """Example: Romberg integration."""
    print("=" * 60)
    print("ROMBERG INTEGRATION EXAMPLE")
    print("=" * 60)
    
    f = lambda x: -10 / (x ** 1.5)
    
    integrator = RombergIntegration()
    integral = integrator.solve(f, 5, 10, n=5)
    
    print(f"Function: f(x) = -10 / x^(3/2)")
    print(f"Interval: [5, 10]")
    print(f"Integral (Romberg): {integral:.10f}")
    print()


def example_gaussian_quadrature():
    """Example: Gaussian quadrature."""
    print("=" * 60)
    print("GAUSSIAN QUADRATURE EXAMPLE")
    print("=" * 60)
    
    f = lambda x: np.exp(x ** 2)
    
    integrator = GaussianQuadrature()
    
    print(f"Function: f(x) = e^(x^2)")
    print(f"Interval: [0, 1]")
    print("\nApproximations with different node counts:")
    
    for n in range(2, 10):
        integral = integrator.solve(f, 0, 1, n=n)
        print(f"  n={n}: {integral:.10f}")
    print()


def example_adaptive_quadrature():
    """Example: Adaptive quadrature."""
    print("=" * 60)
    print("ADAPTIVE QUADRATURE EXAMPLE")
    print("=" * 60)
    
    f = lambda x: (1/5) * (np.cos(2*x) - np.cos(3*x))
    
    integrator = AdaptiveQuadrature()
    integral = integrator.solve(f, 0, 2*np.pi, tol=1e-4)
    
    print(f"Function: f(x) = (1/5)(cos(2x) - cos(3x))")
    print(f"Interval: [0, 2π]")
    print(f"Tolerance: 1e-4")
    print(f"Integral: {integral:.10f}")
    print()


def example_simpsons_double_integral():
    """Example: Simpson's rule for double integrals."""
    print("=" * 60)
    print("SIMPSON'S DOUBLE INTEGRAL EXAMPLE")
    print("=" * 60)
    
    # ∫∫ e^(-(x²+y²)) dy dx over quarter circle
    f = lambda x, y: math.exp(-(x**2 + y**2))
    
    integrator = SimpsonsDoubleIntegral()
    integral = integrator.solve(
        f, 0, 1,
        lambda x: 0,
        lambda x: math.sqrt(1 - x**2),
        m=14, n=14
    )
    
    print(f"Function: f(x, y) = e^(-(x²+y²))")
    print(f"Domain: Quarter circle, 0 ≤ x ≤ 1, 0 ≤ y ≤ √(1-x²)")
    print(f"Double integral: {integral:.10f}")
    print()


def example_euler_method():
    """Example: Euler's method for ODE solving."""
    print("=" * 60)
    print("EULER METHOD EXAMPLE")
    print("=" * 60)
    
    # dy/dt = -0.006√64.2 * y^(-1.5)
    f = lambda t, y: -0.006 * math.sqrt(64.2) * y ** (-1.5)
    
    solver = EulerMethod()
    t_vals, y_vals = solver.solve(f, 0, 1600, 8, num_steps=80)
    
    print(f"ODE: dy/dt = -0.006√64.2 * y^(-1.5)")
    print(f"Initial condition: y(0) = 8")
    print(f"Time interval: [0, 1600]")
    print(f"Number of steps: 80")
    
    print(f"\nResults:")
    for i in [0, 20, 40, 60, 80]:
        if i < len(t_vals):
            print(f"  t[{i:2}] = {t_vals[i]:7.1f}, y[{i:2}] = {y_vals[i]:8.5f}")
    print()


def example_rungekutta4():
    """Example: Fourth-order Runge-Kutta method."""
    print("=" * 60)
    print("RUNGE-KUTTA 4TH ORDER METHOD EXAMPLE")
    print("=" * 60)
    
    # dy/dt = -0.006√64.2 * y^(-1.5)
    f = lambda t, y: -0.006 * math.sqrt(64.2) * y ** (-1.5)
    
    solver = RungeKutta4()
    t_vals, y_vals = solver.solve(f, 0, 1600, 8, num_steps=80)
    
    print(f"ODE: dy/dt = -0.006√64.2 * y^(-1.5)")
    print(f"Initial condition: y(0) = 8")
    print(f"Time interval: [0, 1600]")
    print(f"Number of steps: 80")
    
    print(f"\nResults:")
    for i in [0, 20, 40, 60, 80]:
        if i < len(t_vals):
            print(f"  t[{i:2}] = {t_vals[i]:7.1f}, y[{i:2}] = {y_vals[i]:8.5f}")
    print()


if __name__ == "__main__":
    example_composite_simpson()
    example_romberg()
    example_gaussian_quadrature()
    example_adaptive_quadrature()
    example_simpsons_double_integral()
    example_euler_method()
    example_rungekutta4()
