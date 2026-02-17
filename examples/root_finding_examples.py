"""
Examples demonstrating root finding methods in Numersis.
"""

import numpy as np
import math
from numersis import BisectionMethod, FixedPointMethod, NewtonRaphsonMethod, MullersMethod


def example_bisection():
    """Example: Find root of f(x) = x^2 - 2 in [0, 2]."""
    print("=" * 60)
    print("BISECTION METHOD EXAMPLE")
    print("=" * 60)
    
    f = lambda x: x**2 - 2
    
    solver = BisectionMethod(tolerance=1e-6)
    root, iterations = solver.solve(f, 0, 2)
    
    print(f"Function: f(x) = x^2 - 2")
    print(f"Interval: [0, 2]")
    print(f"Root found: {root:.10f}")
    print(f"Expected:   {math.sqrt(2):.10f}")
    print(f"Iterations: {iterations}")
    print(f"Error: {abs(root - math.sqrt(2)):.2e}\n")


def example_fixed_point():
    """Example: Find fixed point of g(x) = cos(x)."""
    print("=" * 60)
    print("FIXED POINT METHOD EXAMPLE")
    print("=" * 60)
    
    g = lambda x: math.cos(x)
    
    solver = FixedPointMethod(tolerance=1e-6, max_iterations=100)
    fixed_point, iterations, history = solver.solve(g, 0.5)
    
    print(f"Function: g(x) = cos(x)")
    print(f"Initial guess: 0.5")
    print(f"Fixed point: {fixed_point:.10f}")
    print(f"Iterations: {iterations}")
    print(f"Verification: g(x) = {g(fixed_point):.10f})")
    print(f"Difference: {abs(g(fixed_point) - fixed_point):.2e}\n")


def example_newton_raphson():
    """Example: Find root of f(x) = x^3 - x - 1."""
    print("=" * 60)
    print("NEWTON-RAPHSON METHOD EXAMPLE")
    print("=" * 60)
    
    f = lambda x: x**3 - x - 1
    df = lambda x: 3*x**2 - 1
    
    solver = NewtonRaphsonMethod(tolerance=1e-10)
    root, iterations = solver.solve(f, df, 1.5)
    
    print(f"Function: f(x) = x^3 - x - 1")
    print(f"Derivative: f'(x) = 3x^2 - 1")
    print(f"Initial guess: 1.5")
    print(f"Root found: {root:.10f}")
    print(f"f(root) = {f(root):.2e}")
    print(f"Iterations: {iterations}\n")


def example_mullers():
    """Example: Find root using Müller's method."""
    print("=" * 60)
    print("MÜLLER'S METHOD EXAMPLE")
    print("=" * 60)
    
    f = lambda x: x**4 - 16*x**3 + 500*x**2 - 8000*x + 32000
    
    solver = MullersMethod(tolerance=1e-4)
    root, iterations = solver.solve(f, 0, 1, 2)
    
    print(f"Function: f(x) = x^4 - 16x^3 + 500x^2 - 8000x + 32000")
    print(f"Initial points: p0=0, p1=1, p2=2")
    print(f"Root found: {root:.6f}")
    print(f"f(root) = {f(root):.2e}")
    print(f"Iterations: {iterations}\n")


if __name__ == "__main__":
    example_bisection()
    example_fixed_point()
    example_newton_raphson()
    example_mullers()
