"""
Examples demonstrating interpolation methods in Numersis.
"""

import numpy as np
from numersis import LagrangePolynomial, NewtonDividedDifference, ClampedCubicSpline


def example_lagrange():
    """Example: Lagrange polynomial interpolation."""
    print("=" * 60)
    print("LAGRANGE POLYNOMIAL EXAMPLE")
    print("=" * 60)
    
    # Data points
    x_data = np.array([0, 10, 20, 30, 40, 50])
    y_data = np.array([179323, 203302, 226542, 249633, 281422, 308746])
    
    interpolator = LagrangePolynomial(x_data, y_data)
    
    # Evaluate at several points
    test_points = [-10, 15, 54, 60]
    
    print("Data points:")
    for x, y in zip(x_data, y_data):
        print(f"  ({x}, {y})")
    
    print("\nInterpolation results:")
    for x_test in test_points:
        y_interp = interpolator(x_test)
        print(f"  f({x_test:3}) ≈ {y_interp:10.2f}")
    print()


def example_newton_divided_difference():
    """Example: Newton's divided difference formula."""
    print("=" * 60)
    print("NEWTON DIVIDED DIFFERENCE EXAMPLE")
    print("=" * 60)
    
    x_data = np.array([0, 10, 20, 30, 40, 50])
    y_data = np.array([179323, 203302, 226542, 249633, 281422, 308746])
    
    interpolator = NewtonDividedDifference(x_data, y_data)
    
    print("Data points:")
    for x, y in zip(x_data, y_data):
        print(f"  ({x}, {y})")
    
    print("\nDivided difference table:")
    for i, row in enumerate(interpolator.dd_table):
        print(f"  Row {i}: {row}")
    
    print("\nInterpolation results:")
    for x_test in [-10, 15, 54, 60]:
        y_interp = interpolator(x_test)
        print(f"  f({x_test:3}) ≈ {y_interp:10.2f}")
    print()


def example_clamped_cubic_spline():
    """Example: Clamped cubic spline interpolation."""
    print("=" * 60)
    print("CLAMPED CUBIC SPLINE EXAMPLE")
    print("=" * 60)
    
    x_data = np.array([0, 3, 5, 8, 13], dtype=float)
    y_data = np.array([0, 225, 383, 623, 993], dtype=float)
    
    # Clamped spline with specified derivatives at endpoints
    spline = ClampedCubicSpline(
        x_data, y_data,
        first_deriv_start=75,  # f'(0) = 75
        first_deriv_end=72     # f'(13) = 72
    )
    
    print("Data points:")
    for x, y in zip(x_data, y_data):
        print(f"  ({x}, {y})")
    
    print(f"\nBoundary conditions:")
    print(f"  f'(0) = {75}")
    print(f"  f'(13) = {72}")
    
    print("\nSpline evaluation:")
    x_test = 10
    y_spline = spline(x_test)
    print(f"  f({x_test}) ≈ {y_spline:.2f}")
    print()


if __name__ == "__main__":
    example_lagrange()
    example_newton_divided_difference()
    example_clamped_cubic_spline()
