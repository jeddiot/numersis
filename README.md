# Numersis

A comprehensive Python library implementing numerical analysis algorithms based on Burden & Faires' Numerical Analysis textbook.

![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)

## Installation

### Via pip (when published)
```bash
pip install numersis
```

### From source
```bash
git clone <repository-url>
cd numersis
pip install -e .
```

### Requirements
- Python 3.7+
- NumPy
- SciPy (optional, for enhanced performance)

## Usage

### Root Finding Methods
```python
from numersis import BisectionMethod, NewtonRaphsonMethod

# Bisection method
bisection = BisectionMethod(tolerance=1e-6)
root, iterations = bisection.solve(lambda x: x**2 - 2, 0, 2)

# Newton-Raphson method
newton = NewtonRaphsonMethod(tolerance=1e-10)
root, iterations = newton.solve(lambda x: x**2 - 2, lambda x: 2*x, 1.5)
```

### Numerical Integration
```python
from numersis import CompositeSimpsonRule, RombergIntegration

# Simpson's composite rule
simpson = CompositeSimpsonRule()
integral = simpson.solve(lambda x: x**2, 0, 1, n=100)

# Romberg integration for higher precision
romberg = RombergIntegration()
integral = romberg.solve(lambda x: x**2, 0, 1, n=5)
```

### Interpolation
```python
from numersis import LagrangePolynomial, NewtonDividedDifference

# Lagrange polynomial interpolation
import numpy as np
x_data = np.array([0, 1, 2])
y_data = np.array([1, 2, 5])
lagrange = LagrangePolynomial(x_data, y_data)
y_at_0_5 = lagrange(0.5)  # Evaluate at x=0.5

# Newton divided difference
newton_diff = NewtonDividedDifference(x_data, y_data)
y_at_0_5 = newton_diff.evaluate(0.5)
```

### ODE Solving
```python
from numersis import RungeKutta4, EulerMethod

# 4th-order Runge-Kutta
rk4 = RungeKutta4()
t_vals, y_vals = rk4.solve(lambda t, y: -0.1*y, 0, 10, 1.0, num_steps=100)

# Euler's method
euler = EulerMethod()
t_vals, y_vals = euler.solve(lambda t, y: -0.1*y, 0, 10, 1.0, num_steps=100)
```

### Linear Systems
```python
from numersis import GaussElimination, JacobiIterationMethod

# Gaussian elimination with pivoting
A = [[4, 3], [6, 3]]
b = [10, 12]
gauss = GaussElimination()
x = gauss.solve(A, b)

# Jacobi iteration for sparse systems
jacobi = JacobiIterationMethod(tolerance=1e-6)
x = jacobi.solve(A, b)
```

## Core Modules

```
numersis/core/
├── root_finding.py          (5 methods)
│   └── BisectionMethod, FixedPointMethod, NewtonRaphsonMethod, 
│       FalsePositionMethod, MullersMethod
│
├── interpolation.py         (3 methods)
│   └── LagrangePolynomial, NewtonDividedDifference, ClampedCubicSpline
│
├── integration.py           (5 methods)
│   └── CompositeSimpsonRule, RombergIntegration, GaussianQuadrature,
│       AdaptiveQuadrature, SimpsonsDoubleIntegral
│
├── ode_solvers.py           (7 methods)
│   ├── Differentiation: ThreePointMidpointFormula, ThreePointEndpointFormula, FivePointFormula
│   └── ODE: EulerMethod, HigherOrderTaylorMethod, RungeKutta4, AdamsFourthOrderPredictorCorrector
│
└── linear_algebra.py        (8 methods)
    ├── Factorization: LUFactorization, CholeskyFactorization, LDLtFactorization
    └── Solvers: GaussElimination, JacobiIterationMethod, GaussSeidelMethod, LUSolver, CholeskySolver
```

## Documentation

- **[OOP_REFACTORING_GUIDE.md](OOP_REFACTORING_GUIDE.md)** - Comprehensive guide to the OOP architecture, migration guide from old API, and design patterns
- **[examples/](examples/)** - Example scripts demonstrating each category of methods:
  - root_finding_examples.py
  - interpolation_examples.py
  - integration_ode_examples.py
  - linear_algebra_examples.py

## Design

All methods follow a consistent object-oriented design:

1. **Inheritance Hierarchy** - All methods inherit from abstract base classes:
   - `RootFinder` - for root-finding methods
   - `Interpolator` - for interpolation
   - `Integrator` - for numerical integration
   - `Differentiator` - for numerical differentiation
   - `ODESolver` - for ODE solvers
   - `LinearSolver` / `MatrixFactorization` - for linear algebra

2. **Uniform Interface** - All solvers use consistent method signatures:
   ```python
   solver = MethodClass(tolerance=1e-6, max_iterations=100)
   result = solver.solve(...)
   ```

3. **State Management** - Iteration counts and convergence information are tracked:
   ```python
   from numersis import BisectionMethod
   solver = BisectionMethod()
   root, iterations = solver.solve(f, a, b)
   ```

## References

- [Numerical Analysis, 10th Edition - Burden & Faires, 2015](https://www.researchgate.net/publication/271133326_Numerical_Analysis_10th_ed)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
