"""
Core numerical methods module for Numersis library.
"""

# Base classes
from .base import (
    NumericalMethod,
    RootFinder,
    Interpolator,
    Integrator,
    Differentiator,
    ODESolver,
    LinearSolver,
    MatrixFactorization,
)

# Root finding methods
from .root_finding import (
    BisectionMethod,
    FixedPointMethod,
    NewtonRaphsonMethod,
    FalsePositionMethod,
    MullersMethod,
)

# Interpolation methods
from .interpolation import (
    LagrangePolynomial,
    NewtonDividedDifference,
    ClampedCubicSpline,
)

# Integration methods
from .integration import (
    CompositeSimpsonRule,
    RombergIntegration,
    GaussianQuadrature,
    AdaptiveQuadrature,
    SimpsonsDoubleIntegral,
)

# Differentiation methods
from .ode_solvers import (
    ThreePointMidpointFormula,
    ThreePointEndpointFormula,
    FivePointFormula,
    EulerMethod,
    HigherOrderTaylorMethod,
    RungeKutta4,
    AdamsFourthOrderPredictorCorrector,
)

# Linear algebra methods
from .linear_algebra import (
    LUFactorization,
    CholeskyFactorization,
    LDLtFactorization,
    GaussElimination,
    JacobiIterationMethod,
    GaussSeidelMethod,
    LUSolver,
    CholeskySolver,
)

__all__ = [
    # Base classes
    "NumericalMethod",
    "RootFinder",
    "Interpolator",
    "Integrator",
    "Differentiator",
    "ODESolver",
    "LinearSolver",
    "MatrixFactorization",
    # Root finding
    "BisectionMethod",
    "FixedPointMethod",
    "NewtonRaphsonMethod",
    "FalsePositionMethod",
    "MullersMethod",
    # Interpolation
    "LagrangePolynomial",
    "NewtonDividedDifference",
    "ClampedCubicSpline",
    # Integration
    "CompositeSimpsonRule",
    "RombergIntegration",
    "GaussianQuadrature",
    "AdaptiveQuadrature",
    "SimpsonsDoubleIntegral",
    # Differentiation and ODE
    "ThreePointMidpointFormula",
    "ThreePointEndpointFormula",
    "FivePointFormula",
    "EulerMethod",
    "HigherOrderTaylorMethod",
    "RungeKutta4",
    "AdamsFourthOrderPredictorCorrector",
    # Linear algebra
    "LUFactorization",
    "CholeskyFactorization",
    "LDLtFactorization",
    "GaussElimination",
    "JacobiIterationMethod",
    "GaussSeidelMethod",
    "LUSolver",
    "CholeskySolver",
]
