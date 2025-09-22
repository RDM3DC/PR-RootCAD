"""Lightweight constraint solver scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .numeric import get_tolerance


@dataclass
class SolverVariable:
    name: str
    value: float = 0.0
    locked: bool = False


@dataclass
class SolverConstraint:
    name: str
    evaluate: Callable[[Dict[str, SolverVariable]], float]
    project: Optional[Callable[[Dict[str, SolverVariable], float], None]] = None
    weight: float = 1.0


class ConstraintSolver:
    def __init__(self) -> None:
        self.variables: Dict[str, SolverVariable] = {}
        self.constraints: Dict[str, SolverConstraint] = {}
        policy = get_tolerance()
        self.max_iterations: int = policy.max_iterations
        self.convergence: float = policy.linear

    def add_variable(self, name: str, value: float = 0.0, *, locked: bool = False) -> SolverVariable:
        var = SolverVariable(name=name, value=value, locked=locked)
        self.variables[name] = var
        return var

    def add_constraint(
        self,
        name: str,
        evaluate: Callable[[Dict[str, SolverVariable]], float],
        *,
        project: Optional[Callable[[Dict[str, SolverVariable], float], None]] = None,
        weight: float = 1.0,
    ) -> SolverConstraint:
        con = SolverConstraint(name=name, evaluate=evaluate, project=project, weight=weight)
        self.constraints[name] = con
        return con

    def solve(self) -> float:
        if not self.constraints:
            return 0.0
        tol = self.convergence
        max_residual = float("inf")
        iteration = 0
        while iteration < self.max_iterations and max_residual > tol:
            iteration += 1
            max_residual = 0.0
            for constraint in self.constraints.values():
                residual = constraint.evaluate(self.variables)
                max_residual = max(max_residual, abs(residual))
                if abs(residual) <= tol:
                    continue
                if constraint.project is None:
                    continue
                constraint.project(self.variables, residual * constraint.weight)
        return max_residual


__all__ = ["SolverVariable", "SolverConstraint", "ConstraintSolver"]
