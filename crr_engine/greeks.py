"""Analytical and numerical finite-difference Greeks engine for the CRR Binomial Model.

This module provides two distinct extraction methodologies:
1. Lattice Greeks: Extracted directly from early nodes (steps 0, 1, 2) of the binomial tree
   without requiring extra model evaluations.
2. Bump-and-Reprice Greeks: Central finite-difference perturbations for Delta, Gamma,
   Theta, Vega, and Rho.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from crr_engine.models import (
    CRRBinomialTree,
    ExerciseStyle,
    OptionContract,
    OptionType,
    BlackScholesBenchmark,
)


@dataclass(frozen=True)
class GreeksResult:
    """Container for option sensitivities (Greeks).

    Attributes:
        delta: First derivative of option price with respect to underlying spot (dV/dS).
        gamma: Second derivative of option price with respect to underlying spot (d^2V/dS^2).
        theta: Derivative of option price with respect to calendar time (dV/dt), annualized.
        theta_daily: Theta per calendar day (theta / 365.0).
        vega: Derivative with respect to volatility (dV/dsigma), per 1.0 (100%) volatility move.
        rho: Derivative with respect to risk-free interest rate (dV/dr), per 1.0 (100%) rate move.
        method: Methodology used ("lattice", "numerical_bump", or "analytical_bs").
        steps: Number of binomial steps N used (if applicable).
    """

    delta: float
    gamma: float
    theta: float
    theta_daily: float
    vega: Optional[float] = None
    rho: Optional[float] = None
    method: str = "lattice"
    steps: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert sensitivities to a structured dictionary."""
        return {
            "method": self.method,
            "steps": self.steps,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta_annual": self.theta,
            "theta_daily": self.theta_daily,
            "vega_100pct": self.vega,
            "vega_1pct": (self.vega / 100.0) if self.vega is not None else None,
            "rho_100pct": self.rho,
            "rho_1pct": (self.rho / 100.0) if self.rho is not None else None,
        }


class GreeksEngine:
    """High-performance Greeks calculator for binomial option pricing."""

    @staticmethod
    def calculate_lattice_greeks(contract: OptionContract, steps: int = 100) -> GreeksResult:
        """Extract Delta, Gamma, and Theta directly from early lattice nodes.

        Requires steps >= 2.
        - Delta: Finite difference across step 1 nodes:
            Delta = (V_{1,1} - V_{1,0}) / (S_{1,1} - S_{1,0})
        - Gamma: Curvature across step 2 nodes:
            Delta_u = (V_{2,2} - V_{2,1}) / (S_{2,2} - S_{2,1})
            Delta_d = (V_{2,1} - V_{2,0}) / (S_{2,1} - S_{2,0})
            Gamma = (Delta_u - Delta_d) / [0.5 * (S_{2,2} - S_{2,0})]
        - Theta: Time decay between node (0,0) and identical-spot node (2,1) [since S_{2,1} = S_0]:
            Theta = (V_{2,1} - V_{0,0}) / (2 * dt)

        Args:
            contract: Option contract parameters.
            steps: Discretization steps (must be >= 2).

        Returns:
            GreeksResult containing lattice Delta, Gamma, Theta, and daily Theta.
        """
        if steps < 2:
            raise ValueError(f"Lattice Greek extraction requires at least 2 steps, got {steps}")

        tree_model = CRRBinomialTree(contract, steps=steps)
        tree = tree_model.build_tree()

        # Step 0: root node
        node_0_0 = tree[0][0]
        v_0 = node_0_0.option_value

        # Step 1: up and down nodes
        node_1_0 = tree[1][0]
        node_1_1 = tree[1][1]
        denom_delta = node_1_1.spot - node_1_0.spot
        if abs(denom_delta) < 1e-15:
            delta = 0.0
        else:
            delta = (node_1_1.option_value - node_1_0.option_value) / denom_delta

        # Step 2: 3 nodes
        node_2_0 = tree[2][0]
        node_2_1 = tree[2][1]
        node_2_2 = tree[2][2]

        denom_u = node_2_2.spot - node_2_1.spot
        denom_d = node_2_1.spot - node_2_0.spot

        delta_u = (node_2_2.option_value - node_2_1.option_value) / denom_u if abs(denom_u) > 1e-15 else 0.0
        delta_d = (node_2_1.option_value - node_2_0.option_value) / denom_d if abs(denom_d) > 1e-15 else 0.0

        h_spread = 0.5 * (node_2_2.spot - node_2_0.spot)
        gamma = (delta_u - delta_d) / h_spread if abs(h_spread) > 1e-15 else 0.0

        # Theta from root (t=0) to center node (t=2*dt) where spot is unchanged S_{2,1} = S_0
        time_elapsed = 2.0 * tree_model.dt
        theta = (node_2_1.option_value - v_0) / time_elapsed if time_elapsed > 0.0 else 0.0
        theta_daily = theta / 365.0

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            theta_daily=theta_daily,
            vega=None,
            rho=None,
            method="lattice",
            steps=steps,
        )

    @classmethod
    def calculate_bump_greeks(
        cls,
        contract: OptionContract,
        steps: int = 100,
        spot_rel_bump: float = 0.03,
        vol_abs_bump: float = 0.01,
        rate_abs_bump: float = 0.0001,
        time_step_days: float = 1.0,
    ) -> GreeksResult:
        """Compute sensitivities using central finite-difference perturbations.

        Quantitative Note on Lattice Gamma:
            Discrete binomial trees exhibit high-frequency sawtooth oscillations (strike-
            crossing effects) when evaluating numerical second differences at very small h.
            To avoid numerical instability, spot_rel_bump is set to a multi-node span (default 3%).
            For a zero-recomputation, mesh-consistent Gamma, `calculate_lattice_greeks()` is
            preferred.

        Args:
            contract: Option contract parameters.
            steps: Number of steps N for pricing each bumped tree.
            spot_rel_bump: Relative perturbation h_s = spot_rel_bump * S_0 (default 3%).
            vol_abs_bump: Absolute volatility perturbation h_vol (default 1.0%).
            rate_abs_bump: Absolute interest rate perturbation h_r (default 1 bp = 0.0001).
            time_step_days: Time decay step for Theta in calendar days (default 1 day).

        Returns:
            GreeksResult containing Delta, Gamma, Theta, Vega, and Rho.
        """
        base_price = CRRBinomialTree(contract, steps=steps).price()

        # --- Delta & Gamma (Spot bump) ---
        h_s = max(contract.spot * spot_rel_bump, 1e-4)

        contract_up = OptionContract(
            spot=contract.spot + h_s,
            strike=contract.strike,
            expiry=contract.expiry,
            rate=contract.rate,
            volatility=contract.volatility,
            dividend_yield=contract.dividend_yield,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        contract_dn = OptionContract(
            spot=max(contract.spot - h_s, 1e-6),
            strike=contract.strike,
            expiry=contract.expiry,
            rate=contract.rate,
            volatility=contract.volatility,
            dividend_yield=contract.dividend_yield,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )

        p_up = CRRBinomialTree(contract_up, steps=steps).price()
        p_dn = CRRBinomialTree(contract_dn, steps=steps).price()

        delta = (p_up - p_dn) / (2.0 * h_s)
        gamma = (p_up - 2.0 * base_price + p_dn) / (h_s * h_s)

        # --- Theta (Time decay bump) ---
        # Theta = dV/dt = -dV/dT. We decay maturity by h_t years.
        dt_years = time_step_days / 365.0
        if contract.expiry > dt_years * 1.5:
            contract_decay = OptionContract(
                spot=contract.spot,
                strike=contract.strike,
                expiry=contract.expiry - dt_years,
                rate=contract.rate,
                volatility=contract.volatility,
                dividend_yield=contract.dividend_yield,
                option_type=contract.option_type,
                exercise_style=contract.exercise_style,
            )
            p_decay = CRRBinomialTree(contract_decay, steps=steps).price()
            theta = (p_decay - base_price) / dt_years
            theta_daily = (p_decay - base_price) / time_step_days
        else:
            theta = 0.0
            theta_daily = 0.0

        # --- Vega (Volatility bump) ---
        h_v = vol_abs_bump
        contract_vol_up = OptionContract(
            spot=contract.spot,
            strike=contract.strike,
            expiry=contract.expiry,
            rate=contract.rate,
            volatility=contract.volatility + h_v,
            dividend_yield=contract.dividend_yield,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        contract_vol_dn = OptionContract(
            spot=contract.spot,
            strike=contract.strike,
            expiry=contract.expiry,
            rate=contract.rate,
            volatility=max(contract.volatility - h_v, 1e-4),
            dividend_yield=contract.dividend_yield,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        p_vol_up = CRRBinomialTree(contract_vol_up, steps=steps).price()
        p_vol_dn = CRRBinomialTree(contract_vol_dn, steps=steps).price()
        actual_h_v = (contract_vol_up.volatility - contract_vol_dn.volatility) / 2.0
        vega = (p_vol_up - p_vol_dn) / (2.0 * actual_h_v)

        # --- Rho (Interest rate bump) ---
        h_r = rate_abs_bump
        contract_r_up = OptionContract(
            spot=contract.spot,
            strike=contract.strike,
            expiry=contract.expiry,
            rate=contract.rate + h_r,
            volatility=contract.volatility,
            dividend_yield=contract.dividend_yield,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        contract_r_dn = OptionContract(
            spot=contract.spot,
            strike=contract.strike,
            expiry=contract.expiry,
            rate=contract.rate - h_r,
            volatility=contract.volatility,
            dividend_yield=contract.dividend_yield,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        p_r_up = CRRBinomialTree(contract_r_up, steps=steps).price()
        p_r_dn = CRRBinomialTree(contract_r_dn, steps=steps).price()
        rho = (p_r_up - p_r_dn) / (2.0 * h_r)

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            theta_daily=theta_daily,
            vega=vega,
            rho=rho,
            method="numerical_bump",
            steps=steps,
        )

    @classmethod
    def calculate_analytical_bs_greeks(cls, contract: OptionContract) -> GreeksResult:
        """Calculate exact analytical European Black-Scholes Greeks."""
        delta = BlackScholesBenchmark.delta(contract)
        gamma = BlackScholesBenchmark.gamma(contract)
        theta = BlackScholesBenchmark.theta(contract)
        theta_daily = theta / 365.0
        vega = BlackScholesBenchmark.vega(contract)
        rho = BlackScholesBenchmark.rho(contract)

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            theta_daily=theta_daily,
            vega=vega,
            rho=rho,
            method="analytical_bs",
            steps=None,
        )

    @classmethod
    def compare_greeks(cls, contract: OptionContract, steps: int = 200) -> Dict[str, Any]:
        """Generate a comparison report across Lattice, Bump, and Analytical Greeks."""
        lattice = cls.calculate_lattice_greeks(contract, steps=steps)
        bump = cls.calculate_bump_greeks(contract, steps=steps)

        result: Dict[str, Any] = {
            "lattice": lattice.as_dict(),
            "numerical_bump": bump.as_dict(),
        }

        if contract.is_european():
            analytical = cls.calculate_analytical_bs_greeks(contract)
            result["analytical_bs"] = analytical.as_dict()

        return result
