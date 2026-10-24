"""Core pricing models for the Cox-Ross-Rubinstein (CRR) Binomial Tree.

This module provides the fundamental data structures and numerical routines for
pricing European and American options using the CRR (1979) lattice methodology,
as well as analytical Black-Scholes benchmark solutions for verification.
All calculations rely exclusively on the Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OptionType(str, Enum):
    """Specification of option payoff type."""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    """Specification of option exercise style."""

    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class OptionContract:
    """Option contract specification and market parameters.

    Attributes:
        spot: Current underlying asset spot price (S_0 > 0).
        strike: Option strike / exercise price (K > 0).
        expiry: Time to expiration in years (T > 0).
        rate: Continuous risk-free interest rate (annualized).
        volatility: Annualized asset volatility (sigma > 0).
        dividend_yield: Continuous annualized dividend yield (q >= 0).
        option_type: CALL or PUT.
        exercise_style: EUROPEAN or AMERICAN.
    """

    spot: float
    strike: float
    expiry: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        """Validate financial parameters against domain invariants."""
        if self.spot <= 0.0:
            raise ValueError(f"Spot price must be strictly positive, got {self.spot}")
        if self.strike <= 0.0:
            raise ValueError(f"Strike price must be strictly positive, got {self.strike}")
        if self.expiry <= 0.0:
            raise ValueError(f"Time to expiry must be strictly positive, got {self.expiry}")
        if self.volatility <= 0.0:
            raise ValueError(f"Volatility must be strictly positive, got {self.volatility}")
        if self.dividend_yield < 0.0:
            raise ValueError(f"Dividend yield must be non-negative, got {self.dividend_yield}")

    def intrinsic_value(self, underlying_price: float) -> float:
        """Calculate immediate exercise intrinsic payoff.

        Args:
            underlying_price: Current asset price.

        Returns:
            Max(S - K, 0) for Call, Max(K - S, 0) for Put.
        """
        if self.option_type == OptionType.CALL:
            return max(underlying_price - self.strike, 0.0)
        return max(self.strike - underlying_price, 0.0)

    def is_call(self) -> bool:
        """Check if option is a Call."""
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """Check if option is a Put."""
        return self.option_type == OptionType.PUT

    def is_american(self) -> bool:
        """Check if option is American exercise style."""
        return self.exercise_style == ExerciseStyle.AMERICAN

    def is_european(self) -> bool:
        """Check if option is European exercise style."""
        return self.exercise_style == ExerciseStyle.EUROPEAN


@dataclass
class BinomialNode:
    """Individual state node in a binomial lattice.

    Attributes:
        step: Time index i in {0, ..., N}.
        state: Up-move count j in {0, ..., i}.
        spot: Asset price at this node S_{i,j}.
        option_value: Final option value after backward induction.
        continuation_value: Discounted risk-neutral expected continuation value.
        intrinsic_value: Immediate exercise value at this node.
        is_early_exercised: True if American early exercise is optimal.
        delta: Local lattice delta (if computable).
    """

    step: int
    state: int
    spot: float
    option_value: float = 0.0
    continuation_value: float = 0.0
    intrinsic_value: float = 0.0
    is_early_exercised: bool = False
    delta: Optional[float] = None


class CRRBinomialTree:
    """Cox-Ross-Rubinstein (1979) Binomial Option Pricing Model.

    Constructs a recombining binomial tree with parameters:
        dt = T / N
        u = exp(sigma * sqrt(dt))
        d = 1 / u = exp(-sigma * sqrt(dt))
        growth = exp((r - q) * dt)
        discount = exp(-r * dt)
        p = (growth - d) / (u - d)
    """

    def __init__(self, contract: OptionContract, steps: int = 100) -> None:
        """Initialize the CRR Binomial Tree.

        Args:
            contract: Option contract parameters.
            steps: Number of time discretization steps N (must be >= 1).

        Raises:
            ValueError: If steps < 1 or if no-arbitrage bounds are violated.
        """
        if steps < 1:
            raise ValueError(f"Tree steps must be at least 1, got {steps}")

        self.contract = contract
        self.steps = steps

        # Time discretization
        self.dt: float = contract.expiry / steps

        # CRR Up and Down factors
        self.u: float = math.exp(contract.volatility * math.sqrt(self.dt))
        self.d: float = 1.0 / self.u

        # Cost-of-carry growth and discount factors
        self.growth: float = math.exp((contract.rate - contract.dividend_yield) * self.dt)
        self.discount: float = math.exp(-contract.rate * self.dt)

        # Risk-neutral probability
        denominator = self.u - self.d
        if abs(denominator) < 1e-15:
            raise ValueError("Up and down factors collapsed; check volatility or time step.")

        self.p: float = (self.growth - self.d) / denominator
        self.q_prob: float = 1.0 - self.p

        # No-arbitrage validation
        if not (self.d < self.growth < self.u):
            raise ValueError(
                f"No-arbitrage condition violated: d ({self.d:.6f}) < growth ({self.growth:.6f}) < u ({self.u:.6f}). "
                f"Calculated risk-neutral probability p = {self.p:.6f}. Increase steps N or adjust parameters."
            )

    def asset_price(self, step: int, state: int) -> float:
        """Compute the underlying asset price at a given step and state.

        Using S_{i, j} = S_0 * u^j * d^(i - j) = S_0 * u^(2j - i).

        Args:
            step: Time step index i (0 <= i <= N).
            state: Number of up moves j (0 <= j <= i).

        Returns:
            Asset price at node (i, j).
        """
        return self.contract.spot * (self.u ** state) * (self.d ** (step - state))

    def price(self) -> float:
        """Compute the option fair value using O(N) space rolling buffer.

        Performs backward induction from maturity step N back to 0.
        Complexity: O(N^2) time, O(N) auxiliary space.

        Returns:
            Option fair value at t = 0.
        """
        n = self.steps
        # Terminal payoffs at step N: length N + 1
        values = [
            self.contract.intrinsic_value(self.asset_price(n, j))
            for j in range(n + 1)
        ]

        is_american = self.contract.is_american()

        # Backward induction
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                continuation = self.discount * (self.p * values[j + 1] + self.q_prob * values[j])
                if is_american:
                    intrinsic = self.contract.intrinsic_value(self.asset_price(i, j))
                    values[j] = max(continuation, intrinsic)
                else:
                    values[j] = continuation

        return values[0]

    def build_tree(self) -> List[List[BinomialNode]]:
        """Construct and return the full 2D triangular binomial lattice.

        Recommended for visualization, inspection, and local Greek extraction.
        Complexity: O(N^2) time, O(N^2) auxiliary space.

        Returns:
            A 2D list where tree[i][j] is the BinomialNode at step i and state j.
        """
        n = self.steps
        tree: List[List[BinomialNode]] = []

        # 1. Initialize node geometry and terminal payoffs
        for i in range(n + 1):
            row: List[BinomialNode] = []
            for j in range(i + 1):
                spot = self.asset_price(i, j)
                intrinsic = self.contract.intrinsic_value(spot)
                node = BinomialNode(
                    step=i,
                    state=j,
                    spot=spot,
                    intrinsic_value=intrinsic,
                )
                if i == n:
                    node.option_value = intrinsic
                    node.continuation_value = intrinsic
                    node.is_early_exercised = False
                row.append(node)
            tree.append(row)

        is_american = self.contract.is_american()

        # 2. Backward induction
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                down_child = tree[i + 1][j]
                up_child = tree[i + 1][j + 1]

                continuation = self.discount * (
                    self.p * up_child.option_value + self.q_prob * down_child.option_value
                )
                curr_node = tree[i][j]
                curr_node.continuation_value = continuation

                if is_american:
                    if curr_node.intrinsic_value > continuation:
                        curr_node.option_value = curr_node.intrinsic_value
                        curr_node.is_early_exercised = True
                    else:
                        curr_node.option_value = continuation
                        curr_node.is_early_exercised = False
                else:
                    curr_node.option_value = continuation
                    curr_node.is_early_exercised = False

                # Local node delta
                denom = up_child.spot - down_child.spot
                if abs(denom) > 1e-12:
                    curr_node.delta = (up_child.option_value - down_child.option_value) / denom

        return tree

    def early_exercise_premium(self) -> float:
        """Compute the American early exercise premium over European counterpart.

        Early Exercise Premium = American Price - European Price.

        Returns:
            Premium amount (always >= 0.0, up to float precision).
        """
        if self.contract.is_european():
            return 0.0

        european_contract = OptionContract(
            spot=self.contract.spot,
            strike=self.contract.strike,
            expiry=self.contract.expiry,
            rate=self.contract.rate,
            volatility=self.contract.volatility,
            dividend_yield=self.contract.dividend_yield,
            option_type=self.contract.option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        euro_tree = CRRBinomialTree(european_contract, steps=self.steps)
        american_price = self.price()
        euro_price = euro_tree.price()
        return max(0.0, american_price - euro_price)


class BlackScholesBenchmark:
    """Exact analytical Black-Scholes-Merton (1973) benchmark solver.

    Zero-dependency implementation utilizing math.erf for Gaussian cumulative
    distribution evaluation.
    """

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Cumulative distribution function of standard normal distribution N(0, 1).

        Uses standard library math.erf: N(x) = 0.5 * (1 + erf(x / sqrt(2))).
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Probability density function of standard normal distribution n(0, 1)."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def d1_d2(cls, contract: OptionContract) -> Tuple[float, float]:
        """Compute d1 and d2 parameters.

        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / [sigma * sqrt(T)]
        d2 = d1 - sigma * sqrt(T)
        """
        s = contract.spot
        k = contract.strike
        t = contract.expiry
        r = contract.rate
        q = contract.dividend_yield
        v = contract.volatility

        numerator = math.log(s / k) + (r - q + 0.5 * v * v) * t
        denominator = v * math.sqrt(t)

        d1 = numerator / denominator
        d2 = d1 - denominator
        return d1, d2

    @classmethod
    def price(cls, contract: OptionContract) -> float:
        """Analytical European option price under Black-Scholes model.

        Call: S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
        Put:  K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)
        """
        s = contract.spot
        k = contract.strike
        t = contract.expiry
        r = contract.rate
        q = contract.dividend_yield

        d1, d2 = cls.d1_d2(contract)
        disc_s = s * math.exp(-q * t)
        disc_k = k * math.exp(-r * t)

        if contract.is_call():
            return disc_s * cls.norm_cdf(d1) - disc_k * cls.norm_cdf(d2)
        else:
            return disc_k * cls.norm_cdf(-d2) - disc_s * cls.norm_cdf(-d1)

    @classmethod
    def delta(cls, contract: OptionContract) -> float:
        """Analytical Black-Scholes Delta (dV/dS)."""
        t = contract.expiry
        q = contract.dividend_yield
        d1, _ = cls.d1_d2(contract)
        discount_q = math.exp(-q * t)

        if contract.is_call():
            return discount_q * cls.norm_cdf(d1)
        else:
            return discount_q * (cls.norm_cdf(d1) - 1.0)

    @classmethod
    def gamma(cls, contract: OptionContract) -> float:
        """Analytical Black-Scholes Gamma (d^2V/dS^2)."""
        s = contract.spot
        t = contract.expiry
        q = contract.dividend_yield
        v = contract.volatility
        d1, _ = cls.d1_d2(contract)

        return (math.exp(-q * t) * cls.norm_pdf(d1)) / (s * v * math.sqrt(t))

    @classmethod
    def theta(cls, contract: OptionContract) -> float:
        """Analytical Black-Scholes Theta (dV/dt = -dV/dT), annualized."""
        s = contract.spot
        k = contract.strike
        t = contract.expiry
        r = contract.rate
        q = contract.dividend_yield
        v = contract.volatility
        d1, d2 = cls.d1_d2(contract)

        term1 = -(s * math.exp(-q * t) * cls.norm_pdf(d1) * v) / (2.0 * math.sqrt(t))

        if contract.is_call():
            term2 = -r * k * math.exp(-r * t) * cls.norm_cdf(d2)
            term3 = q * s * math.exp(-q * t) * cls.norm_cdf(d1)
            return term1 + term2 + term3
        else:
            term2 = r * k * math.exp(-r * t) * cls.norm_cdf(-d2)
            term3 = -q * s * math.exp(-q * t) * cls.norm_cdf(-d1)
            return term1 + term2 + term3

    @classmethod
    def vega(cls, contract: OptionContract) -> float:
        """Analytical Black-Scholes Vega (dV/dsigma), per 1.0 (100%) volatility change."""
        s = contract.spot
        t = contract.expiry
        q = contract.dividend_yield
        d1, _ = cls.d1_d2(contract)

        return s * math.exp(-q * t) * math.sqrt(t) * cls.norm_pdf(d1)

    @classmethod
    def rho(cls, contract: OptionContract) -> float:
        """Analytical Black-Scholes Rho (dV/dr), per 1.0 (100%) rate change."""
        k = contract.strike
        t = contract.expiry
        r = contract.rate
        _, d2 = cls.d1_d2(contract)

        if contract.is_call():
            return k * t * math.exp(-r * t) * cls.norm_cdf(d2)
        else:
            return -k * t * math.exp(-r * t) * cls.norm_cdf(-d2)
