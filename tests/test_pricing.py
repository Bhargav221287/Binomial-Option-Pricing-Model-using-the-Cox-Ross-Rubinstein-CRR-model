"""Comprehensive unit tests for the Cox-Ross-Rubinstein pricing engine.

Tests verify:
1. Black-Scholes benchmark analytical formulas, Put-Call Parity, and distribution metrics.
2. CRR European price convergence against Black-Scholes solutions across N steps.
3. American option boundary conditions, Merton's non-exercise theorem, and early exercise premiums.
4. Lattice and finite-difference Greeks extraction accuracy.
5. Equivalence between O(N) rolling buffer pricing and 2D lattice full tree construction.
6. Robust input parameter validation and edge cases.
"""

from __future__ import annotations

import math
import unittest

from crr_engine.models import (
    BinomialNode,
    BlackScholesBenchmark,
    CRRBinomialTree,
    ExerciseStyle,
    OptionContract,
    OptionType,
)
from crr_engine.greeks import (
    GreeksEngine,
    GreeksResult,
)
from crr_engine.visualizer import (
    TreeVisualizer,
)


class TestBlackScholesBenchmark(unittest.TestCase):
    """Test suite for analytical Black-Scholes-Merton solver."""

    def setUp(self) -> None:
        self.contract_call = OptionContract(
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        self.contract_put = OptionContract(
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

    def test_norm_cdf_properties(self) -> None:
        """Verify normal CDF values at standard quantiles."""
        self.assertAlmostEqual(BlackScholesBenchmark.norm_cdf(0.0), 0.5, places=7)
        self.assertAlmostEqual(BlackScholesBenchmark.norm_cdf(1.95996398454), 0.975, places=5)
        self.assertAlmostEqual(BlackScholesBenchmark.norm_cdf(-1.95996398454), 0.025, places=5)

    def test_norm_pdf_properties(self) -> None:
        """Verify normal PDF maximum and symmetry."""
        peak = 1.0 / math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(BlackScholesBenchmark.norm_pdf(0.0), peak, places=7)
        self.assertAlmostEqual(
            BlackScholesBenchmark.norm_pdf(1.5),
            BlackScholesBenchmark.norm_pdf(-1.5),
            places=7,
        )

    def test_put_call_parity_analytical(self) -> None:
        """Verify C - P = S*exp(-qT) - K*exp(-rT) analytically."""
        c = BlackScholesBenchmark.price(self.contract_call)
        p = BlackScholesBenchmark.price(self.contract_put)
        s = self.contract_call.spot
        k = self.contract_call.strike
        r = self.contract_call.rate
        t = self.contract_call.expiry

        parity_lhs = c - p
        parity_rhs = s - k * math.exp(-r * t)
        self.assertAlmostEqual(parity_lhs, parity_rhs, places=10)

    def test_known_literature_values(self) -> None:
        """Benchmark against Hull standard textbook example (S=50, K=50, T=5/12, r=10%, sigma=40%)."""
        hull_call = OptionContract(
            spot=50.0,
            strike=50.0,
            expiry=5.0 / 12.0,
            rate=0.10,
            volatility=0.40,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
        )
        hull_put = OptionContract(
            spot=50.0,
            strike=50.0,
            expiry=5.0 / 12.0,
            rate=0.10,
            volatility=0.40,
            dividend_yield=0.0,
            option_type=OptionType.PUT,
        )
        price_c = BlackScholesBenchmark.price(hull_call)
        price_p = BlackScholesBenchmark.price(hull_put)

        # Expected: Call ~ 6.1166, Put ~ 4.0760
        self.assertAlmostEqual(price_c, 6.1166, places=3)
        self.assertAlmostEqual(price_p, 4.0760, places=3)

    def test_analytical_greeks_consistency(self) -> None:
        """Verify mathematical consistency of Black-Scholes Greeks."""
        c = self.contract_call
        p = self.contract_put

        delta_c = BlackScholesBenchmark.delta(c)
        delta_p = BlackScholesBenchmark.delta(p)
        gamma_c = BlackScholesBenchmark.gamma(c)
        gamma_p = BlackScholesBenchmark.gamma(p)
        vega_c = BlackScholesBenchmark.vega(c)
        vega_p = BlackScholesBenchmark.vega(p)

        # Delta bounds
        self.assertTrue(0.0 < delta_c < 1.0)
        self.assertTrue(-1.0 < delta_p < 0.0)
        # Delta difference = exp(-qT)
        self.assertAlmostEqual(delta_c - delta_p, 1.0, places=10)
        # Call and Put Gamma and Vega must be identical
        self.assertAlmostEqual(gamma_c, gamma_p, places=10)
        self.assertAlmostEqual(vega_c, vega_p, places=10)
        self.assertGreater(gamma_c, 0.0)
        self.assertGreater(vega_c, 0.0)


class TestCRREuropeanConvergence(unittest.TestCase):
    """Test suite for CRR European option convergence to Black-Scholes."""

    def test_european_call_convergence(self) -> None:
        """Test European Call convergence as N increases."""
        contract = OptionContract(
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        bs_price = BlackScholesBenchmark.price(contract)

        # As N increases, error must decrease
        err_25 = abs(CRRBinomialTree(contract, steps=25).price() - bs_price)
        err_100 = abs(CRRBinomialTree(contract, steps=100).price() - bs_price)
        err_500 = abs(CRRBinomialTree(contract, steps=500).price() - bs_price)

        self.assertLess(err_25, 0.20)
        self.assertLess(err_100, 0.05)
        self.assertLess(err_500, 0.015)

    def test_european_put_convergence(self) -> None:
        """Test European Put convergence as N increases."""
        contract = OptionContract(
            spot=100.0,
            strike=105.0,
            expiry=0.75,
            rate=0.04,
            volatility=0.25,
            dividend_yield=0.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        bs_price = BlackScholesBenchmark.price(contract)
        crr_price_500 = CRRBinomialTree(contract, steps=500).price()
        self.assertAlmostEqual(crr_price_500, bs_price, delta=0.02)

    def test_convergence_with_dividend_yield(self) -> None:
        """Verify convergence when underlying has continuous dividend yield q > 0."""
        contract = OptionContract(
            spot=100.0,
            strike=95.0,
            expiry=1.0,
            rate=0.06,
            volatility=0.30,
            dividend_yield=0.03,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        bs_price = BlackScholesBenchmark.price(contract)
        crr_price_400 = CRRBinomialTree(contract, steps=400).price()
        self.assertAlmostEqual(crr_price_400, bs_price, delta=0.02)


class TestAmericanOptionProperties(unittest.TestCase):
    """Test suite for American options, early exercise, and Merton theorems."""

    def test_american_price_never_less_than_european(self) -> None:
        """Theorem: An American option is always worth >= European equivalent."""
        for opt_type in (OptionType.CALL, OptionType.PUT):
            contract_eur = OptionContract(
                spot=100.0,
                strike=100.0,
                expiry=1.0,
                rate=0.05,
                volatility=0.20,
                option_type=opt_type,
                exercise_style=ExerciseStyle.EUROPEAN,
            )
            contract_ame = OptionContract(
                spot=100.0,
                strike=100.0,
                expiry=1.0,
                rate=0.05,
                volatility=0.20,
                option_type=opt_type,
                exercise_style=ExerciseStyle.AMERICAN,
            )
            p_eur = CRRBinomialTree(contract_eur, steps=100).price()
            p_ame = CRRBinomialTree(contract_ame, steps=100).price()
            self.assertGreaterEqual(p_ame, p_eur - 1e-12)

    def test_merton_theorem_non_dividend_call(self) -> None:
        """Merton (1973): For q=0, American Call price strictly equals European Call price."""
        contract_eur = OptionContract(
            spot=100.0,
            strike=90.0,
            expiry=0.5,
            rate=0.05,
            volatility=0.25,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        contract_ame = OptionContract(
            spot=100.0,
            strike=90.0,
            expiry=0.5,
            rate=0.05,
            volatility=0.25,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
        )
        p_eur = CRRBinomialTree(contract_eur, steps=100).price()
        p_ame = CRRBinomialTree(contract_ame, steps=100).price()

        # Premium must be zero within floating precision
        self.assertAlmostEqual(p_ame, p_eur, places=10)
        tree_ame = CRRBinomialTree(contract_ame, steps=100)
        self.assertAlmostEqual(tree_ame.early_exercise_premium(), 0.0, places=10)

    def test_deep_itm_american_put_early_exercise(self) -> None:
        """Deep in-the-money American Put with r > 0 must exhibit positive early exercise premium."""
        contract_eur = OptionContract(
            spot=70.0,
            strike=100.0,
            expiry=1.0,
            rate=0.08,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        contract_ame = OptionContract(
            spot=70.0,
            strike=100.0,
            expiry=1.0,
            rate=0.08,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.AMERICAN,
        )
        p_eur = CRRBinomialTree(contract_eur, steps=100).price()
        p_ame = CRRBinomialTree(contract_ame, steps=100).price()

        # American put must be strictly greater than European put
        self.assertGreater(p_ame, p_eur)
        early_prem = p_ame - p_eur
        self.assertGreater(early_prem, 0.50)  # Significant early exercise premium

    def test_american_call_with_dividend_early_exercise(self) -> None:
        """American Call on high dividend stock (q > r) can have positive early exercise premium."""
        contract_eur = OptionContract(
            spot=100.0,
            strike=80.0,
            expiry=1.0,
            rate=0.02,
            volatility=0.20,
            dividend_yield=0.08,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        contract_ame = OptionContract(
            spot=100.0,
            strike=80.0,
            expiry=1.0,
            rate=0.02,
            volatility=0.20,
            dividend_yield=0.08,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
        )
        p_eur = CRRBinomialTree(contract_eur, steps=100).price()
        p_ame = CRRBinomialTree(contract_ame, steps=100).price()

        self.assertGreater(p_ame, p_eur)


class TestGreeksEngine(unittest.TestCase):
    """Test suite for analytical and numerical finite-difference Greeks extraction."""

    def setUp(self) -> None:
        self.contract = OptionContract(
            spot=100.0,
            strike=100.0,
            expiry=0.5,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

    def test_lattice_greeks_convergence_to_bs(self) -> None:
        """Lattice Delta, Gamma, Theta must converge closely to Black-Scholes analytical values."""
        bs_delta = BlackScholesBenchmark.delta(self.contract)
        bs_gamma = BlackScholesBenchmark.gamma(self.contract)
        bs_theta = BlackScholesBenchmark.theta(self.contract)

        lattice_res = GreeksEngine.calculate_lattice_greeks(self.contract, steps=300)

        self.assertAlmostEqual(lattice_res.delta, bs_delta, delta=0.01)
        self.assertAlmostEqual(lattice_res.gamma, bs_gamma, delta=0.005)
        self.assertAlmostEqual(lattice_res.theta, bs_theta, delta=0.50)

    def test_bump_greeks_match_analytical_bs(self) -> None:
        """Bump Greeks must match Black-Scholes analytical sensitivities."""
        bs_delta = BlackScholesBenchmark.delta(self.contract)
        bs_gamma = BlackScholesBenchmark.gamma(self.contract)
        bs_vega = BlackScholesBenchmark.vega(self.contract)
        bs_rho = BlackScholesBenchmark.rho(self.contract)

        bump_res = GreeksEngine.calculate_bump_greeks(self.contract, steps=300)

        self.assertAlmostEqual(bump_res.delta, bs_delta, delta=0.01)
        self.assertAlmostEqual(bump_res.gamma, bs_gamma, delta=0.005)
        self.assertIsNotNone(bump_res.vega)
        self.assertIsNotNone(bump_res.rho)
        self.assertAlmostEqual(bump_res.vega, bs_vega, delta=0.5)
        self.assertAlmostEqual(bump_res.rho, bs_rho, delta=0.5)

    def test_greeks_comparison_report(self) -> None:
        """Verify structure of comparison report dictionary."""
        report = GreeksEngine.compare_greeks(self.contract, steps=50)
        self.assertIn("lattice", report)
        self.assertIn("numerical_bump", report)
        self.assertIn("analytical_bs", report)


class TestTreeStructureAndEquivalence(unittest.TestCase):
    """Test equivalence between fast 1D rolling array and full 2D tree construction."""

    def test_fast_buffer_matches_full_tree_price(self) -> None:
        """O(N) rolling buffer price must match root node of 2D tree exactly."""
        for style in (ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN):
            for opt_type in (OptionType.CALL, OptionType.PUT):
                contract = OptionContract(
                    spot=100.0,
                    strike=98.0,
                    expiry=0.5,
                    rate=0.04,
                    volatility=0.22,
                    dividend_yield=0.01,
                    option_type=opt_type,
                    exercise_style=style,
                )
                tree_model = CRRBinomialTree(contract, steps=40)
                price_1d = tree_model.price()
                tree_2d = tree_model.build_tree()
                price_2d = tree_2d[0][0].option_value

                self.assertAlmostEqual(price_1d, price_2d, places=12)

    def test_lattice_geometry(self) -> None:
        """Verify triangular lattice node dimensions and spot calculations."""
        contract = OptionContract(
            spot=50.0,
            strike=50.0,
            expiry=0.25,
            rate=0.05,
            volatility=0.30,
        )
        n = 5
        tree_model = CRRBinomialTree(contract, steps=n)
        tree = tree_model.build_tree()

        self.assertEqual(len(tree), n + 1)
        for i in range(n + 1):
            self.assertEqual(len(tree[i]), i + 1)
            for j in range(i + 1):
                expected_spot = 50.0 * (tree_model.u ** j) * (tree_model.d ** (i - j))
                self.assertAlmostEqual(tree[i][j].spot, expected_spot, places=10)


class TestValidationAndEdgeCases(unittest.TestCase):
    """Test validation boundaries, edge cases, and exception handling."""

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Inadmissible financial parameters must raise descriptive ValueErrors."""
        with self.assertRaises(ValueError):
            OptionContract(spot=-10.0, strike=100.0, expiry=1.0, rate=0.05, volatility=0.20)
        with self.assertRaises(ValueError):
            OptionContract(spot=100.0, strike=0.0, expiry=1.0, rate=0.05, volatility=0.20)
        with self.assertRaises(ValueError):
            OptionContract(spot=100.0, strike=100.0, expiry=-0.5, rate=0.05, volatility=0.20)
        with self.assertRaises(ValueError):
            OptionContract(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, volatility=-0.1)
        with self.assertRaises(ValueError):
            OptionContract(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, volatility=0.2, dividend_yield=-0.01)

    def test_invalid_tree_steps(self) -> None:
        """Step count < 1 must raise ValueError."""
        contract = OptionContract(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, volatility=0.2)
        with self.assertRaises(ValueError):
            CRRBinomialTree(contract, steps=0)
        with self.assertRaises(ValueError):
            CRRBinomialTree(contract, steps=-10)

    def test_visualizer_renders_without_error(self) -> None:
        """Tree visualizer methods should produce valid formatted strings."""
        contract = OptionContract(
            spot=100.0,
            strike=100.0,
            expiry=0.25,
            rate=0.05,
            volatility=0.20,
            exercise_style=ExerciseStyle.AMERICAN,
            option_type=OptionType.PUT,
        )
        tree = CRRBinomialTree(contract, steps=3)
        viz = TreeVisualizer(use_colors=False)

        summary = viz.render_summary(tree)
        lattice = viz.render_lattice(tree)
        table = viz.render_node_table(tree)
        boundary = viz.render_early_exercise_boundary(tree)

        self.assertIn("COX-ROSS-RUBINSTEIN", summary)
        self.assertIn("BINOMIAL RECOMBINING LATTICE", lattice)
        self.assertIn("DETAILED LATTICE NODE AUDIT", table)
        self.assertIn("AMERICAN OPTION EARLY EXERCISE FRONTIER", boundary)


if __name__ == "__main__":
    unittest.main()
