"""Interactive CLI Demonstration for the CRR Binomial Option Pricing Engine.

Showcases:
1. Multi-contract pricing & benchmark comparisons (European vs American, Calls vs Puts).
2. ASCII text-based tree visualization for small N (N=3 to 5) with early exercise flags.
3. High-precision numerical convergence analysis against the Black-Scholes benchmark.
4. Comprehensive Greeks engine comparison (Lattice vs Bump-and-Reprice vs Analytical).
5. Interactive custom contract pricing console.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Optional

from crr_engine.models import (
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


def print_banner(viz: TreeVisualizer) -> None:
    """Print ASCII logo banner and engine description."""
    line_border = "+" + "-" * 76 + "+"
    print(viz._c(line_border, viz.CYAN))
    print(viz._c("|", viz.CYAN) + viz._c("   ____ ____  ____    ____  _                             _       ", viz.BOLD + viz.CYAN) + "  " + viz._c("|", viz.CYAN))
    print(viz._c("|", viz.CYAN) + viz._c(r"  / ___|  _ \|  _ \  | __ )(_)_ __   ___  _ __ ___ (_) __ _| |      ", viz.BOLD + viz.CYAN) + "  " + viz._c("|", viz.CYAN))
    print(viz._c("|", viz.CYAN) + viz._c(r" | |   | |_) | |_) | |  _ \| | '_ \ / _ \| '_ ` _ \| |/ _` | |      ", viz.BOLD + viz.CYAN) + "  " + viz._c("|", viz.CYAN))
    print(viz._c("|", viz.CYAN) + viz._c(r" | |___|  _ <|  _ <  | |_) | | | | | (_) | | | | | | | (_| | |      ", viz.BOLD + viz.CYAN) + "  " + viz._c("|", viz.CYAN))
    print(viz._c("|", viz.CYAN) + viz._c(r"  \____|_| \_\_| \_\ |____/|_|_| |_|\___/|_| |_| |_|_|\__,_|_|      ", viz.BOLD + viz.CYAN) + "  " + viz._c("|", viz.CYAN))
    print(viz._c("|", viz.CYAN) + viz._c("  PRODUCTION-GRADE OPTION PRICING & GREEKS ENGINE (ZERO-DEPENDENCY) ", viz.BOLD + viz.YELLOW) + "  " + viz._c("|", viz.CYAN))
    print(viz._c(line_border, viz.CYAN))


def run_demo_contracts(viz: TreeVisualizer) -> None:
    """Run curated showcase contracts illustrating European/American dynamics."""
    print(viz._c("\n=== [1] SAMPLE CONTRACTS SHOWCASE ===", viz.BOLD + viz.CYAN))

    contracts = [
        (
            "1. At-The-Money European Call (Benchmark)",
            OptionContract(
                spot=100.0,
                strike=100.0,
                expiry=1.0,
                rate=0.05,
                volatility=0.20,
                dividend_yield=0.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN,
            ),
            200,
        ),
        (
            "2. Deep In-The-Money American Put (Early Exercise Active)",
            OptionContract(
                spot=70.0,
                strike=100.0,
                expiry=1.0,
                rate=0.08,
                volatility=0.20,
                dividend_yield=0.0,
                option_type=OptionType.PUT,
                exercise_style=ExerciseStyle.AMERICAN,
            ),
            200,
        ),
        (
            "3. Dividend-Paying European Put (q = 3.5%)",
            OptionContract(
                spot=105.0,
                strike=100.0,
                expiry=0.75,
                rate=0.04,
                volatility=0.25,
                dividend_yield=0.035,
                option_type=OptionType.PUT,
                exercise_style=ExerciseStyle.EUROPEAN,
            ),
            200,
        ),
        (
            "4. American Call on High-Dividend Stock (q = 7%, r = 2%)",
            OptionContract(
                spot=100.0,
                strike=80.0,
                expiry=0.5,
                rate=0.02,
                volatility=0.22,
                dividend_yield=0.07,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.AMERICAN,
            ),
            200,
        ),
    ]

    print(
        f"{'Contract Name':<42} | {'Style':<4} | {'Type':<4} | {'CRR Price':>10} | "
        f"{'BS Price':>10} | {'Early Prem':>10} | {'Time (ms)':>9}"
    )
    print("-" * 105)

    for name, contract, steps in contracts:
        t0 = time.perf_counter()
        model = CRRBinomialTree(contract, steps=steps)
        crr_val = model.price()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if contract.is_european():
            bs_val = BlackScholesBenchmark.price(contract)
            bs_str = f"{bs_val:10.4f}"
            early_prem_str = "    N/A   "
        else:
            bs_val_euro = BlackScholesBenchmark.price(contract)
            bs_str = f"({bs_val_euro:8.4f})*"
            early_prem = model.early_exercise_premium()
            early_prem_str = f"{early_prem:10.4f}"

        print(
            f"{name:<42} | {contract.exercise_style.value[:3].upper():<4} | "
            f"{contract.option_type.value.upper():<4} | "
            f"{viz._c(f'{crr_val:10.4f}', viz.GREEN)} | {bs_str:>10} | "
            f"{viz._c(early_prem_str, viz.MAGENTA if early_prem_str.strip() != 'N/A' else viz.DIM)} | "
            f"{elapsed_ms:9.3f}"
        )

    print("-" * 105)
    print(viz._c("* Note: Parenthesized BS values represent the European counterpart for reference.\n", viz.DIM))


def run_convergence_analysis(viz: TreeVisualizer, contract: Optional[OptionContract] = None) -> None:
    """Run CRR step convergence analysis against Black-Scholes benchmark."""
    if contract is None:
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

    print(viz._c("=== [2] CONVERGENCE ANALYSIS: CRR vs BLACK-SCHOLES ===", viz.BOLD + viz.CYAN))
    print(
        f"Contract: S0={contract.spot}, K={contract.strike}, T={contract.expiry}yr, "
        f"r={contract.rate*100}%, sigma={contract.volatility*100}%, q={contract.dividend_yield*100}%"
    )

    bs_price = BlackScholesBenchmark.price(contract)
    print(f"Exact Black-Scholes Benchmark Price: {viz._c(f'{bs_price:.6f}', viz.BOLD + viz.GREEN)}\n")

    steps_list = [10, 25, 50, 100, 200, 400, 800, 1500]

    print(
        f"{'Steps (N)':>10} | {'dt (days)':>10} | {'u':>9} | {'p':>8} | "
        f"{'CRR Price':>11} | {'Abs Error':>11} | {'Rel Err (bps)':>14} | {'Elapsed (ms)':>12}"
    )
    print("-" * 96)

    for n in steps_list:
        t0 = time.perf_counter()
        tree = CRRBinomialTree(contract, steps=n)
        crr_price = tree.price()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        abs_err = abs(crr_price - bs_price)
        rel_err_bps = (abs_err / bs_price) * 10000.0
        dt_days = tree.dt * 365.0

        print(
            f"{n:10d} | {dt_days:10.2f} | {tree.u:9.5f} | {tree.p:8.5f} | "
            f"{crr_price:11.6f} | {abs_err:11.6f} | {rel_err_bps:14.2f} | {elapsed_ms:12.3f}"
        )

    print("-" * 96)
    print(viz._c("Observation: CRR price exhibits O(1/N) oscillation and converges monotonically with step parity.\n", viz.DIM))


def run_greeks_comparison(viz: TreeVisualizer, contract: Optional[OptionContract] = None, steps: int = 250) -> None:
    """Compare Lattice Greeks, Numerical Bump Greeks, and Analytical Black-Scholes Greeks."""
    if contract is None:
        contract = OptionContract(
            spot=100.0,
            strike=100.0,
            expiry=0.5,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.0,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

    print(viz._c("=== [3] GREEKS EXTRACTION ENGINE COMPARISON ===", viz.BOLD + viz.CYAN))
    print(
        f"Contract: S0={contract.spot}, K={contract.strike}, T={contract.expiry}yr, "
        f"r={contract.rate*100}%, sigma={contract.volatility*100}%, Steps={steps}\n"
    )

    # 1. Lattice Greeks
    t0 = time.perf_counter()
    lat = GreeksEngine.calculate_lattice_greeks(contract, steps=steps)
    t_lat = (time.perf_counter() - t0) * 1000.0

    # 2. Bump Greeks
    t0 = time.perf_counter()
    bmp = GreeksEngine.calculate_bump_greeks(contract, steps=steps)
    t_bmp = (time.perf_counter() - t0) * 1000.0

    # 3. Analytical BS
    t0 = time.perf_counter()
    ana = GreeksEngine.calculate_analytical_bs_greeks(contract)
    t_ana = (time.perf_counter() - t0) * 1000.0

    print(
        f"{'Greek':<16} | {'Lattice Extraction':>18} | {'Central Diff Bump':>18} | "
        f"{'Analytical BS':>18} | {'Lat vs BS Diff':>16}"
    )
    print("-" * 96)

    def row(label: str, v_lat: Optional[float], v_bmp: Optional[float], v_ana: Optional[float]) -> str:
        s_lat = f"{v_lat:18.6f}" if v_lat is not None else "        N/A       "
        s_bmp = f"{v_bmp:18.6f}" if v_bmp is not None else "        N/A       "
        s_ana = f"{v_ana:18.6f}" if v_ana is not None else "        N/A       "
        diff = f"{abs(v_lat - v_ana):16.6f}" if (v_lat is not None and v_ana is not None) else "       N/A      "
        return f"{label:<16} | {s_lat} | {s_bmp} | {s_ana} | {diff}"

    print(row("Delta (dV/dS)", lat.delta, bmp.delta, ana.delta))
    print(row("Gamma (d2V/dS2)", lat.gamma, bmp.gamma, ana.gamma))
    print(row("Theta (annual)", lat.theta, bmp.theta, ana.theta))
    print(row("Theta (daily)", lat.theta_daily, bmp.theta_daily, ana.theta_daily))
    print(row("Vega (per 1%)", (lat.vega / 100.0) if lat.vega else None, (bmp.vega / 100.0) if bmp.vega else None, (ana.vega / 100.0) if ana.vega else None))
    print(row("Rho (per 1%)", (lat.rho / 100.0) if lat.rho else None, (bmp.rho / 100.0) if bmp.rho else None, (ana.rho / 100.0) if ana.rho else None))

    print("-" * 96)
    print(
        f"Execution Timings: Lattice = {t_lat:.3f} ms | Bump (6 trees) = {t_bmp:.3f} ms | "
        f"Analytical BS = {t_ana:.3f} ms\n"
    )


def run_tree_visualizer_demo(viz: TreeVisualizer) -> None:
    """Render terminal ASCII binomial tree for N=3 and N=4."""
    print(viz._c("=== [4] ASCII TERMINAL TREE VISUALIZATION (N = 3) ===", viz.BOLD + viz.CYAN))

    # American Put with active early exercise
    contract = OptionContract(
        spot=100.0,
        strike=100.0,
        expiry=0.30,
        rate=0.06,
        volatility=0.25,
        dividend_yield=0.0,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    tree_model = CRRBinomialTree(contract, steps=3)

    print(viz.render_summary(tree_model))
    print()
    print(viz.render_lattice(tree_model))
    print()
    print(viz.render_early_exercise_boundary(tree_model))
    print()
    print(viz.render_node_table(tree_model))
    print()


def interactive_menu(viz: TreeVisualizer) -> None:
    """Prompt-driven interactive session for custom contract experimentation."""
    print_banner(viz)

    while True:
        print(viz._c("\nSELECT WORKFLOW MODE:", viz.BOLD + viz.YELLOW))
        print("  1. Run Standard Showcase Demo (European & American Contracts)")
        print("  2. Run Convergence Analysis Table (N = 10 to 1,500)")
        print("  3. Compare Greeks Extraction (Lattice vs Bump vs Black-Scholes)")
        print("  4. Render ASCII Binomial Tree Visualizer (N = 3)")
        print("  5. Price Custom User Contract (Interactive Input)")
        print("  6. Run Full Unit Test Suite")
        print("  0. Exit")

        try:
            choice = input(viz._c("\nEnter choice [0-6]: ", viz.BOLD)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if choice == "1":
            run_demo_contracts(viz)
        elif choice == "2":
            run_convergence_analysis(viz)
        elif choice == "3":
            run_greeks_comparison(viz)
        elif choice == "4":
            run_tree_visualizer_demo(viz)
        elif choice == "5":
            prompt_custom_contract(viz)
        elif choice == "6":
            run_test_suite()
        elif choice in ("0", "q", "exit"):
            print("Goodbye.")
            break
        else:
            print(viz._c("Invalid option. Please enter 0-6.", viz.RED))


def prompt_custom_contract(viz: TreeVisualizer) -> None:
    """Interactive prompt to input custom parameters and compute valuation & Greeks."""
    print(viz._c("\n--- Enter Contract Parameters ---", viz.BOLD + viz.CYAN))

    def get_float(prompt: str, default: float) -> float:
        val = input(f"{prompt} [{default}]: ").strip()
        return float(val) if val else default

    def get_int(prompt: str, default: int) -> int:
        val = input(f"{prompt} [{default}]: ").strip()
        return int(val) if val else default

    try:
        spot = get_float("Underlying Spot Price (S0)", 100.0)
        strike = get_float("Strike Price (K)", 100.0)
        expiry = get_float("Time to Expiration in years (T)", 1.0)
        rate = get_float("Annual Risk-Free Rate (r, e.g. 0.05 for 5%)", 0.05)
        vol = get_float("Annual Volatility (sigma, e.g. 0.20 for 20%)", 0.20)
        div = get_float("Annual Dividend Yield (q, e.g. 0.0 for none)", 0.0)

        t_input = input("Option Type (C for Call, P for Put) [C]: ").strip().upper()
        opt_type = OptionType.PUT if t_input == "P" else OptionType.CALL

        s_input = input("Exercise Style (E for European, A for American) [E]: ").strip().upper()
        opt_style = ExerciseStyle.AMERICAN if s_input == "A" else ExerciseStyle.EUROPEAN

        steps = get_int("Binomial Discretization Steps (N)", 100)

        contract = OptionContract(
            spot=spot,
            strike=strike,
            expiry=expiry,
            rate=rate,
            volatility=vol,
            dividend_yield=div,
            option_type=opt_type,
            exercise_style=opt_style,
        )

        tree = CRRBinomialTree(contract, steps=steps)
        price = tree.price()

        print("\n" + viz.render_summary(tree))

        # Greeks
        if steps >= 2:
            lat_greeks = GreeksEngine.calculate_lattice_greeks(contract, steps=steps)
            print(viz._c("\nLattice Greeks:", viz.BOLD + viz.CYAN))
            print(f"  Delta: {lat_greeks.delta:10.5f} | Gamma: {lat_greeks.gamma:10.5f} | Theta (daily): {lat_greeks.theta_daily:10.5f}")

        if contract.is_european():
            bs_price = BlackScholesBenchmark.price(contract)
            diff = abs(price - bs_price)
            print(f"  Black-Scholes Benchmark : {bs_price:10.5f} (Diff: {diff:.6f})")

        if steps <= 6:
            view_tree = input("\nDisplay ASCII tree lattice? (y/n) [y]: ").strip().lower()
            if view_tree in ("", "y", "yes"):
                print("\n" + viz.render_lattice(tree))

    except Exception as exc:
        print(viz._c(f"\nError processing contract: {exc}", viz.RED))


def run_test_suite() -> None:
    """Execute unit test suite directly from CLI."""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def main() -> None:
    """Entrypoint with CLI argument dispatch."""
    parser = argparse.ArgumentParser(
        description="Cox-Ross-Rubinstein (CRR) Binomial Option Pricing & Greeks Engine."
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo", "convergence", "greeks", "visualize", "test"],
        default="interactive",
        help="Execution mode (default: interactive)",
    )
    parser.add_argument("--spot", type=float, default=100.0, help="Spot price S0")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price K")
    parser.add_argument("--expiry", type=float, default=1.0, help="Expiry T in years")
    parser.add_argument("--rate", type=float, default=0.05, help="Annual risk-free rate r")
    parser.add_argument("--vol", type=float, default=0.20, help="Annual volatility sigma")
    parser.add_argument("--div", type=float, default=0.0, help="Annual dividend yield q")
    parser.add_argument(
        "--type", choices=["call", "put"], default="call", help="Option type (call/put)"
    )
    parser.add_argument(
        "--style",
        choices=["european", "american"],
        default="european",
        help="Exercise style (european/american)",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of tree steps N")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI terminal colors")

    args = parser.parse_args()
    viz = TreeVisualizer(use_colors=False if args.no_color else None)

    if args.mode == "demo":
        print_banner(viz)
        run_demo_contracts(viz)
    elif args.mode == "convergence":
        contract = OptionContract(
            spot=args.spot,
            strike=args.strike,
            expiry=args.expiry,
            rate=args.rate,
            volatility=args.vol,
            dividend_yield=args.div,
            option_type=OptionType(args.type),
            exercise_style=ExerciseStyle(args.style),
        )
        run_convergence_analysis(viz, contract)
    elif args.mode == "greeks":
        contract = OptionContract(
            spot=args.spot,
            strike=args.strike,
            expiry=args.expiry,
            rate=args.rate,
            volatility=args.vol,
            dividend_yield=args.div,
            option_type=OptionType(args.type),
            exercise_style=ExerciseStyle(args.style),
        )
        run_greeks_comparison(viz, contract, steps=args.steps)
    elif args.mode == "visualize":
        contract = OptionContract(
            spot=args.spot,
            strike=args.strike,
            expiry=args.expiry,
            rate=args.rate,
            volatility=args.vol,
            dividend_yield=args.div,
            option_type=OptionType(args.type),
            exercise_style=ExerciseStyle(args.style),
        )
        tree = CRRBinomialTree(contract, steps=min(args.steps, 5))
        print(viz.render_summary(tree))
        print()
        print(viz.render_lattice(tree))
    elif args.mode == "test":
        run_test_suite()
    else:
        interactive_menu(viz)


if __name__ == "__main__":
    main()
