# Binomial Option Pricing & Greeks Engine (CRR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#)
[![Tests: Passing](https://img.shields.io/badge/tests-20%20passed-success.svg)](#)

A high-performance, production-grade, **zero-dependency** quantitative finance engine implementing the **Cox-Ross-Rubinstein (1979) Binomial Option Pricing Model**, early exercise boundary detection, finite-difference Greeks extraction, and terminal ASCII lattice visualization using **only the Python standard library**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Mathematical Foundation](#mathematical-foundation)
  - [CRR Parameterization](#crr-parameterization)
  - [Risk-Neutral Expectation & Backward Induction](#risk-neutral-expectation--backward-induction)
  - [American Early Exercise Dynamics](#american-early-exercise-dynamics)
  - [Lattice Greeks Extraction](#lattice-greeks-extraction)
- [Project Architecture](#project-architecture)
- [Installation & Quickstart](#installation--quickstart)
- [Python API Usage](#python-api-usage)
  - [1. Pricing European & American Options](#1-pricing-european--american-options)
  - [2. Extracting Option Greeks](#2-extracting-option-greeks)
  - [3. Terminal Lattice Visualization](#3-terminal-lattice-visualization)
- [Interactive CLI Reference](#interactive-cli-reference)
- [Convergence & Benchmark Analysis](#convergence--benchmark-analysis)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [License](#license)

---

## Overview

The **Cox-Ross-Rubinstein (CRR) model** is a cornerstone of quantitative options pricing. By discretizing continuous asset paths into a recombining binomial lattice, it computes arbitrage-free fair values for both **European** and **American** contingent claims.

While European options possess closed-form solutions under the Black-Scholes-Merton model, **American options**—which permit early exercise at any stopping time prior to expiration—lack general analytical formulas. This engine provides a numerically stable, memory-efficient implementation featuring:
- **$O(N)$ Space Backward Induction** for high-throughput batch pricing.
- **$O(N^2)$ Explicit Lattice Mesh** for node inspection, early exercise frontiers, and zero-recomputation Greek extraction.
- **Strict Zero-Dependency Design**: Powered entirely by Python's built-in `math` module (including Gaussian integration via `math.erf`).

---

## Key Features

- **European & American Options**: Handles Calls and Puts with continuous dividend yields $q \ge 0$.
- **Early Exercise Boundary Engine**: Automatically identifies the critical asset price frontier $S^*(t)$ where American early exercise dominates continuation value.
- **Dual Greeks Framework**:
  - **Lattice-Extracted Greeks**: Direct extraction of Delta ($\Delta$), Gamma ($\Gamma$), and Theta ($\Theta$) from steps $0, 1, 2$ of the existing lattice without additional tree valuations.
  - **Bump-and-Reprice Greeks**: Central finite-difference perturbations for Delta, Gamma, Theta, Vega ($\nu$), and Rho ($\rho$).
- **ASCII Terminal Tree Visualizer**: High-clarity terminal tree rendering for small $N$ ($N \in [1, 6]$) highlighting asset prices, option values, and optimal exercise nodes (`*EX*`).
- **Analytical Black-Scholes Benchmark**: Exact benchmarks implemented using standard-library Gaussian cumulative distributions for continuous convergence tracking.
- **Type-Safe & Modular**: Adheres strictly to PEP 8 standards with 100% type hinting and zero external dependencies.

---

## Mathematical Foundation

### CRR Parameterization

Given an underlying asset with spot price $S_0$, strike $K$, annualized risk-free rate $r$, continuous dividend yield $q$, annualized volatility $\sigma$, and time to expiry $T$:

Dividing the maturity $T$ into $N$ discrete steps of length $\Delta t = T / N$:

$$\Delta t = \frac{T}{N}$$

$$u = e^{\sigma \sqrt{\Delta t}}, \quad d = e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}$$

The cost-of-carry growth factor $a$ and discount factor $D$ are:

$$a = e^{(r - q)\Delta t}, \quad D = e^{-r \Delta t}$$

The risk-neutral up-move probability $p$ is derived from the martingale condition:

$$p = \frac{a - d}{u - d}, \quad 1 - p = \frac{u - a}{u - d}$$

**No-Arbitrage Condition**:
The model guarantees absence of dynamic arbitrage if and only if:

$$d < e^{(r - q)\Delta t} < u \iff 0 < p < 1$$

At time step $i \in \{0, \dots, N\}$ and state $j \in \{0, \dots, i\}$ (representing $j$ up-moves):

$$S_{i, j} = S_0 \cdot u^j \cdot d^{i - j} = S_0 \cdot u^{2j - i}$$

---

### Risk-Neutral Expectation & Backward Induction

At terminal expiration step $N$:

$$V_{N, j} = \begin{cases} \max(S_{N, j} - K, 0) & \text{for Call} \\ \max(K - S_{N, j}, 0) & \text{for Put} \end{cases}$$

For time steps $i = N - 1$ down to $0$:

1. **Expected Continuation Value**:
   $$C_{i, j} = e^{-r \Delta t} \Big[ p \cdot V_{i+1, j+1} + (1 - p) \cdot V_{i+1, j} \Big]$$

2. **Node Valuation**:
   $$\text{European: } V_{i, j} = C_{i, j}$$
   $$\text{American: } V_{i, j} = \max\Big(C_{i, j}, \, \text{Intrinsic}(S_{i, j})\Big)$$

---

### American Early Exercise Dynamics

- **Merton's Theorem (1973)**: For an American call option on a non-dividend-paying stock ($q = 0$), early exercise is never optimal prior to maturity. Hence:
  $$V_{\text{American Call}}(q=0) \equiv V_{\text{European Call}}$$
- **American Puts**: A deep in-the-money American put with $r > 0$ carries a positive early exercise premium due to the time-value of money on cash proceeds:
  $$\text{Early Exercise Premium} = V_{\text{American}} - V_{\text{European}} \ge 0$$

---

### Lattice Greeks Extraction

Greeks are extracted directly from the early nodes of the lattice mesh:

| Greek | Lattice Formulation | Mathematical Meaning |
| :--- | :--- | :--- |
| **Delta ($\Delta$)** | $\Delta = \frac{V_{1, 1} - V_{1, 0}}{S_{1, 1} - S_{1, 0}}$ | First derivative with respect to underlying spot ($dV/dS$) |
| **Gamma ($\Gamma$)** | $\Gamma = \frac{\Delta_u - \Delta_d}{\frac{1}{2}(S_{2, 2} - S_{2, 0})}$ | Curvature / second derivative ($d^2V/dS^2$) |
| **Theta ($\Theta$)** | $\Theta = \frac{V_{2, 1} - V_{0, 0}}{2 \Delta t}$ | Time decay holding spot constant ($S_{2, 1} = S_0 u d = S_0$) |

---

## Project Architecture

```text
binomial-pricing-engine/
├── crr_engine/
│   ├── __init__.py         # Public library API & exports
│   ├── models.py           # Core CRR binomial pricing engine & Black-Scholes benchmark
│   ├── greeks.py           # Lattice-based & numerical finite-difference Greeks engine
│   └── visualizer.py       # Terminal ASCII lattice renderer & early exercise inspector
├── tests/
│   ├── __init__.py
│   └── test_pricing.py     # 20 rigorous unit tests validating convergence & Greeks
├── main.py                 # Interactive CLI demo, convergence suites, and visualizer
├── .gitignore              # Standard Python gitignore
├── LICENSE                 # MIT License
└── README.md               # Technical documentation
```

---

## Installation & Quickstart

### Zero External Dependencies
This engine requires **only Python 3.8+** with the standard library. No `pip install` or external packages (NumPy, SciPy, Pandas) are required.

```bash
# Clone the repository
git clone https://github.com/Bhargav221287/binomial-pricing-engine.git
cd binomial-pricing-engine

# Verify installation by running test suite
python -m unittest discover -s tests -v
```

---

## Python API Usage

### 1. Pricing European & American Options

```python
from crr_engine import OptionContract, OptionType, ExerciseStyle, CRRBinomialTree, BlackScholesBenchmark

# Define an American Put contract
contract = OptionContract(
    spot=100.0,
    strike=100.0,
    expiry=1.0,          # 1 year
    rate=0.05,           # 5% annual risk-free rate
    volatility=0.20,     # 20% annualized volatility
    dividend_yield=0.0,  # No dividends
    option_type=OptionType.PUT,
    exercise_style=ExerciseStyle.AMERICAN,
)

# Initialize CRR engine with N = 200 steps
tree = CRRBinomialTree(contract, steps=200)

american_price = tree.price()
early_prem = tree.early_exercise_premium()

print(f"American Put Fair Value : ${american_price:.4f}")
print(f"Early Exercise Premium  : ${early_prem:.4f}")
```

### 2. Extracting Option Greeks

```python
from crr_engine import GreeksEngine

# Extract lattice Greeks (zero extra tree computations)
lattice_greeks = GreeksEngine.calculate_lattice_greeks(contract, steps=200)
print(f"Delta: {lattice_greeks.delta:.5f}")
print(f"Gamma: {lattice_greeks.gamma:.5f}")
print(f"Daily Theta: ${lattice_greeks.theta_daily:.5f}/day")

# Extract full sensitivity suite via Central Finite Difference perturbations
bump_greeks = GreeksEngine.calculate_bump_greeks(contract, steps=200)
print(f"Vega (per 1% vol) : ${bump_greeks.vega / 100:.5f}")
print(f"Rho (per 1% rate) : ${bump_greeks.rho / 100:.5f}")
```

### 3. Terminal Lattice Visualization

```python
from crr_engine import TreeVisualizer

small_tree = CRRBinomialTree(contract, steps=3)
viz = TreeVisualizer()

# Print contract summary
print(viz.render_summary(small_tree))

# Print ASCII lattice
print(viz.render_lattice(small_tree))

# Print early exercise frontier
print(viz.render_early_exercise_boundary(small_tree))
```

---

## Interactive CLI Reference

The engine includes a full-featured CLI via `main.py`:

```bash
# Launch interactive terminal console
python main.py

# Run standard showcase demo of curated contracts
python main.py --mode demo

# Run step convergence analysis (N = 10 to 1,500)
python main.py --mode convergence --spot 100 --strike 100 --expiry 1.0 --rate 0.05 --vol 0.20

# Compare Greeks across Lattice, Bump, and Analytical Black-Scholes
python main.py --mode greeks --steps 250

# Render ASCII lattice tree for N = 3
python main.py --mode visualize --steps 3 --type put --style american
```

### Sample ASCII Tree Output:
```text
  BINOMIAL RECOMBINING LATTICE
  [S: Asset Price | V: Option Value | *EX*: Early Exercised]
-----------------------------------------------------------------------------
 Step 0 (t=0.00)  |  Step 1 (t=0.33)  |  Step 2 (t=0.67)  |  Step 3 (t=1.00) 
-----------------------------------------------------------------------------
                                                            S:141.40    
                                                           [V:  0.00 ]  

                                          S:125.98                      
                                         [V:  0.00 ]                    

                        S:112.24                          S:112.24    
                       [V:  2.20 ]                       [V:  0.00 ]  

      S:100.00                          S:100.00                      
     [V:  6.50 ]                       [V:  4.89 ]                    

                        S: 89.09                          S: 89.09    
                       [V: 11.87 ]                       [V: 10.91 ]  

                                          S: 79.38                      
                                         [V:20.62* ]                    

                                                            S: 70.72    
                                                           [V: 29.28 ]  
-----------------------------------------------------------------------------
```

---

## Convergence & Benchmark Analysis

At-The-Money European Call ($S=100, K=100, T=1.0, r=5\%, \sigma=20\%$):
Exact Black-Scholes Benchmark Price: **10.450584**

| Steps ($N$) | Step Size $\Delta t$ | Up Factor ($u$) | Prob ($p$) | CRR Price | Abs Error | Rel Error | Elapsed (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10** | 36.50 days | 1.06529 | 0.52379 | 10.253409 | 0.197175 | 188.67 bps | 0.028 ms |
| **25** | 14.60 days | 1.04081 | 0.51502 | 10.520966 | 0.070382 | 67.35 bps | 0.038 ms |
| **50** | 7.30 days | 1.02869 | 0.51061 | 10.410692 | 0.039892 | 38.17 bps | 0.107 ms |
| **100** | 3.65 days | 1.02020 | 0.50750 | 10.430612 | 0.019972 | 19.11 bps | 0.327 ms |
| **200** | 1.82 days | 1.01424 | 0.50530 | 10.440591 | 0.009992 | 9.56 bps | 1.245 ms |
| **400** | 0.91 days | 1.01005 | 0.50375 | 10.445586 | 0.004998 | 4.78 bps | 4.755 ms |
| **800** | 0.46 days | 1.00710 | 0.50265 | 10.448084 | 0.002499 | 2.39 bps | 20.096 ms |
| **1500** | 0.24 days | 1.00518 | 0.50194 | 10.449251 | 0.001333 | 1.28 bps | 70.742 ms |

---

## Testing & Quality Assurance

The project includes 20 comprehensive unit tests using Python's native `unittest` framework:

```bash
python -m unittest discover -s tests -v
```

### Coverage Highlights:
- **Mathematical Benchmark**: Normal CDF/PDF properties, analytical Put-Call Parity, and verified textbook literature values (Hull).
- **Asymptotic Convergence**: Verified error bounds for Calls and Puts across steps $N \in [25, 1000]$.
- **Arbitrage Bounds**: Strict validation of American $\ge$ European pricing, Merton's non-exercise theorem on non-dividend calls, and positive early exercise premiums on deep ITM puts.
- **Sensitivity Validation**: Lattice and central difference Greeks checked against Black-Scholes analytical values.
- **Algorithmic Equivalence**: Verified exact numerical identity between the $O(N)$ rolling buffer and full 2D lattice tree.

---

## License

This project is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 P Sai Bhargav.
