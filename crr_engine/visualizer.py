"""ASCII text-based terminal tree renderer for small N (N=1 to 6).

Renders financial lattice geometry, node valuation details, early exercise
decisions, and summary cards in standard terminal environments without external dependencies.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from crr_engine.models import (
    BinomialNode,
    CRRBinomialTree,
    OptionContract,
)


class TreeVisualizer:
    """Renderer for displaying Cox-Ross-Rubinstein binomial lattices in ASCII text."""

    # ANSI Color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    DIM = "\033[2m"

    def __init__(self, use_colors: Optional[bool] = None) -> None:
        """Initialize visualizer.

        Args:
            use_colors: Force enable/disable ANSI colors. If None, auto-detected.
        """
        if use_colors is None:
            self.use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        else:
            self.use_colors = use_colors

    def _c(self, text: str, color_code: str) -> str:
        """Apply ANSI color styling if enabled."""
        if not self.use_colors:
            return text
        return f"{color_code}{text}{self.RESET}"

    def render_summary(self, model: CRRBinomialTree) -> str:
        """Generate a formatted ASCII summary card of contract parameters and CRR factors."""
        contract = model.contract
        price = model.price()
        early_prem = model.early_exercise_premium() if contract.is_american() else 0.0

        lines: List[str] = [
            self._c("=" * 72, self.CYAN),
            self._c("  COX-ROSS-RUBINSTEIN (CRR) BINOMIAL VALUATION SUMMARY", self.BOLD + self.CYAN),
            self._c("=" * 72, self.CYAN),
            f"  Contract Type        : {contract.exercise_style.value.upper()} {contract.option_type.value.upper()}",
            f"  Spot Price (S0)      : {contract.spot:10.4f}      Strike Price (K)     : {contract.strike:10.4f}",
            f"  Time to Expiry (T)   : {contract.expiry:10.4f} yrs  Annual Risk-Free (r) : {contract.rate * 100:9.2f}%",
            f"  Volatility (sigma)   : {contract.volatility * 100:9.2f}%     Dividend Yield (q)   : {contract.dividend_yield * 100:9.2f}%",
            self._c("-" * 72, self.DIM),
            f"  Discretization Steps : {model.steps:10d}      Time Step (dt)       : {model.dt:10.5f} yrs",
            f"  Up Factor (u)        : {model.u:10.5f}      Down Factor (d)      : {model.d:10.5f}",
            f"  Growth Factor (e^rdt): {model.growth:10.5f}      Discount Factor      : {model.discount:10.5f}",
            f"  Risk-Neutral Prob (p): {model.p:10.5f}      Down Prob (1-p)      : {model.q_prob:10.5f}",
            self._c("-" * 72, self.DIM),
            f"  {self._c('OPTION FAIR VALUE', self.BOLD + self.GREEN)}  : {self._c(f'{price:10.4f}', self.BOLD + self.GREEN)}",
        ]

        if contract.is_american():
            prem_str = f"{early_prem:10.4f}"
            lines.append(f"  Early Exercise Prem  : {self._c(prem_str, self.MAGENTA)}")

        lines.append(self._c("=" * 72, self.CYAN))
        return "\n".join(lines)

    def render_lattice(self, model: CRRBinomialTree) -> str:
        """Render a vertical aligned 2D ASCII binomial lattice grid.

        Best suited for N in range 1 to 6.
        Displays rows ordered by net up moves from top (highest S) to bottom (lowest S).
        Each node prints:
            S: underlying price
            V: option value
            *EX* flag if American early exercise is optimal.

        Args:
            model: CRRBinomialTree instance.

        Returns:
            Multi-line string representation of the tree lattice.
        """
        n = model.steps
        if n > 7:
            return (
                f"[Visualizer Notice] Full lattice visualization is optimized for N <= 6. "
                f"Current tree has N = {n} steps (total nodes = {(n + 1) * (n + 2) // 2}). "
                f"Use render_node_table() or run with N in [2..6] for graphical display."
            )

        tree = model.build_tree()
        col_width = 17

        # 1. Header with time steps
        header_parts: List[str] = []
        for i in range(n + 1):
            t_curr = i * model.dt
            header_parts.append(f"Step {i} (t={t_curr:.2f})".center(col_width))
        header_str = " | ".join(header_parts)
        divider = "-" * len(header_str)

        # 2. Lattice Grid Construction
        # A tree of step N has rows corresponding to net up-moves: 2*j - i in [-N, N]
        # Total possible row levels = 2*N + 1
        lines: List[str] = [
            self._c("  BINOMIAL RECOMBINING LATTICE", self.BOLD),
            self._c("  [S: Asset Price | V: Option Value | *EX*: Early Exercised]", self.DIM),
            divider,
            self._c(header_str, self.CYAN + self.BOLD),
            divider,
        ]

        # Row levels range from +N down to -N
        for row_level in range(n, -n - 1, -1):
            # Check if any step i has a node with 2*j - i == row_level
            has_node_in_row = False
            for i in range(n + 1):
                # 2*j - i == row_level  =>  2*j == row_level + i
                if (row_level + i) % 2 == 0:
                    j = (row_level + i) // 2
                    if 0 <= j <= i:
                        has_node_in_row = True
                        break

            if not has_node_in_row:
                continue

            # Two visual sub-lines per row: Line 1 for Spot, Line 2 for Value
            spot_cols: List[str] = []
            val_cols: List[str] = []

            for i in range(n + 1):
                if (row_level + i) % 2 == 0:
                    j = (row_level + i) // 2
                    if 0 <= j <= i:
                        node = tree[i][j]
                        spot_str = f"S:{node.spot:6.2f}"
                        if node.is_early_exercised:
                            val_str = f"V:{node.option_value:5.2f}*"
                            val_fmt = self._c(f"[{val_str:^9}]", self.RED + self.BOLD)
                        else:
                            val_str = f"V:{node.option_value:6.2f}"
                            val_fmt = self._c(f"[{val_str:^9}]", self.GREEN)

                        spot_cols.append(self._c(f"{spot_str:^15}", self.YELLOW))
                        val_cols.append(f"{val_fmt:^15}")
                        continue

                # Empty placeholder column
                spot_cols.append(" " * col_width)
                val_cols.append(" " * col_width)

            lines.append("   " + " ".join(spot_cols))
            lines.append("   " + " ".join(val_cols))
            lines.append("")  # breathing space between rows

        lines.append(divider)
        return "\n".join(lines)

    def render_node_table(self, model: CRRBinomialTree) -> str:
        """Render a tabular inspection list of all nodes with continuation & intrinsic values."""
        tree = model.build_tree()
        lines: List[str] = [
            self._c("  DETAILED LATTICE NODE AUDIT", self.BOLD + self.CYAN),
            self._c("-" * 84, self.DIM),
            f"  {'Step':>4} | {'State':>5} | {'Net':>4} | {'Asset Spot':>10} | {'Option Val':>10} | "
            f"{'Continuation':>12} | {'Intrinsic':>9} | {'Action':>10}",
            self._c("-" * 84, self.DIM),
        ]

        for i, row in enumerate(tree):
            for j, node in enumerate(row):
                net_moves = 2 * j - i
                if node.is_early_exercised:
                    action = self._c("EXERCISE", self.RED + self.BOLD)
                elif i == model.steps:
                    action = self._c("EXPIRE", self.DIM)
                else:
                    action = self._c("HOLD", self.GREEN)

                lines.append(
                    f"  {i:4d} | {j:5d} | {net_moves:+4d} | {node.spot:10.4f} | "
                    f"{node.option_value:10.4f} | {node.continuation_value:12.4f} | "
                    f"{node.intrinsic_value:9.4f} | {action:>10}"
                )

        lines.append(self._c("-" * 84, self.DIM))
        return "\n".join(lines)

    def render_early_exercise_boundary(self, model: CRRBinomialTree) -> str:
        """Render the early exercise boundary frontier for American options."""
        if not model.contract.is_american():
            return "Early exercise boundary is only applicable to American options."

        tree = model.build_tree()
        lines: List[str] = [
            self._c("  AMERICAN OPTION EARLY EXERCISE FRONTIER", self.BOLD + self.MAGENTA),
            self._c("-" * 60, self.DIM),
            f"  {'Step':>4} | {'Time (yrs)':>10} | {'Critical Spot S*':>18} | {'Status':>16}",
            self._c("-" * 60, self.DIM),
        ]

        is_put = model.contract.is_put()

        for i, row in enumerate(tree):
            t_curr = i * model.dt
            # For put, exercise is optimal when S <= S*
            # For call, exercise is optimal when S >= S*
            exercised_nodes = [node for node in row if node.is_early_exercised]

            if exercised_nodes:
                if is_put:
                    # Highest spot among exercised nodes
                    crit_node = max(exercised_nodes, key=lambda n: n.spot)
                    lines.append(
                        f"  {i:4d} | {t_curr:10.4f} | {self._c(f'<= {crit_node.spot:10.4f}', self.RED):>27} | "
                        f"{self._c('EARLY EXERCISE', self.BOLD + self.RED):>16}"
                    )
                else:
                    # Lowest spot among exercised nodes
                    crit_node = min(exercised_nodes, key=lambda n: n.spot)
                    lines.append(
                        f"  {i:4d} | {t_curr:10.4f} | {self._c(f'>= {crit_node.spot:10.4f}', self.RED):>27} | "
                        f"{self._c('EARLY EXERCISE', self.BOLD + self.RED):>16}"
                    )
            else:
                lines.append(
                    f"  {i:4d} | {t_curr:10.4f} | {'No Early Ex':>18} | "
                    f"{self._c('CONTINUE', self.GREEN):>16}"
                )

        lines.append(self._c("-" * 60, self.DIM))
        return "\n".join(lines)
