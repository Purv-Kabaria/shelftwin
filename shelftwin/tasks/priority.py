"""Task priority, adaptive low-stock threshold, time-to-empty (spec Section 3.9)."""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

# Shopper response to an out-of-stock (Gruen et al. 2002): 31% buy elsewhere + 9% do not buy
P_LOST_DEFAULT = 0.40


def low_stock_threshold(demand_per_h: float, t_resp_h: float = 0.5, t_safe_h: float = 0.25) -> float:
    """theta_s = d_s * (T_resp + T_safe), in units. Fast movers warn earlier."""
    return max(1.0, demand_per_h * (t_resp_h + t_safe_h))


def priority(demand_per_h: float, t_gap_h: float, unit_margin: float, p_lost: float = P_LOST_DEFAULT,
             critical: bool = False, beta: float = 1000.0) -> float:
    """P_s = d_s * T_gap * m_s * p_lost + beta * 1[critical]  (expected lost margin, Rs)."""
    return demand_per_h * t_gap_h * unit_margin * p_lost + (beta if critical else 0.0)


@dataclass
class DemandPosterior:
    """Gamma posterior on a Poisson depletion rate (units/hour)."""
    a0: float = 1.0      # prior shape  (prior mean a0/b0 = 2 units/h)
    b0: float = 0.5      # prior rate (hours)
    n: float = 0.0       # units removed observed
    t_h: float = 0.0     # observed hours

    def update(self, units_removed: float, hours: float) -> None:
        self.n += max(0.0, units_removed)
        self.t_h += max(0.0, hours)

    @property
    def shape(self) -> float:
        return self.a0 + self.n

    @property
    def rate(self) -> float:
        return self.b0 + self.t_h

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    def quantile(self, q: float) -> float:
        return float(stats.gamma.ppf(q, a=self.shape, scale=1.0 / self.rate))


def time_to_empty(units_left: float, post: DemandPosterior, q_lo: float = 0.1, q_hi: float = 0.9) -> dict:
    """T_empty = u_s / d_s, reported as a range from posterior quantiles (hours)."""
    if units_left <= 0:
        return dict(mean_h=0.0, lo_h=0.0, hi_h=0.0)
    d_mean, d_hi, d_lo = post.mean, post.quantile(q_hi), post.quantile(q_lo)
    return dict(mean_h=units_left / max(d_mean, 1e-6),
                lo_h=units_left / max(d_hi, 1e-6),    # fast demand -> empties sooner
                hi_h=units_left / max(d_lo, 1e-6))


def units_from_coverage(coverage: float, facings: int, depth_units: int) -> float:
    """Rough estimate u_s ~ coverage x facings x depth (a camera sees only the front row)."""
    return max(0.0, coverage) * facings * depth_units
