"""Planogram alignment per shelf row (spec Section 3.7).

Needleman-Wunsch with a position term:
    D(i,j) = min( D(i-1,j-1) + alpha*|x_i - x^_j| + beta*[id_i != id^_j],   # match / substitution
                  D(i-1,j)   + gamma_gap,                                   # expected slot unmatched -> gap
                  D(i,j-1)   + gamma_ins )                                  # observed pack unmatched -> insertion
i indexes expected slots, j observed packs (both left to right).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = "unknown"


@dataclass
class AlignConfig:
    alpha: float = 0.05      # per cm
    beta: float = 1.0
    gamma_gap: float = 0.8
    gamma_ins: float = 0.8
    bay_change_frac: float = 0.40


@dataclass
class ExpectedSlot:
    slot_id: str
    x_cm: float              # slot centre
    sku_id: str


@dataclass
class ObservedPack:
    x_cm: float              # pack centre
    sku_id: str              # "unknown" if identity abstained
    conf: float = 1.0


def _id_cost(obs: str, exp: str) -> float:
    # An abstained identity is neither a match nor proof of a wrong product
    if obs == UNKNOWN:
        return 0.5
    return 0.0 if obs == exp else 1.0


def align_row(expected: list[ExpectedSlot], observed: list[ObservedPack], cfg: AlignConfig | None = None) -> list[dict]:
    """Returns one outcome per expected slot and per unmatched observation:
    {'kind': match|substitution|gap|insertion|uncertain, 'slot_id', 'expected', 'observed', 'x_cm'}."""
    cfg = cfg or AlignConfig()
    exp = sorted(expected, key=lambda e: e.x_cm)
    obs = sorted(observed, key=lambda o: o.x_cm)
    n, m = len(exp), len(obs)
    D = np.zeros((n + 1, m + 1))
    P = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up (gap), 2 left (insertion)
    D[1:, 0] = np.arange(1, n + 1) * cfg.gamma_gap
    D[0, 1:] = np.arange(1, m + 1) * cfg.gamma_ins
    P[1:, 0] = 1
    P[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = D[i - 1, j - 1] + cfg.alpha * abs(exp[i - 1].x_cm - obs[j - 1].x_cm) \
                + cfg.beta * _id_cost(obs[j - 1].sku_id, exp[i - 1].sku_id)
            up = D[i - 1, j] + cfg.gamma_gap
            left = D[i, j - 1] + cfg.gamma_ins
            k = int(np.argmin([diag, up, left]))
            D[i, j] = (diag, up, left)[k]
            P[i, j] = k
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        k = P[i, j]
        if i > 0 and j > 0 and k == 0:
            e, o = exp[i - 1], obs[j - 1]
            if o.sku_id == e.sku_id:
                kind = "match"
            elif o.sku_id == UNKNOWN:
                kind = "uncertain"
            else:
                kind = "substitution"
            out.append(dict(kind=kind, slot_id=e.slot_id, expected=e.sku_id, observed=o.sku_id, x_cm=o.x_cm))
            i, j = i - 1, j - 1
        elif i > 0 and (k == 1 or j == 0):
            e = exp[i - 1]
            out.append(dict(kind="gap", slot_id=e.slot_id, expected=e.sku_id, observed=None, x_cm=e.x_cm))
            i -= 1
        else:
            o = obs[j - 1]
            out.append(dict(kind="insertion", slot_id=None, expected=None, observed=o.sku_id, x_cm=o.x_cm))
            j -= 1
    return out[::-1]


def align_slots(expected: list[ExpectedSlot], observed_by_slot: dict[str, list[ObservedPack]],
                cfg: AlignConfig | None = None) -> list[dict]:
    """Convenience: flatten per-slot observations into one row and align."""
    obs = [o for v in observed_by_slot.values() for o in v]
    return align_row(expected, obs, cfg)


def bay_layout_changed(outcomes_by_row: list[list[dict]], cfg: AlignConfig | None = None) -> bool:
    """Design rule 6: many simultaneous mismatches -> one 'layout changed?' question."""
    cfg = cfg or AlignConfig()
    slots = [o for row in outcomes_by_row for o in row if o["slot_id"] is not None]
    if not slots:
        return False
    bad = sum(o["kind"] in ("substitution",) for o in slots)
    return bad / len(slots) > cfg.bay_change_frac


def slot_verdicts(outcomes: list[dict]) -> dict[str, str]:
    """Per-slot summary used by the task service: match|gap|substitution|uncertain.
    A slot with several facings is a gap only if none of its facings matched."""
    v: dict[str, str] = {}
    rank = {"match": 3, "substitution": 2, "uncertain": 1, "gap": 0}
    for o in outcomes:
        sid = o["slot_id"]
        if sid is None:
            continue
        if sid not in v or rank[o["kind"]] > rank[v[sid]]:
            v[sid] = o["kind"]
    return v
