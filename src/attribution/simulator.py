"""Counterfactual GRAP policy simulator - pure, testable, UI-independent.

First-order linear counterfactual: a reduction of X% on an emission source
removes (X% x that source's share) of the total PM2.5 mass. This is a
documented FIRST-ORDER approximation (linear response, no chemistry feedback)
and the UI must always present it as such.

The function is pure so it can be unit-tested for bounds and monotonicity.
"""

from __future__ import annotations

from typing import Dict

INTERVENTION_KEYS = ("vehicular", "stubble", "industrial", "dust")


def simulate_policy(
    current_pm25: float,
    source_shares_pct: Dict[str, float],
    interventions_pct: Dict[str, float],
) -> dict:
    """Apply policy levers to source shares.

    Args:
        current_pm25: current PM2.5 (ug/m3) from real telemetry.
        source_shares_pct: mapping of intervention key -> share of PM2.5 mass (%),
            e.g. {"vehicular": 36.2, "stubble": 31.4, "industrial": 18.1, "dust": 14.3}
        interventions_pct: mapping of intervention key -> reduction applied (0-100 %).

    Returns:
        dict with new_pm25, avoided_ug, avoided_pct, post_shares (renormalised %),
        and a validity flag. Raises ValueError on out-of-range inputs so callers
        cannot silently produce impossible scenarios.
    """
    if current_pm25 < 0:
        raise ValueError("current_pm25 must be >= 0")
    for k, v in interventions_pct.items():
        if k not in INTERVENTION_KEYS:
            raise ValueError(f"unknown intervention key: {k}")
        if not (0.0 <= float(v) <= 100.0):
            raise ValueError(f"intervention '{k}' must be within [0, 100], got {v}")

    total_share = sum(float(source_shares_pct.get(k, 0.0)) for k in INTERVENTION_KEYS)
    if total_share <= 0:
        raise ValueError("source shares must sum to a positive value")

    avoided_fraction = 0.0
    post_shares: Dict[str, float] = {}
    for k in INTERVENTION_KEYS:
        share = float(source_shares_pct.get(k, 0.0))
        reduction = float(interventions_pct.get(k, 0.0))
        avoided_fraction += (reduction / 100.0) * (share / 100.0)
        post_shares[k] = share * (1.0 - reduction / 100.0)

    # Renormalise surviving sources to 100% (trapped/atmospheric share unchanged).
    surviving = sum(post_shares.values())
    if surviving > 0:
        post_shares = {k: 100.0 * v / surviving for k, v in post_shares.items()}

    new_pm25 = current_pm25 * (1.0 - avoided_fraction)
    return {
        "valid": True,
        "model": "first-order linear counterfactual (no chemistry feedback)",
        "current_pm25": float(current_pm25),
        "new_pm25": float(new_pm25),
        "avoided_ug": float(current_pm25 - new_pm25),
        "avoided_pct": float(100.0 * avoided_fraction),
        "post_shares": post_shares,
    }


def shares_to_intervention_keys(share_dict: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    """Map full source names -> intervention keys (vehicular/stubble/industrial/dust)."""
    out: Dict[str, float] = {}
    for source_name, share in share_dict.items():
        key = mapping.get(source_name)
        if key:
            out[key] = float(share)
    return out


# Documented mapping between receptor factor names and simulator levers.
SOURCE_TO_LEVER = {
    "Vehicular Fleet Exhaust": "vehicular",
    "Agricultural Biomass / Stubble Burning": "stubble",
    "Industrial & Thermal Combustion": "industrial",
    "Crustal & Resuspended Dust": "dust",
}


if __name__ == "__main__":
    demo = simulate_policy(
        current_pm25=284.0,
        source_shares_pct={"vehicular": 36.2, "stubble": 31.4, "industrial": 18.1, "dust": 14.3},
        interventions_pct={"vehicular": 30, "stubble": 45, "industrial": 35, "dust": 25},
    )
    for k, v in demo.items():
        print(k, "=", v)
