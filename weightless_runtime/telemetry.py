#!/usr/bin/env python3
"""Engine-neutral directional telemetry formulas."""

def directional_scalar_metrics(
    coefficient: float,
    residual_norm: float,
    effective_alpha: float,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    normalized = coefficient / max(residual_norm, epsilon)
    return {
        "coefficient": coefficient,
        "post_coefficient": (1.0 - effective_alpha) * coefficient,
        "normalized_coefficient": normalized,
        "residual_norm": residual_norm,
        "directional_energy": normalized * normalized,
        "effective_alpha": effective_alpha,
        "delta_norm": abs(effective_alpha * coefficient),
    }
