"""
Shared small numerical constants for hygiene and consistency.

These values centralize previously inlined literals to avoid drift. Do not
change their values without auditing dependent math.
"""

# Small positive floor to keep logs finite
EPS_LOG: float = 1e-300

# Minimum physically meaningful flux floor used in guards
FLUX_MIN_FLOOR: float = 1e-22

# Default expansion factors for ensuring a grid straddles a selection threshold
THRESH_FACTOR_LOW: float = 1e-2
THRESH_FACTOR_HIGH: float = 1e2
