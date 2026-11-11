"""Frozen-SAL interval correction; run `python -m interval_correction.cli --help`.

Purpose: expose independent A/B/C/D experiments. Inputs: saved SAL outputs.
Outputs: interval predictions and gain scores. Assumptions: class 0 is spoof.
Limitations: event metrics are research diagnostics, not official frame metrics.
"""
