"""Run regression checks for the authored fictional scenarios.

This is not empirical or independent validation of decision thresholds.
Run: python run_scenarios.py
"""
import sys

import pandas as pd

from modules.scenario_validation import evaluate_synthetic_scenarios

results = evaluate_synthetic_scenarios()
print("Authored fictional scenario regression checks - not real-world validation")
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_columns", 20)
print(results.to_string(index=False))
if results.empty or not results["Match"].all():
    sys.exit(1)
