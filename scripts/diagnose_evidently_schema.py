"""One-off diagnostic: dumps the real structure of Evidently's report.dict()
output so src/drift/detect.py's _reduce_to_fingerprint() can be fixed against
actual output instead of guessed key names. Run this once, paste the output
back. Safe to delete afterward - not part of the pipeline.
"""

import json

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

rng = np.random.default_rng(0)
reference = pd.DataFrame(
    {
        "DebtRatio": rng.random(200),
        "RevolvingUtilizationOfUnsecuredLines": rng.random(200),
        "MonthlyIncome": rng.uniform(1000, 8000, 200),
    }
)
# deliberately drifted current batch so at least one column clearly drifts
current = pd.DataFrame(
    {
        "DebtRatio": rng.random(200) + 0.8,
        "RevolvingUtilizationOfUnsecuredLines": rng.random(200),
        "MonthlyIncome": rng.uniform(1000, 8000, 200),
    }
)

report = Report([DataDriftPreset()])
result = report.run(current_data=current, reference_data=reference)
raw = result.dict()

print("=== top-level keys ===")
print(list(raw.keys()))

print("\n=== full dict (pretty) ===")
print(json.dumps(raw, indent=2, default=str))
