"""SAL fixed10 training, validation, and test entry. Default: check only."""
import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from sal.pipeline import experiment_main

if __name__ == "__main__":
    raise SystemExit(experiment_main("fixed10"))
