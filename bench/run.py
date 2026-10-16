"""Run the TaskBot benchmark. See README, "Running the benchmark".

    python bench/run.py --backend fake --limit 5
    python bench/run.py --compare bench/results/OLD.csv bench/results/NEW.csv
"""

import sys
from pathlib import Path

from taskbot.bench import main

if __name__ == "__main__":
    sys.exit(main(bench_dir=Path(__file__).resolve().parent))
