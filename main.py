"""Entry point: download, clean and split Airbnb listing data."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pipeline import run_data_pipeline  # noqa: E402


def main() -> None:
    """Run the full data pipeline."""
    run_data_pipeline()


if __name__ == "__main__":
    main()
