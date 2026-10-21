"""Hardware and CUDA diagnostic script for FlashSpeculativeMamba."""

import sys
import os
from pathlib import Path

# Add project root to sys.path so src can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend import print_backend_summary

def main() -> None:
    print_backend_summary()

if __name__ == "__main__":
    main()
