#!/usr/bin/env python3
"""Run from any working directory; imports cannot fall through to baseline."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from BrainPromptPro.cli import main

if __name__ == '__main__':
    main()
