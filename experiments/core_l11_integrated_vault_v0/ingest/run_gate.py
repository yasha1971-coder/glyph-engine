"""Explicit isolated-mode entry point: make this directory importable under -I."""
import runpy, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
runpy.run_path(str(Path(__file__).with_name('gate.py')),run_name='__main__')
