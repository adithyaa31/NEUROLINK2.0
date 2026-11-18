import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
runpy.run_module("backend.tools.predict_model", run_name="__main__")
