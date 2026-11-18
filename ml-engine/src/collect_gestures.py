import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.argv.insert(1, "gesture")
runpy.run_module("backend.tools.collect_webcam", run_name="__main__")
