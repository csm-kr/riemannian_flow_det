"""Entry point so `python -m dataset` runs the smoke test in __init__."""
import runpy
runpy.run_module("dataset.__init__", run_name="__main__")
