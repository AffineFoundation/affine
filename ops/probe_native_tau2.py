#!/usr/bin/env python3
"""Run bounded original tau2 native orchestrator HTTP transport conformance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from subnet.native_tau2_probe import main
if __name__=='__main__': main()
