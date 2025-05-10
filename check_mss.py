#!/usr/bin/env python3

try:
    import mss
    print(f"MSS is installed: {mss.__version__}")
except ImportError:
    print("MSS is NOT installed")