import os
import sys
from pathlib import Path

print("CWD:", os.getcwd())
print("Executable:", sys.executable)
print("__file__:", Path(__file__).resolve())
print("sys.path (top 5):")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")