import subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["step01_acquire.py", "step02_audit.py"]

print("MEMBER A - Data Acquisition and Quality Audit")
print("Report sections owned: 4.1-4.4, Appendix B")

for script in STEPS:
    print("-" * 70)
    print(script)
    print("-" * 70)
    r = subprocess.run([sys.executable, str(SRC / script)])
    if r.returncode != 0:
        print("Failed. Stopping.")
        raise SystemExit(r.returncode)

print("MEMBER A STAGE COMPLETE - share the data folder with B, C and D")
