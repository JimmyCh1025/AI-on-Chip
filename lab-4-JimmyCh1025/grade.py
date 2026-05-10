import sys
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cycles", type=int, required=True, help="Threshold target cycles")
parser.add_argument("logfile", type=str, help="Path to the simulation log output")
args = parser.parse_args()

try:
    with open(args.logfile, "r") as f:
        text = f.read()
except FileNotFoundError:
    print(f"FAIL: Log file {args.logfile} not found! Did the simulation fail to run?")
    sys.exit(1)

if "[TB/CPU] *** TEST PASSED" not in text and "Errors        : 0  [PASS]" not in text:
    print("FAIL: The test did not pass the correctness check.")
    sys.exit(1)

match = re.search(r"Cycles\s*:\s*(\d+)", text)
if not match:
    print("FAIL: Could not parse cycle count from output.")
    sys.exit(1)

actual_cycles = int(match.group(1))

if actual_cycles <= args.cycles:
    print(f"SUCCESS: Student {actual_cycles} <= Threshold {args.cycles} ✅")
    sys.exit(0)
else:
    print(f"FAIL: Student {actual_cycles} > Threshold {args.cycles} ❌")
    sys.exit(1)
