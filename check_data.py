import numpy as np
import os

DATA_PATH = os.path.join('data', 'sequences')
SIGNS = ['hello', 'yes', 'no', 'help', 'thanks', 'good']

print("Checking data...\n")
all_good = True

for sign in SIGNS:
    sign_path = os.path.join(DATA_PATH, sign)
    if not os.path.exists(sign_path):
        print(f"MISSING folder: {sign}")
        all_good = False
        continue

    files = os.listdir(sign_path)
    print(f"{sign}: {len(files)} sequences", end="")

    broken = []
    for f in files:
        arr = np.load(os.path.join(sign_path, f))
        if arr.shape != (30, 42):
            broken.append(f)

    if broken:
        print(f" — BAD FILES: {broken}")
        all_good = False
    else:
        print(" — OK")

print("\nAll good!" if all_good else "\nFix issues above before training.")