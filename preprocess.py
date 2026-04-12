import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join('data', 'sequences')
SIGNS = ['hello', 'yes', 'no', 'help', 'thanks', 'good']

X, y = [], []

for label, sign in enumerate(SIGNS):
    sign_path = os.path.join(DATA_PATH, sign)
    for f in os.listdir(sign_path):
        seq = np.load(os.path.join(sign_path, f))
        X.append(seq)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {SIGNS}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("Saved to data/ folder.")