import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import random

# ============================================================
#  FEATURE DEFINITIONS (must match backend.py)
# ============================================================
FEATURE_COLUMNS = [
    "paste_count_10s",
    "copy_count_10s",
    "rightclick_count_10s",
    "tab_switch_count_10s",
    "selection_count_10s",
    "mousemove_count_10s",
    "keydown_count_10s",
    "iki_mean_10s",
    "iki_std_10s",
    "event_rate_10s"
]

# ============================================================
#  IMPROVED SYNTHETIC DATA GENERATOR
# ============================================================

def generate_sample(is_cheater):
    """
    Improved generator:
      - more overlap
      - more noise
      - realistic variation
    """

    # Paste / Copy / Right-click / Tab
    paste = np.random.poisson(0.2 if not is_cheater else 1.2)
    paste += np.random.binomial(3, 0.1 if not is_cheater else 0.3)

    copy = np.random.poisson(0.3 if not is_cheater else 1.0)
    right = np.random.poisson(0.2 if not is_cheater else 0.8)
    tab = np.random.poisson(0.1 if not is_cheater else 1.0)

    selection = np.random.poisson(1.5 if not is_cheater else 2.5)

    # Mouse movement overlap
    mouse = int(np.random.normal(110, 25))
    if is_cheater:
        mouse = int(np.random.normal(90, 30))

    # Keydown overlap
    keydown = int(np.random.normal(85, 20))
    if is_cheater:
        keydown = int(np.random.normal(75, 25))

    # IKIs
    iki_mean = abs(np.random.normal(170, 50))
    if is_cheater:
        iki_mean = abs(np.random.normal(140, 60))

    iki_std = abs(np.random.normal(35, 12))
    if is_cheater:
        iki_std = abs(np.random.normal(25, 15))

    # Event rate
    event_rate = (mouse + keydown) / 10.0
    event_rate += np.random.normal(0, 1.0)

    # Random anomalies
    if random.random() < 0.07:
        paste += 2
    if random.random() < 0.07:
        tab += 1

    return [
        paste,
        copy,
        right,
        tab,
        selection,
        mouse,
        keydown,
        iki_mean,
        iki_std,
        event_rate
    ]

# ============================================================
#  DATASET CREATION
# ============================================================

N = 4000
cheat_fraction = 0.35

rows = []
labels = []

for _ in range(N):
    is_cheater = (random.random() < cheat_fraction)
    rows.append(generate_sample(is_cheater))
    labels.append(1 if is_cheater else 0)

df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
df["label"] = labels

df.to_csv("synthetic_dataset.csv", index=False)
print("[INFO] Synthetic dataset written to synthetic_dataset.csv")

# ============================================================
#  TRAINING — RANDOM FOREST
# ============================================================

X = df[FEATURE_COLUMNS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n[INFO] Classification report on synthetic data:")
print(classification_report(y_test, y_pred, digits=4))

joblib.dump(model, "model.pkl")
print("[INFO] model.pkl saved successfully.")