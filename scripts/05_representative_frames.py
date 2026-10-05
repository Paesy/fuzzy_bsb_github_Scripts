#!/usr/bin/env python3

import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

MEMBERSHIP_FILE = "fuzzy_memberships_04.csv"
DATASET_FILE = "dataset_wt_clean.csv"
OUTPUT_FILE = "representative_frames.csv"
SAMPLING_INTERVAL_PS = 10.0


# =====================================================
# LOAD FUZZY MEMBERSHIPS
# =====================================================

df = pd.read_csv(MEMBERSHIP_FILE)


# =====================================================
# LOAD REPLICA INFORMATION
# =====================================================

# fuzzy_memberships_04.csv does not contain the replica identifier.
# The original dataset preserves the replica information in the same
# row order, so it is added after checking that both files are aligned.
dataset = pd.read_csv(
    DATASET_FILE,
    usecols=["Replica", "Frame"]
)

if len(df) != len(dataset):
    raise ValueError(
        "The membership file and dataset have different numbers of rows."
    )

if not df["Frame"].equals(dataset["Frame"]):
    raise ValueError(
        "Frame numbering does not match between the membership file "
        "and the original dataset."
    )

df["Replica"] = dataset["Replica"].values


# =====================================================
# SELECT REPRESENTATIVE FRAMES
# =====================================================

representatives = []

for state in [0, 1, 2]:

    col = f"State_{state}"

    best_idx = df[col].idxmax()

    replica = df.loc[best_idx, "Replica"]

    frame = int(
        df.loc[best_idx, "Frame"]
    )

    membership = float(
        df.loc[best_idx, col]
    )

    time_ns = frame * SAMPLING_INTERVAL_PS / 1000.0

    representatives.append(
        {
            "State": f"S{state}",
            "Replica": replica,
            "Frame": frame,
            "Time_ns": time_ns,
            "Membership": membership
        }
    )

    print()

    print(
        f"STATE S{state}"
    )

    print(
        f"Replica = {replica}"
    )

    print(
        f"Frame = {frame}"
    )

    print(
        f"Time = {time_ns:.2f} ns"
    )

    print(
        f"Membership = {membership:.4f}"
    )


# =====================================================
# SAVE REPRESENTATIVE FRAMES
# =====================================================

representatives_df = pd.DataFrame(representatives)
representatives_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(f"Representative frames saved to {OUTPUT_FILE}")
