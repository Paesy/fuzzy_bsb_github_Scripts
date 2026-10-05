#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# ============================================================
# CONFIGURATION
# ============================================================

N_CLUSTERS = 3

# ============================================================
# READING
# ============================================================

print("\nReading dataset...")

df = pd.read_csv("dataset_wt_clean.csv")

print(df.shape)

features = ["RMSD", "Rg", "SASA", "HBONDS"]

X = df[features].values

# ============================================================
# NORMALIZING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# PCA
# ============================================================

print("\nExecuting PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Explained variance:")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

# ============================================================
# FUZZY C-MEANS
# ============================================================

print("\nExecuting Fuzzy C-Means...")

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled.T,
    c=N_CLUSTERS,
    m=2.0,
    error=0.005,
    maxiter=1000,
    seed=42
)

print("FPC =", round(fpc,4))

# ============================================================
# CLUSTER DOMINANTE
# ============================================================

labels = np.argmax(u, axis=0)

df["FUZZY_CLUSTER"] = labels

# ============================================================
# MEMBERSHIPS
# ============================================================

for i in range(N_CLUSTERS):
    df[f"MU_{i}"] = u[i]

# ============================================================
# ENTROPIA
# ============================================================

entropy = -np.sum(
    u * np.log(u + 1e-10),
    axis=0
)

df["ENTROPY"] = entropy

print("\nEntropy median =", round(entropy.mean(),4))

# ============================================================
# DETECÇÃO DE TRANSIÇÕES POR ENTROPIA
# ============================================================

print("\nDetecting transition regions :)")

entropy_threshold = np.percentile(
    entropy,
    95
)

transition_frames = np.where(
    entropy >= entropy_threshold
)[0]

transition_df = df.iloc[
    transition_frames
]

transition_df.to_csv(
    "transition_frames.csv",
    index=False
)

print(
    "Entroyp threshold:",
    round(entropy_threshold,4)
)

print(
    "Transition frames:",
    len(transition_frames)
)

print(
    "Percent:",
    round(
        100 * len(transition_frames) / len(df),
        2
    ),
    "%"
)
# ============================================================
# SAVING RESULTS
# ============================================================

df.to_csv(
    "fuzzy_memberships.csv",
    index=False
)

# ============================================================
# STATE POPULATION
# ============================================================

population = (
    pd.Series(labels)
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

population.to_csv("state_population.csv")

print("\nPopulation (%)")

print(population)

# ============================================================
# FIGURES
# ============================================================

plt.figure(figsize=(7,5))

population.plot(kind="bar")

plt.ylabel("Population (%)")
plt.xlabel("State")

plt.title("State Population")

plt.tight_layout()

plt.savefig(
    "state_population.png",
    dpi=300
)

plt.close()

# ============================================================
# MEMBERSHIP DURING EXECUTION
# ============================================================

plt.figure(figsize=(14,6))

for i in range(N_CLUSTERS):

    plt.plot(
        df["Frame"],
        df[f"MU_{i}"],
        lw=1,
        label=f"State {i}"
    )

plt.xlabel("Frame")
plt.ylabel("Membership")

plt.title("Fuzzy Membership Through Time")

plt.legend()

plt.tight_layout()

plt.savefig(
    "membership_time_series.png",
    dpi=300
)

plt.close()

# ============================================================
# ENTROPY
# ============================================================

plt.figure(figsize=(14,6))

plt.plot(
    df["Frame"],
    entropy,
    lw=1
)

plt.xlabel("Frame")
plt.ylabel("Entropy")

plt.title("Conformational Uncertainty")

plt.tight_layout()

plt.savefig(
    "entropy_time_series.png",
    dpi=300
)

plt.close()
# ============================================================
# ENTROPY HISTOGRAM
# ============================================================

plt.figure(figsize=(8,6))

plt.hist(
    entropy,
    bins=100
)

plt.axvline(
    entropy_threshold,
    linestyle="--",
    linewidth=2,
    label=f"P95 = {entropy_threshold:.3f}"
)

plt.xlabel("Entropy")
plt.ylabel("Frames")

plt.title(
    "Entropy Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "entropy_histogram.png",
    dpi=300
)

plt.close()

# ============================================================
# TRANSITION REGIONS IN PCA
# ============================================================

plt.figure(figsize=(10,8))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c="lightgray",
    s=2,
    alpha=0.20
)

plt.scatter(
    X_pca[transition_frames,0],
    X_pca[transition_frames,1],
    c=entropy[transition_frames],
    cmap="inferno",
    s=8
)

plt.colorbar(
    label="Entropy"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "High-Entropy Transition Regions"
)

plt.tight_layout()

plt.savefig(
    "transition_regions_entropy.png",
    dpi=300
)

plt.close()

# ============================================================
# TRANSITION MATRIX
# ============================================================

transition_matrix = np.zeros(
    (N_CLUSTERS, N_CLUSTERS),
    dtype=int
)

for i in range(len(labels)-1):

    current_state = labels[i]
    next_state = labels[i+1]

    transition_matrix[
        current_state,
        next_state
    ] += 1

transition_df = pd.DataFrame(
    transition_matrix,
    index=[f"S{i}" for i in range(N_CLUSTERS)],
    columns=[f"S{i}" for i in range(N_CLUSTERS)]
)

transition_df.to_csv(
    "transition_matrix.csv"
)

print("\nMatriz de transições")

print(transition_df)

# ============================================================
# HEATMAP
# ============================================================

plt.figure(figsize=(7,6))

plt.imshow(
    transition_matrix,
    cmap="viridis"
)

plt.colorbar()

plt.xticks(
    range(N_CLUSTERS),
    [f"S{i}" for i in range(N_CLUSTERS)]
)

plt.yticks(
    range(N_CLUSTERS),
    [f"S{i}" for i in range(N_CLUSTERS)]
)

plt.xlabel("Next State")
plt.ylabel("Current State")

plt.title("State Transition Matrix")

plt.tight_layout()

plt.savefig(
    "transition_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# PCA
# ============================================================

plt.figure(figsize=(10,8))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="viridis",
    s=4
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    f"Fuzzy States (c={N_CLUSTERS})"
)

plt.tight_layout()

plt.savefig(
    "fuzzy_states_pca.png",
    dpi=300
)

plt.close()

membership_df = pd.DataFrame(u.T)

membership_df.columns = [
    f"State_{i}" for i in range(N_CLUSTERS)
]

membership_df["Frame"] = df["Frame"]

membership_df.to_csv(
    "fuzzy_memberships_04.csv",
    index=False
)


print("\n================================")
print("CONCLUDED")
print("================================")
