#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# CONFIGURATION
# =====================================================

DATASET = "dataset_wt_clean.csv"

CLUSTERS_TO_TEST = [2, 3, 4]

FEATURES = [
    "RMSD",
    "Rg",
    "SASA",
    "HBONDS"
]

# =====================================================
# READING
# =====================================================

print("\nReading dataset...")

df = pd.read_csv(DATASET)

print(df.shape)

# =====================================================
# FEATURES
# =====================================================

X = df[FEATURES].copy()

# =====================================================
# NORMALIZING
# =====================================================

print("\nNormalizing...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# PCA
# =====================================================

print("\nExecuting PCA...")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

print(
    "\nExplained variance:"
)

print(
    pca.explained_variance_ratio_
)

print(
    "Total:",
    np.sum(
        pca.explained_variance_ratio_
    )
)

# =====================================================
# LOADINGS
# =====================================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=FEATURES
)

loadings.to_csv(
    "PCA_loadings.csv"
)

print("\nPCA LOADINGS")
print(loadings)

# =====================================================
#  CLUSTERS TEST
# =====================================================

for c in CLUSTERS_TO_TEST:

    print("\n================================")
    print(f"CLUSTERS = {c}")
    print("================================")

    # ==========================================
    # FUZZY
    # ==========================================

    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        X_scaled.T,
        c=c,
        m=2,
        error=0.005,
        maxiter=1000
    )

    print(
        f"FPC = {fpc:.4f}"
    )

    # ==========================================
    # CLUSTER DOMINANTE
    # ==========================================

    fuzzy_cluster = np.argmax(
        u,
        axis=0
    )

    # ==========================================
    # ENTROPY
    # ==========================================

    entropy = -np.sum(
        u * np.log(
            u + 1e-12
        ),
        axis=0
    )

    print(
        f"Entropy mean = {np.mean(entropy):.4f}"
    )

    # ==========================================
    # DATAFRAME
    # ==========================================

    result_df = df.copy()

    result_df[
        "FUZZY_CLUSTER"
    ] = fuzzy_cluster

    result_df[
        "FUZZY_ENTROPY"
    ] = entropy

    for i in range(c):

        result_df[
            f"Membership_C{i}"
        ] = u[i]

    # ==========================================
    # SAVE MEMBERSHIPS
    # ==========================================

    result_df.to_csv(
        f"membership_{c}.csv",
        index=False
    )

    # ==========================================
    # CLUSTERS MEDIAN
    # ==========================================

    cluster_means = (
        result_df
        .groupby(
            "FUZZY_CLUSTER"
        )[FEATURES]
        .mean()
    )

    print(
        "\nCluster means:"
    )

    print(
        cluster_means
    )

    cluster_means.to_csv(
        f"cluster_means_{c}.csv"
    )

    # ==========================================
    # POPULATION MEMBERSHIP - CLUSTERS
    # ==========================================

    cluster_population = (
        result_df[
            "FUZZY_CLUSTER"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    print(
        "\nCluster population (%)"
    )

    print(
        cluster_population
    )

    cluster_population.to_csv(
        f"cluster_population_{c}.csv",
        header=[
            "Population_percent"
        ]
    )

    # ==========================================
    # PCA FIGURE
    # ==========================================

    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=fuzzy_cluster,
        s=2
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        f"Fuzzy Clustering (c={c})"
    )

    plt.tight_layout()

    plt.savefig(
        f"PCA_clusters_{c}.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # ENTROPY
    # ==========================================

    plt.figure(
        figsize=(10,4)
    )

    plt.plot(
        entropy,
        linewidth=0.5
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Entropy"
    )

    plt.title(
        f"Fuzzy Entropy (c={c})"
    )

    plt.tight_layout()

    plt.savefig(
        f"entropy_{c}.png",
        dpi=300
    )

    plt.close()

    # ==========================================
    # POPULATION
    # ==========================================

    plt.figure(
        figsize=(6,4)
    )

    cluster_population.plot(
        kind="bar"
    )

    plt.ylabel(
        "Population (%)"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.title(
        f"Cluster Population (c={c})"
    )

    plt.tight_layout()

    plt.savefig(
        f"population_{c}.png",
        dpi=300
    )

    plt.close()

print("\n================================")
print("ANALYSIS CONCLUDED")
print("================================")

print("\nFiles generated:")

print("PCA_loadings.csv")

for c in CLUSTERS_TO_TEST:

    print(
        f"membership_{c}.csv"
    )

    print(
        f"cluster_means_{c}.csv"
    )

    print(
        f"cluster_population_{c}.csv"
    )

    print(
        f"PCA_clusters_{c}.png"
    )

    print(
        f"entropy_{c}.png"
    )

    print(
        f"population_{c}.png"
    )
