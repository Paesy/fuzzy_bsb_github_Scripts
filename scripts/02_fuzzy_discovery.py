#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import skfuzzy as fuzz

# =====================================================
# CONFIGURAÇÃO
# =====================================================

DATASET = "dataset_wt_clean.csv"

MIN_CLUSTERS = 2
MAX_CLUSTERS = 7

# =====================================================
# LEITURA
# =====================================================

print("\nReading dataset...")

df = pd.read_csv(DATASET)

print(df.shape)

# =====================================================
# FEATURES
# =====================================================

features = [
    "RMSD",
    "Rg",
    "SASA",
    "HBONDS"
]

X = df[features].copy()

print("\nDescriptors utilized:")
print(features)

# =====================================================
# NORMALIZAÇÃO
# =====================================================

print("\nNormalizing...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# PCA
# =====================================================

print("\nExecuting PCA...")

pca_full = PCA()

X_pca_full = pca_full.fit_transform(
    X_scaled
)

variance = pca_full.explained_variance_ratio_

pca_table = pd.DataFrame({
    "PC": np.arange(
        1,
        len(variance) + 1
    ),
    "ExplainedVariance": variance,
    "CumulativeVariance": np.cumsum(
        variance
    )
})

pca_table.to_csv(
    "PCA_variance.csv",
    index=False
)

print("\nExplained variance:")

print(pca_table)

# =====================================================
# PCA PARA VISUALIZAÇÃO
# =====================================================

pca_2d = PCA(
    n_components=2
)

X_pca = pca_2d.fit_transform(
    X_scaled
)

plt.figure(figsize=(7,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    s=1,
    alpha=0.5
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "WT CLEAN - PCA"
)

plt.tight_layout()

plt.savefig(
    "PCA_WT.png",
    dpi=300
)

plt.close()

# =====================================================
# FUZZY CLUSTERING
# =====================================================

print("\nTesting different amounts of clusters...")

results = []

for c in range(
    MIN_CLUSTERS,
    MAX_CLUSTERS + 1
):

    print(f"\nClusters = {c}")

    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        X_scaled.T,
        c=c,
        m=2,
        error=0.005,
        maxiter=1000
    )

    entropy = -np.sum(
        u * np.log(
            u + 1e-12
        ),
        axis=0
    )

    mean_entropy = np.mean(
        entropy
    )

    results.append([
        c,
        fpc,
        mean_entropy
    ])

    print(
        f"FPC = {fpc:.4f}"
    )

    print(
        f"Entropy = {mean_entropy:.4f}"
    )

# =====================================================
# RESULTADOS
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Clusters",
        "FPC",
        "Entropy"
    ]
)

results_df.to_csv(
    "Fuzzy_cluster_selection.csv",
    index=False
)

print("\n========================")
print("RESULTADOS")
print("========================")

print(results_df)

# =====================================================
# FPC PLOT
# =====================================================

plt.figure(figsize=(7,5))

plt.plot(
    results_df["Clusters"],
    results_df["FPC"],
    marker="o"
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "FPC"
)

plt.title(
    "Fuzzy Partition Coefficient"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "FPC_vs_clusters.png",
    dpi=300
)

plt.close()

# =====================================================
# ENTROPY PLOT
# =====================================================

plt.figure(figsize=(7,5))

plt.plot(
    results_df["Clusters"],
    results_df["Entropy"],
    marker="o"
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "Mean Entropy"
)

plt.title(
    "Mean Fuzzy Entropy"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "Entropy_vs_clusters.png",
    dpi=300
)

plt.close()

# =====================================================
# MELHOR MODELO
# =====================================================

best_row = results_df.loc[
    results_df["FPC"].idxmax()
]

print("\n========================")
print("BEST MODEL")
print("========================")

print(
    f"Clusters = {int(best_row['Clusters'])}"
)

print(
    f"FPC = {best_row['FPC']:.4f}"
)

print(
    f"Entropy = {best_row['Entropy']:.4f}"
)

print("\nFiles generated:")

print("PCA_variance.csv")
print("PCA_WT.png")
print("Fuzzy_cluster_selection.csv")
print("FPC_vs_clusters.png")
print("Entropy_vs_clusters.png")
