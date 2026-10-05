import pandas as pd
import numpy as np

# =====================================================
# XVG READER
# =====================================================

def read_xvg(file):

    values = []

    with open(file) as f:

        for line in f:

            if line.startswith("@"):
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            values.append(
                float(parts[1])
            )

    return np.array(values)

# =====================================================
# REPLICAS
# =====================================================

replicas = {

    "WT1": {
        "RMSD": "WT_rmsd.xvg",
        "Rg": "WT_gyrate.xvg",
        "SASA": "WT_sasa_protein.xvg",
        "HBONDS": "WT_hbnum_protein.xvg"
    },

    "WT2": {
        "RMSD": "WT2_rmsd.xvg",
        "Rg": "WT2_gyrate.xvg",
        "SASA": "WT2_sasa_protein.xvg",
        "HBONDS": "WT2_hbnum_protein.xvg"
    },

    "WT3": {
        "RMSD": "WT3_rmsd.xvg",
        "Rg": "WT3_gyrate.xvg",
        "SASA": "WT3_sasa_protein.xvg",
        "HBONDS": "WT3_hbnum_protein.xvg"
    }
}

# =====================================================
# DATASET
# =====================================================

all_data = []

for replica, files in replicas.items():

    print(f"Lendo {replica}")

    rmsd = read_xvg(
        files["RMSD"]
    )

    rg = read_xvg(
        files["Rg"]
    )

    sasa = read_xvg(
        files["SASA"]
    )

    hb = read_xvg(
        files["HBONDS"]
    )

    n = min(
        len(rmsd),
        len(rg),
        len(sasa),
        len(hb)
    )

    df = pd.DataFrame({

        "Replica":
        [replica] * n,

        "Frame":
        np.arange(n),

        "RMSD":
        rmsd[:n],

        "Rg":
        rg[:n],

        "SASA":
        sasa[:n],

        "HBONDS":
        hb[:n]

    })

    all_data.append(df)

dataset = pd.concat(
    all_data,
    ignore_index=True
)

dataset.to_csv(
    "dataset_wt_clean.csv",
    index=False
)

print("\nDataset criado")

print(dataset.shape)

print(dataset.head())
