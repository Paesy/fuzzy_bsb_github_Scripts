# Fuzzy Conformational States in Mycobacterium tuberculosis KatG

Python scripts for the analysis described in:

> **Identification of Fuzzy Conformational States in Molecular Dynamics Simulations of Mycobacterium tuberculosis KatG Using Fuzzy C-Means Clustering**

The workflow analyzes three independent 500 ns molecular dynamics trajectories (1.5 µs in total) using four structural descriptors: RMSD, radius of gyration (Rg), protein SASA, and the number of protein hydrogen bonds. PCA is used for visualization, while Fuzzy C-Means (FCM) is applied directly to the standardized descriptor space. Shannon entropy is used to quantify ambiguity in fuzzy state assignment.

## Repository structure

```text
fuzzy-katg-md/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 01_build_dataset.py
│   ├── 02_fuzzy_discovery.py
│   ├── 03_fuzzy_states.py
│   ├── 04_fuzzy_dynamics.py
│   └── 05_representative_frames.py
```

Raw MD trajectories and large simulation files are intentionally not included in this repository. Generated descriptor files, datasets, figures, and analysis outputs are also ignored by Git by default.

## Requirements

- Python 3.10 or newer
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- Scikit-fuzzy

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Input data

The analysis expects four descriptor files for each of three WT replicas:

| Replica | RMSD | Rg | SASA | H-bonds |
|---|---|---|---|---|
| WT1 | `WT_rmsd.xvg` | `WT_gyrate.xvg` | `WT_sasa_protein.xvg` | `WT_hbnum_protein.xvg` |
| WT2 | `WT2_rmsd.xvg` | `WT2_gyrate.xvg` | `WT2_sasa_protein.xvg` | `WT2_hbnum_protein.xvg` |
| WT3 | `WT3_rmsd.xvg` | `WT3_gyrate.xvg` | `WT3_sasa_protein.xvg` | `WT3_hbnum_protein.xvg` |

See `data/README.md` for the expected input organization.

## Analysis workflow

### 1. Build the combined dataset

`01_build_dataset.py` reads the descriptor `.xvg` files for WT1, WT2, and WT3 and creates `data/dataset_wt_clean.csv` while retaining the replica identifier. All analysis outputs are written to `results/`.

```bash
python scripts/01_build_dataset.py
```

### 2. Explore the number of fuzzy clusters

`02_fuzzy_discovery.py` standardizes the four descriptors, performs PCA, and evaluates FCM solutions with 2–7 clusters using the Fuzzy Partition Coefficient (FPC) and mean membership entropy.

```bash
python scripts/02_fuzzy_discovery.py
```

### 3. Analyze candidate fuzzy states

`03_fuzzy_states.py` performs detailed FCM analyses for 2, 3, and 4 clusters, saving memberships, cluster means, populations, PCA projections, and entropy plots.

```bash
python scripts/03_fuzzy_states.py
```

### 4. Perform the final three-state analysis

`04_fuzzy_dynamics.py` applies the final three-cluster FCM model, calculates Shannon entropy, identifies the upper 5% of the entropy distribution as high-entropy frames, and generates the state-population, membership, entropy, PCA, and temporal connectivity outputs.

```bash
python scripts/04_fuzzy_dynamics.py
```

### 5. Extract representative structures

`05_frames_representativos.py` identifies, for each fuzzy state, the trajectory frame with the highest membership value and reports the corresponding replica, frame, time, and membership. It also writes `representative_frames.csv`.

```bash
python scripts/05_frames_representativos.py
```

## Important methodological details

- Four descriptors are used: RMSD, Rg, SASA, and protein hydrogen-bond count.
- Descriptors are standardized using z-score normalization before PCA and FCM.
- PCA is used for dimensionality reduction and visualization only; FCM is applied directly to the standardized descriptor space.
- FCM uses a fuzzification coefficient of `m=2.0`, convergence tolerance of `0.005`, and a maximum of `1000` iterations.
- Candidate numbers of clusters from 2 to 7 are evaluated during model exploration.
- The final analysis uses three fuzzy states, despite the highest FPC being obtained for two clusters, because the three-cluster solution provided a reproducible intermediate conformational region across the independent trajectories.
- Shannon entropy is calculated from the FCM membership values to quantify assignment ambiguity.
- High-entropy frames are defined using the 95th percentile of the entropy distribution.

## Reproducibility

The scripts correspond to the computational workflow used in the manuscript. To reproduce the published analysis, use the same descriptor files generated from the MD trajectories and preserve the file names expected by `01_build_dataset.py`.

The repository is intended to provide the analysis code and workflow rather than the full MD trajectories.

## License

This project is released under the MIT License. See `LICENSE`.
