# QCArchive Data Processing Pipeline

This directory contains scripts for downloading and processing quantum chemical (QM) data from QCArchive for force field training. The workflow processes data from the OpenFF Sage 2.3.0 training set, downloading full optimization and torsion scan trajectories with energies and gradients.

## Overview

The complete pipeline consists of:
1. **Initial data download** - Download optimization and torsiondrive metadata from Sage 2.3.0 data pool
2. **Trajectory download** - Download full QM trajectories with coordinates, energies, and gradients
3. **Dataset combination** - Merge optimization and torsiondrive datasets, removing duplicate conformers
4. **Force filtering** - Remove conformers with high RMS forces (> 5 kcal/mol/Å)
5. **SMILES selection** - Filter by parameterizability with Sage force field and select diverse subset
6. **Train/test split** - Create stratified train/validation/test split

## Execution Summary

All scripts were executed successfully with the following statistics:

### 1. Download Sage Data Pool Metadata (`download_sage_pool.py`)

Downloads metadata (not full trajectories) from the Sage 2.3.0 training set that has been pre-processed and stored in `sage-data-pool/`.

**Input:** Pre-downloaded parquet files in `sage-data-pool/`
**Output:** 
- `sage-data-pool/optimization/` - 56,110 unique optimization SMILES
- `sage-data-pool/torsiondrive/` - 10,220 unique torsiondrive SMILES

**Note:** This is a convenience script. Original data was downloaded from QCArchive. See the Sage 2.3.0 `01_download-data/` directory for the full download procedure.

### 2. Download Optimizations (`download-optimizations.py`)

Downloads full optimization trajectories from QCArchive with QM energies and gradients.

**Execution Time:** ~4 hours (06:29:33 → 10:37:19)
**Statistics:**
- Input: 56,110 unique mapped SMILES 
- QCArchive records: 232,043 unique optimization IDs
- Parallel workers: 40
- Downloaded chunks: 233
- Processed entries: 16,147 unique SMILES

**Output:**
- `output/optimizations/dataset/minimum/` - Single minimum conformer per molecule
- `output/optimizations/dataset/trajectory/` - Full optimization trajectories (subsampled to max 10 points)

**Most common molecule:** 275 optimizations for a sulfonyl ether derivative

### 3. Download Torsiondrives (`download-torsiondrives.py`)

Downloads torsiondrive scan trajectories (1D or 2D torsion scans) from QCArchive.

**Execution Time:** ~20.5 hours (08:25:28 → 05:04:16 next day)
**Statistics:**
- Input: 10,220 unique mapped SMILES
- QCArchive records: 12,138 unique torsiondrive IDs
- Parallel workers: 40
- Downloaded chunks: 13
- Processed entries: 7,283 unique SMILES

**Output:**
- `output/torsiondrives/dataset/minimum/` - Global minimum from each torsiondrive
- `output/torsiondrives/dataset/trajectory/` - Full scan trajectories (aggressively subsampled to 3 points)

**Most common molecule:** 573 optimizations across 36 torsiondrives for a thiazole derivative

**Note:** Torsiondrives download takes significantly longer due to hierarchical structure (TorsionDrive → Optimizations → Singlepoints) requiring multiple API calls.

### 4. Combine Optimization and Torsiondrive Datasets

#### Minimum Conformers (`combine-opt-and-torsiondrive.py --mode minimum`)

**Execution Time:** ~3 minutes (05:51:29 → 05:54:46)
**Statistics:**
- Loaded optimizations: 16,147 records
- Loaded torsiondrives: 7,283 records  
- Removed duplicate conformers (RMSD < 0.1 Å)
- Final combined: 16,334 unique molecules

**Output:** `output/combined/minimum-all`

#### Trajectory Conformers (`combine-opt-and-torsiondrive.py --mode trajectory`)

**Execution Time:** ~1.2 hours (05:54:55 → 07:06:34)
**Statistics:**
- Loaded optimizations: 16,147 records
- Loaded torsiondrives: 7,283 records
- Removed duplicate conformers from trajectories
- Final combined: 16,334 unique molecules with trajectory data

**Output:** `output/combined/trajectory-all`

### 5. Filter Out High Forces (`filter-out-high-forces.py`)

Removes conformers with high RMS forces (> 5 kcal/mol/Å threshold) which indicate poor QM convergence or problematic geometries.

#### Minimum Conformers
**Execution Time:** ~50 seconds (01:05:44 → 01:06:34)
**Statistics:**
- Input: 16,334 molecules (337,890 total conformers)
- Removed: 820 conformers (0.24% of total)
- Output: 16,334 molecules retained

**Output:** `output/filtered/minimum-all`

#### Trajectory Conformers  
**Execution Time:** ~2.5 minutes (01:03:00 → 01:05:32)
**Statistics:**
- Input: 16,334 molecules with trajectories
- Removed conformers with RMS force > 5.0 kcal/mol/Å
- Output: 16,334 molecules retained

**Output:** `output/filtered/trajectory-all`

**Note:** Very few conformers filtered (2.3% for trajectories), indicating generally good QM convergence in the Sage dataset.

### 6. Select Parameterizable SMILES (`select-parameterizable-smiles.py`)

Tests each molecule for compatibility with OpenFF Sage 2.3.0 force field parameters.

**Execution Time:** ~1.6 hours (21:48:34 → 23:27:03)  
**Statistics:**
- Input: 16,334 molecules
- Parameterizable: 16,083 molecules (98.5%)
- Non-parameterizable: 251 molecules (1.5%)

**Common rejection reasons:**
- Missing bond parameters (e.g., B-O bonds, exotic heterocycles)
- Missing constraint distances for Si-C systems
- Unusual oxidation states or bonding patterns

**Output:** `smiles/combined-parameterizable.smi`

### 7. Select Diverse SMILES (`select-diverse-smiles.py`)

Uses MaxMin diversity selection with Morgan fingerprints to select a diverse subset of 12,000 molecules.

**Execution Time:** ~10 seconds (11:07:52 → 11:08:02)
**Statistics:**
- Input: 16,083 parameterizable molecules
- Selected: 12,000 diverse molecules
- Fingerprint type: Morgan (radius=3)
- Selection method: MaxMin algorithm

**Output:** `smiles/combined-diverse.smi`

### 8. Train/Test Split (`split-train-test.py`)

Creates stratified train/validation/test split based on molecular diversity.

**Execution Time:** ~2 seconds (11:08:04 → 11:08:06)
**Statistics:**
- Input: 12,000 diverse molecules
- Train: 9,600 molecules (80%)
- Validation: 1,200 molecules (10%)  
- Test: 1,200 molecules (10%)

**Split method:** MaxMin diversity-based selection for validation and test sets

**Output:** `smiles/combined-diverse-split.json`

### 9. Filter by SMILES (`filter-by-smiles.py`)

Filters the combined datasets to include only molecules in the train/validation/test splits.

#### Minimum Conformers
**Execution Time:** ~3 seconds (01:15:58 → 01:16:01)
**Output:**
- `output/filtered/minimum-train` (9,600 molecules)
- `output/filtered/minimum-valid` (1,200 molecules)
- `output/filtered/minimum-test` (1,200 molecules)

#### Trajectory Conformers
**Execution Time:** ~4 seconds (01:16:04 → 01:16:08)
**Output:**
- `output/filtered/trajectory-train` (9,600 molecules)
- `output/filtered/trajectory-valid` (1,200 molecules)
- `output/filtered/trajectory-test` (1,200 molecules)

### 10. Plotting (`plot-entry-info.py`)

Generates diagnostic plots showing distribution of conformers, energies, and forces.

#### Minimum Conformers
**Execution Time:** ~1 minute (23:45:34 → 23:46:31)
**Input:** 16,334 molecules

#### Trajectory Conformers  
**Execution Time:** ~1.5 minutes (23:43:56 → 23:45:27)
**Input:** 16,334 molecules with trajectories

**Output:** PNG plots in `plots/` directory

## Data Schema

All datasets are stored in HuggingFace datasets format with the following schema:

```python
{
    'smiles': str,              # Mapped SMILES with atom indices
    'coords': List[float],      # 3D coordinates in Angstrom (flattened N×3)
    'forces': List[float],      # Forces in kcal/mol/Å (flattened N×3)
    'energy': List[float],     # Energies in kcal/mol
}
```

**Trajectory datasets** contain multiple conformers per molecule; **minimum datasets** contain only the lowest energy conformer.

## Performance Notes

- **Parallel workers:** 30-40 threads used for QCArchive downloads
- **Unit conversions:** Bohr → Angstrom, Hartree → kcal/mol, gradients → forces (F = -∇E)
- **Subsampling:** Optimizations max 10 points, torsiondrives 3 points to reduce memory
- **Memory management:** Explicit garbage collection used for large trajectory processing

## Directory Structure

```
qca/
├── download_sage_pool.py              # Download metadata from Sage data pool
├── download-optimizations.py          # Download optimization trajectories  
├── download-torsiondrives.py          # Download torsiondrive trajectories
├── combine-opt-and-torsiondrive.py    # Merge datasets, remove duplicates
├── filter-out-high-forces.py          # Remove high-force conformers
├── select-parameterizable-smiles.py   # Check Sage FF compatibility
├── select-diverse-smiles.py           # MaxMin diversity selection
├── split-train-test.py                # Create train/valid/test split
├── filter-by-smiles.py                # Filter datasets by split
├── plot-entry-info.py                 # Generate diagnostic plots
├── logs/                              # Execution logs for all scripts
├── output/                            # Processed datasets
│   ├── optimizations/dataset/
│   ├── torsiondrives/dataset/
│   ├── combined/
│   └── filtered/
├── smiles/                            # SMILES lists and split definitions
└── plots/                             # Diagnostic plots

```

## Total Pipeline Execution Time

Approximate wall time: **27+ hours**
- Download phases: ~24.5 hours (optimizations + torsiondrives)
- Processing phases: ~3 hours (combining, filtering, selection)

**Note:** Download time is dominated by API latency. Actual computation is minimal.
