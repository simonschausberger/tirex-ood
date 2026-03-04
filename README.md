# TiRex OOD Detection Project

This project implements an Out-of-Distribution (OOD) detection pipeline for time-series data using TiRex embeddings and the Mahalanobis Distance.

## Overview

The goal of this project is to evaluate the OOD detection performance for the zero-shot time-series model TiRex. The pipeline extracts embeddings from both In-Distribution (ID) and OOD datasets, calculates a baseline distribution from the ID data and then computes OOD scores based on the Mahalanobis distance to the ID distribution. Furthermore, it computes the AUROC and FPR@95%TPR metrics to assess the OOD detection performance.

### Key Components

- **`src/config.py`**: Contains the specifications for ID and OOD datasets.
- **`src/feature_extractor.py`**: Utilizes the TiRex model to extract embeddings from time-series.
- **`src/mahalanobis.py`**: Implements the Mahalanobis distance calculation. It includes the baseline calculation and an OOD scoring method.
- **`src/data_loader.py`**: A streaming data loader that fetches datasets from Hugging Face and Kaggle repositories.
- **`scripts/`**: A set of scripts to execute the full pipeline.
- **`notebooks/results_visualization.ipynb`**: A notebook to visualize the results of the OOD detection experiments.

## Installation

1. Clone the repository and navigate to the project root.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set up Kaggle credentials in `.kaggle/` for accessing extra Chronos datasets.

## Pipeline Workflow

The OOD detection process is divided into these main steps:

### 1. Cache Embeddings
Extract and save embeddings for both ID and OOD datasets. This step can be time-consuming as it involves model inference.
```bash
python -m scripts.cache_embeddings
```
This will save `.pt` files in `outputs/cache_parts/`.

### 2. Compute Baseline
Calculate the mean and inverse covariance matrix from the ID training data embeddings. It evaluates multiple normalization modes (Raw, L2, LayerNorm and L2 + LayerNorm).
```bash
python -m scripts.compute_baseline
```
This generates `outputs/baseline_stats.pt`.

### 3. Calculate OOD Scores
Compute OOD scores using the Mahalanobis Distance for all samples in the ID and OOD benchmarks.
```bash
python -m scripts.calculate_ood_scores
```
This produces `outputs/final_ood_scores.csv`.

### 4. Compute Metrics
Evaluate the performance by calculating metrics such as AUROC and FPR@95%TPR for each OOD dataset.
```bash
python -m scripts.compute_metrics.py
```
Results are saved to `outputs/fpr95tpr.csv`.

### 5. Visualize Results
Results can be visualized by executing the cells in the notebook `notebooks/results_visualization.ipynb`.
Each cell can be executed independently of other cells. The resulting figures are saved in `outputs`. 

## Project Structure

- `src/`: Core logic and modules.
- `scripts/`: Execution scripts for the pipeline.
- `outputs/`: Directory for stored embeddings, baselines and results.
- `logs/`: Execution logs.
- `notebooks/`: Exploratory analysis and visualization.
