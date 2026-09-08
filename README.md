# HyBINN: A Hybrid Biologically-Informed Neural Network for Cancer Survival Prediction

Personal research project under the mentorship of Dr. Hamed Akbari, Santa Clara University.

Project report forthcoming.

## Neural Network Architecture

The hybrid network has 3 input branches:

1. Sparse BINN <—— Genes mapped to pathways in Reactome
2. Gene MLP <—— All other genes (not mapped in Reactome)
3. Clinical MLP <—— clinical phenotype (T/N/M stages)

The model can be run using any combination of these branches. Each branch produces per-patient risk scores. If multiple branches are used, their risk scores are combined with learned weights into a final risk score per-patient.

## Repo Layout

* `configs/`: universal model architecture and hyperparameters
* `data/`: raw and processed gene datasets, reactome data, gene table
* `experiments/`: master scripts to run experiments across seeds / model configurations, and process the results
  * `figures/`: output figures from `analyze_results.py`
  * `runs/`: logged outputs from all configuration-seed combos (saved .pt model files are excluded due to their large size)
* `notebooks/`: jupyter notebooks for data preprocessing and legacy data visualization
* `src/`: model source code
  * `datasets/`: defines dataset class and dataloader creation
  * `models/`: PyTorch model classes for each branch
  * `processing/`: mapped/unmapped gene split, pathway mask
  * `training/`: cox loss and main training/testing functions
  * `utils/`: cindex bootstrapping, stratified splitting, logging, loading
