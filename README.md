# Currently: CSCI 184 Final Project
Final project for CSCI 184: Advanced Machine Learning. 

A Biologically-Informed Neural Network for Breast Cancer Survival Prediction.
The model:
1. Takes gene expression (RNAseq) values as an input
2. Uses a sparsely connected first layer that maps input genes to known pathways
3. Moves through two more hidden layers
4. Outputs a risk score for that patient's survival (an input into Cox Proportional Hazards)

This repo consists of all files used to preprocess data, build the model, and train and test the model.

### Repo Layout:
```
hybinn/
│
├── data/
│   ├── raw/ — raw breast cancer data downloaded from UCSC Xena
│   ├── processed/ — data saved during preprocessing, including the main data.csv used for training
│   └── reactome/ — raw data downloaded from Reactome
│
├── src/
│   ├── datasets/dataset.py — define Dataset class and get_dataloaders() function
│   │
│   ├── models/
│   │   └── binn.py — defines the PyTorch model class
│   │
│   ├── training/
│   │   ├── trainer.py — contains functions to train, validate, and test the model
│   │   └── loss.py — custom survival loss function that wraps CoxPH loss from pycox
│   │
│   ├── preprocessing/
│   │   ├── gene_split.py — filters and splits genes into those mapped to pathways by Reactome, and those not
│   │   └── reactome_processing.py — maps genes to pathways and builds mask matrix for the sparse layer
│   │
│   ├── utils/
│   │   └── logging.py — function to return basic logger object
│
├── experiments/
│   ├── runs/ — contains logs and results from different training runs
│   └── train_hybinn.py — primary script that creates, trains, tests, and logs a model
│
├── notebooks/
│   ├── preprocess_data.ipynb — one-time data pipeline that turns raw UCSC Xena data into data.csv
│   └── plot_losses.ipynb - notebook to manually create figures / graphs from the results
│
└── README.md
```


# Future HyBINN Project
Research project to create an attentive hybrid biologically-informed neural network for breast cancer survival prediction.

### Neural Network Architecture

The network will use 3 input streams:
1. Sparse BINN <—— Genes mapped to pathways in Reactome
2. Dense MLP <—— All other genes (not mapped in Reactome)
3. Clinical MLP <—— Clinical variables

The three streams will be merged into a single output node using an attention mechanism, so the network learns which inputs are more important based on the patient.
The final output node provides a CoxPH survival prediction.
