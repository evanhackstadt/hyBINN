import os
import pandas as pd
import numpy as np
from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from models.binn import StandaloneBINN
from datasets.dataset import get_dataloaders
from training.trainer import train

data_path = "../data/processed/data.csv"
df = pd.read_csv(data_path, index_col=0)

times = df['OS.time'].to_numpy(dtype=np.float32)
events = df['OS'].to_numpy(dtype=np.float32)
gene_df = df.drop(columns=['OS.time', 'OS'])

pathway_map = build_reactome_map("../data/reactome/Ensembl2Reactome_All_Levels.txt")
mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)

# split genes
x_mapped = gene_df[mapped].to_numpy(dtype=np.float32)
x_unmapped = gene_df[unmapped].to_numpy(dtype=np.float32)

# dimensions check
assert x_mapped.shape[1] == mask.shape[0], \
    f"Mapped gene mismatch: {x_mapped.shape[1]} vs {mask.shape[0]}"


model = StandaloneBINN(in_nodes=len(gene_labels),
                       pathway_nodes=len(pathway_labels),
                       hidden_nodes=64,
                       out_nodes=24,
                       pathway_mask=mask)

train_prop, val_prop, test_prop = 0.7, 0.15, 0.15

train_loader, val_loader, test_loader = get_dataloaders(x_mapped, x_unmapped, times, events,
                                                        train=train_prop, val=val_prop, test=test_prop, 
                                                        batch_size=128, random_seed=42)

dir_path = os.path.abspath("runs/binn_20260316_lr1e3_wd1e4")
run_name = "run3"

log_path = os.path.join(dir_path, f"{run_name}.log")
results_path = os.path.join(dir_path, f"{run_name}.csv")

train_loss, val_loss, cindexes = train(model=model, train_loader=train_loader, val_loader=val_loader,
                                     train_proportion=train_prop, val_proportion=val_prop, logfile=log_path,
                                     epochs=15, weight_decay=1e-4, stop_early_patience=7)

results = pd.DataFrame({
    'epoch': range(1, len(train_loss)+1),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'cindexes': cindexes
})

results.to_csv(results_path)


# ---- Baseline: CoxPH ----
from lifelines import CoxPHFitter

baseline_df = pd.DataFrame({
    'OS.time': df['OS.time'], 
    'OS': df['OS']
})

cph = CoxPHFitter().fit(baseline_df, duration_col='OS.time', event_col='OS')
cph.print_summary()