import os
import pandas as pd
import numpy as np
import torch
from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from models.binn import StandaloneBINN
from datasets.dataset import get_dataloaders
from training.trainer import train, test


# Load dataset
data_path = "../data/processed/data.csv"
df = pd.read_csv(data_path, index_col=0)

times = df['OS.time'].to_numpy(dtype=np.float32)
events = df['OS'].to_numpy(dtype=np.float32)
gene_df = df.drop(columns=['OS.time', 'OS'])

# Map pathways and split genes
pathway_map = build_reactome_map("../data/reactome/Ensembl2Reactome_All_Levels.txt")
mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)

x_mapped = gene_df[mapped].to_numpy(dtype=np.float32)
x_unmapped = gene_df[unmapped].to_numpy(dtype=np.float32)

# Dimensions check
assert x_mapped.shape[1] == mask.shape[0], \
    f"Mapped gene mismatch: {x_mapped.shape[1]} vs {mask.shape[0]}"


# Create model and dataloaders
model = StandaloneBINN(in_nodes=len(gene_labels),
                       pathway_nodes=len(pathway_labels),
                       hidden_nodes=64,
                       out_nodes=24,
                       pathway_mask=mask)

train_prop, val_prop, test_prop = 0.7, 0.15, 0.15

train_loader, val_loader, test_loader = get_dataloaders(x_mapped, x_unmapped, times, events,
                                                        train=train_prop, val=val_prop, test=test_prop, 
                                                        batch_size=128, random_seed=42)

# Path setup
dir_path = os.path.abspath("runs/binn_20260316_lr1e3_wd1e4")
run_name = "run4"

log_path = os.path.join(dir_path, f"{run_name}.log")
results_path = os.path.join(dir_path, f"{run_name}.csv")
best_model_path = os.path.join(dir_path, "best_model.pt")


# TRAINING
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

# TESTING
avg_test_loss, test_cindex = test(model=model, test_loader=test_loader, 
                                  best_model_file=best_model_path, logfile=log_path)


# INTERPRETABILITY
# examine stored pathway layer activations on test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_activations = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        X = batch['X_mapped'].to(device)
        _ = model(X)
        all_activations.append(model.pathway_activations.detach().cpu())

# Average activation per pathway across all test patients
mean_activations = torch.cat(all_activations, dim=0).mean(dim=0).numpy()
top_indices = np.argsort(mean_activations)[::-1][:10]
top_pathways = [pathway_labels[i] for i in top_indices]

print("Top 10 activated pathways:")
for rank, (idx, name) in enumerate(zip(top_indices, top_pathways)):
    print(f"  {rank+1}. {name}  (mean activation: {mean_activations[idx]:.4f})")



# ---- Baseline: CoxPH ----
from lifelines import CoxPHFitter

baseline_df = pd.DataFrame({
    'OS.time': df['OS.time'], 
    'OS': df['OS']
})

cph = CoxPHFitter().fit(baseline_df, duration_col='OS.time', event_col='OS')
cph.print_summary()

# TODO:
#   take in CLI args for runs? or automatically increment?