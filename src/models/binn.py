import torch
import torch.nn as nn
import torch.nn.functional as F


# StandaloneBINN - outputs cox PH directly for now - not the branch of hybrid model
class StandaloneBINN(nn.Module):
    
    def __init__(self, in_nodes, pathway_nodes, hidden_nodes, out_nodes, pathway_mask):
        super(StandaloneBINN, self).__init__()
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)  # tune p
        # save pathway mask
        self.register_buffer("mask", torch.tensor(pathway_mask))
        
        # input genes --> pathway layer
        self.sc1 = nn.Linear(in_nodes, pathway_nodes)
        # pathway layer --> hidden layer (superpathways)
        self.sc2 = nn.Linear(pathway_nodes, hidden_nodes)
        # hidden layer --> hidden layer 2 (embedding)
        self.sc3 = nn.Linear(hidden_nodes, out_nodes, bias=False)
        # hidden layer 2 --> output cox layer
        self.sc4 = nn.Linear(out_nodes, 1, bias=False)
        self.sc4.weight.data.uniform_(-0.001, 0.001)    # start CoxPH weights close to 0
    
    
    def forward(self, x):
        # apply the mask matrix and update sc1's weights
        masked_weights = self.sc1.weight * self.mask
        
        # forward pass
        # mask input genes --> pathway layer
        x = self.tanh(F.linear(x, masked_weights, self.sc1.bias))
        # following layers are normal
        x = self.tanh(self.sc2(x))
        x = self.tanh(self.sc3(x))
        x = self.sc4(x)
        return x


# TEMP TESTING
import pandas as pd
from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes

df = pd.read_csv("../../data/processed/data.csv")

pathway_map = build_reactome_map("../../data/reactome/Ensembl2Reactome_All_Levels.txt")
mapped, unmapped, valid_pathways = split_genes(df, pathway_map)
mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)

net = StandaloneBINN(len(gene_labels), len(pathway_labels), 100, 30, mask)