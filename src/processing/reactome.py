import pandas as pd
import numpy as np


# BUILD REACTOME MAP
# https://download.reactome.org/95/Ensembl2Reactome_All_Levels.txt
def build_reactome_map(path):
    """
    Creates a mapping between Ensembl IDs and Reactome Pathways
    
    Args:
        path (str): filepath to the Ensembl2Reactome text file from Reactome
    
    Returns:
        pathway_map (dict): given the Ensembl ID of a gene, what Reactome pathways does it map to
    """
    
    df = pd.read_csv(path, delimiter='\t',
                     names=["EnsemblID", "ReactomePathwayID", "URL", 
                            "PathwayName", "Evidence", "Species"])
    
    # Filter to only human pathways
    df2 = df[df["Species"] == "Homo sapiens"]
    
    # Initial filter to pathways with 15 ≤ n_genes ≤ 300
    # Will perform this filter again based on dataset genes
    pathway_sizes = df2.groupby("ReactomePathwayID")["EnsemblID"].nunique()
    valid_pathways = pathway_sizes[(pathway_sizes >= 15) & (pathway_sizes <= 300)].index
    df3 = df2[df2["ReactomePathwayID"].isin(valid_pathways)]
    
    # Filter to only EnsemblIDs we are using
    indices_to_drop = [row.Index for row in df3.itertuples()
                       if not row.EnsemblID.startswith("ENSG")]
    df4 = df3.drop(index=indices_to_drop)
    
    # Aggregate pathways per gene
    pathway_map = df4.groupby("EnsemblID")["ReactomePathwayID"].apply(list).to_dict()
    
    return pathway_map


# BUILD SPARSE MASK MATRIX
# This will zero-out weights between genes and pathways they aren't part of
def build_mask_matrix(mapped_genes, pathway_map, valid_pathways):
    """
    Builds the mask matrix that determines the sparse mapping of genes to pathway nodes
    
    Args:
        mapped_genes (list): the filtered list of Ensembl IDs that map to Reactome pathways
        pathway_map (dict): a dictionary mapping Ensemble IDs to lists of Reactome pathways
        valid_pathways (list): the list of ReactomePathwayIDs from filter_pathway_map
    
    Returns:
        mask_matrix (ndarray): a binary matrix of size [n_mapped_genes x n_pathways]; M[i,j] = 1 if gene i is part of pathway j
        sorted_genes: mapped_genes, sorted - the row index of the matrix
        sorted_pathways: valid_pathways, sorted - the columns of the matrix
    """
    
    # row / column labels
    sorted_genes = sorted(mapped_genes)
    sorted_pathways = sorted(valid_pathways)
    
    # row / column index lookups
    gene_idx = {val: idx for idx, val in enumerate(sorted_genes)}
    pathway_idx = {val: idx for idx, val in enumerate(sorted_pathways)}
    
    M = np.zeros(shape=[len(mapped_genes), len(valid_pathways)], dtype=np.float32)
    
    for gene, pathways in pathway_map.items():
        if gene not in gene_idx:
            continue
        for pathway in pathways:
            if pathway not in pathway_idx:
                continue
            M[gene_idx[gene], pathway_idx[pathway]] = 1.0
    
    return M, sorted_genes, sorted_pathways