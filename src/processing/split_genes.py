import pandas as pd


# FILTER MAPPED GENES - helper function
def filter_pathway_map(pathway_map: dict, mapped_genes: list):
    """
    Filters mapped_genes to remove genes belonging only to pathways with <15 or >300 genes
    
    Args:
        pathway_map (dict): the Reactome mapping between genes and their pathways from reactome.py build_reactome_map()
        mapped_genes (list): the list of mapped genes to be filtered
    
    Returns:
        filtered_genes: the filtered list of mapped genes
        valid_pathways: the list of valid pathways (that have >=15 and <=300 genes mapped to them)
    """
    
    mapped_set = set(mapped_genes)
    
    # invert dict to get pathways : mapped genes
    pathways_to_genes = {}
    for gene, pathways in pathway_map.items():
        if gene not in mapped_set:
            continue
        else:
            for p in pathways:
                pathways_to_genes.setdefault(p, []).append(gene)
    
    # identify valid pathways (must have >15 and <300 mapped genes)
    valid_pathways = {p for p, genes in pathways_to_genes.items()
                      if 15 <= len(genes) <= 300}
    
    # filter to only genes that have at least one valid pathway
    filtered_genes = [g for g in mapped_genes
                      if any(p in valid_pathways for p in pathway_map[g])]
    
    return filtered_genes, valid_pathways


# SPLIT GENES - main function
def split_genes(expr_df: pd.DataFrame, pathway_map: dict):
    """
    Splits dataset genes into those mapped by Reactome and those not
    
    Args:
        expr_df (DataFrame): the processed gene expression data where columns are Ensembl IDs of genes
        pathway_map (dict): the Reactome mapping between genes and their pathways from reactome.py build_reactome_map()
    
    Returns:
        mapped_genes: a list of the Ensembl IDs in our dataset AND Reactome.
        unmapped_genes: a list of the Ensembl IDs in our dataset but not in Reactome.
        valid_pathways: a list of ReactomePathwayIDs with 15 <= genes <= 300 mapped to them
    """
    
    dataset_genes = expr_df.columns
    pathway_genes = pathway_map.keys()
    
    mapped_genes = list(set(dataset_genes) & set(pathway_genes))    # set intersection
    unmapped_genes = list(set(dataset_genes) - set(pathway_genes))  # set difference
    
    # call filter function and move any filtered-out genes to unmapped_genes
    filtered_genes, valid_pathways = filter_pathway_map(pathway_map, mapped_genes)
    removed = list(set(mapped_genes) - set(filtered_genes))
    unmapped_genes += removed
    
    return filtered_genes, unmapped_genes, valid_pathways