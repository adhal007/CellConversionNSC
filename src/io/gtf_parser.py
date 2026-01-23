"""
GTF file parsing utilities.

This module handles parsing of GTF/GFF files for:
- Gene ID to symbol mappings
- Promoter region extraction
- Gene annotations
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path


def parse_gtf_gene_mapping(gtf_path: str) -> Dict[str, str]:
    """
    Parse GTF file to create Ensembl ID -> Gene Symbol mapping.
    
    Parameters
    ----------
    gtf_path : str
        Path to GTF file
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping Ensembl IDs (without version) to gene symbols
        
    Examples
    --------
    >>> mappings = parse_gtf_gene_mapping("genome.gtf")
    >>> mappings["ENSMUSG00000000001"]
    'Gnai3'
    """
    print(f"      Parsing GTF for gene mappings...")
    
    ensembl_to_symbol = {}
    
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            # Only parse gene entries
            if fields[2] != 'gene':
                continue
            
            attributes = fields[8]
            
            # Extract gene_id and gene_name
            gene_id = None
            gene_name = None
            
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    # gene_id "ENSMUSG00000000001.5"
                    gene_id = attr.split('"')[1].split('.')[0]  # Remove version
                elif attr.startswith('gene_name'):
                    # gene_name "Gnai3"
                    gene_name = attr.split('"')[1]
            
            if gene_id and gene_name:
                ensembl_to_symbol[gene_id] = gene_name
    
    print(f"      Parsed {len(ensembl_to_symbol)} gene mappings from GTF")
    return ensembl_to_symbol


def get_promoters_from_gtf(gtf_path: str, window: int = 2000) -> pd.DataFrame:
    """
    Extract promoter regions (UPSTREAM of TSS) from GTF.
    
    The promoter is defined as the region UPSTREAM of the transcription start site (TSS):
    - For + strand genes: promoter is BEFORE the gene start
    - For - strand genes: promoter is AFTER the gene end
    
    Parameters
    ----------
    gtf_path : str
        Path to GTF file
    window : int, default=2000
        Size of promoter window in base pairs upstream of TSS
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chr, start, end, gene, strand, tss
        
    Examples
    --------
    >>> promoters = get_promoters_from_gtf("genome.gtf", window=2000)
    >>> promoters.head()
    """
    promoters = []
    
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if fields[2] != 'gene':
                continue
            
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            
            # Extract gene symbol
            attrs = {}
            for item in fields[8].split(';'):
                if not item.strip():
                    continue
                key_val = item.strip().split(' ', 1)
                if len(key_val) == 2:
                    attrs[key_val[0]] = key_val[1].strip('"')
            
            gene_name = attrs.get('gene_name', '')
            
            # Get TSS and promoter based on strand
            if strand == '+':
                tss = start
                # Promoter is UPSTREAM (before gene start)
                promoter_start = max(0, tss - window)
                promoter_end = tss
            else:  # - strand
                tss = end
                # Promoter is UPSTREAM (after gene end in coordinates)
                promoter_start = tss
                promoter_end = tss + window
            
            promoters.append({
                'chr': chrom,
                'start': promoter_start,
                'end': promoter_end,
                'gene': gene_name,
                'strand': strand,
                'tss': tss
            })
    
    return pd.DataFrame(promoters)


def parse_gtf_attributes(attribute_string: str) -> Dict[str, str]:
    """
    Parse GTF attribute string into dictionary.
    
    Parameters
    ----------
    attribute_string : str
        GTF attribute field (column 9)
        
    Returns
    -------
    Dict[str, str]
        Dictionary of attribute key-value pairs
        
    Examples
    --------
    >>> attrs = parse_gtf_attributes('gene_id "ENSG001"; gene_name "TP53"')
    >>> attrs['gene_id']
    'ENSG001'
    """
    attrs = {}
    for item in attribute_string.split(';'):
        if not item.strip():
            continue
        key_val = item.strip().split(' ', 1)
        if len(key_val) == 2:
            attrs[key_val[0]] = key_val[1].strip('"')
    return attrs
