"""
Backward compatibility wrapper for pipeline imports.
This allows old code using 'import src.pipeline' to work with the new modular structure.
"""

# Import from the new location
from src.engine.pipeline import NSCAnalysis

# Make it available at the old import path
__all__ = ['NSCAnalysis']

# # ==============================================================================
# # NSCAnalysis Class - Step 1: Data Loading
# # ==============================================================================

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from typing import Optional, Tuple, List, Dict, Set
# # Add these imports at the top
# from pydeseq2.dds import DeseqDataSet
# from pydeseq2.ds import DeseqStats
# from pydeseq2.preprocessing import deseq2_norm_transform
# # Add this import at the top of the class file
# from src.models.stats.bulk_gjsd import BulkGJSD
# import pybedtools
# from scipy import stats
# from statsmodels.stats.multitest import multipletests
# import pyranges as pr

# class NSCAnalysis:
#     """
#     Analysis pipeline for NSC cell conversion project.
    
#     Questions addressed:
#         1. What are pro-neural and pro-glial signatures?
#         2. What factors can convert glial to neuronal?

#     Enhanced with loading of the Chip seq bed files
#     """
# ###############################################################################################################
# ############################### INPUT DATA LOADING IS HERE ####################################################
# ###############################################################################################################  
#     def __init__(
#         self,
#         counts_path: str,
#         metadata_path: str,
#         atac_metadata_path: str,
#         overlap_df_path: str,
#         chip_annotated_path: str,
#         atac_annotated_path: str,
#         tf_list_path: str,
#         gtf_path: str,
#         chip_peaks_path: str = None,  # NEW
#         exclude_samples: List[str] = None
#     ):
#         # Store paths
#         self.paths = {
#             'counts': Path(counts_path),
#             'metadata': Path(metadata_path),
#             'atac_metadata': Path(atac_metadata_path),
#             'overlap_df': Path(overlap_df_path),
#             'chip_annotated': Path(chip_annotated_path),
#             'atac_annotated': Path(atac_annotated_path),
#             'tf_list': Path(tf_list_path),
#             'gtf': Path(gtf_path),
#             'chip_peaks': Path(chip_peaks_path) if chip_peaks_path else None  # NEW
#         }
        
        
#         # Validate paths exist
#         for name, path in self.paths.items():
#             if not path.exists():
#                 raise FileNotFoundError(f"{name}: {path}")
        
#         # Store exclude list
#         self.exclude_samples = exclude_samples or []
        
#         # Data containers
#         self.counts = None
#         self.counts_unfiltered = None
#         self.metadata = None
#         self.atac_metadata = None  # <-- NEW
#         self.merged_metadata = None  # <-- NEW
#         self.atac_to_rna_map = None  # <-- NEW
#         self.overlap_df = None
#         self.chip_annotated = None
#         self.atac_annotated = None
#         self.tf_symbols = None
#         self.tf_ensembl = None
#         self.ensembl_to_symbol = None
        
#         # Results containers
#         self.deseq_results = None
#         self.gjsd_results = None
#         self.dar_results = None
#         self.consensus_peaks = None
#         self.atac_signal_matrix = None
#         self.diff_overlap_df = None
#         self.gene_lengths = None
#         self.tpm = None
#         self.fpkm = None
        
#         # NEW containers
#         self.promoters = None
#         self.chip_peaks = None
        
#         # Load data
#         self._load_data()


#     def _load_data(self):
#         """Load all input data."""
#         print("=" * 60)
#         print("Loading data...")
#         print("=" * 60)
        
#         # 1. Load counts
#         print("\n[1/9] Loading counts...")
#         self.counts = pd.read_csv(self.paths['counts'], sep=';', index_col=1)
#         self.counts = self.counts.drop(columns=['Unnamed: 0'])
        
#         nan_genes = self.counts.isna().any(axis=1).sum()
#         if nan_genes > 0:
#             print(f"      Dropping {nan_genes} genes with NaN values")
#             self.counts = self.counts.dropna()
        
#         print(f"      {self.counts.shape[0]} genes x {self.counts.shape[1]} samples")
        
#         # 2. Load RNA-seq metadata
#         print("\n[2/9] Loading RNA-seq metadata...")
#         self.metadata = pd.read_csv(self.paths['metadata'], sep=';')
#         self.metadata = self.metadata.set_index('SampleID')
#         print(f"      {self.metadata.shape[0]} samples")
#         print(f"      Columns: {list(self.metadata.columns)}")
        
#         # 3. Load ATAC-seq metadata
#         print("\n[3/9] Loading ATAC-seq metadata...")
#         self.atac_metadata = pd.read_csv(self.paths['atac_metadata'], sep=',')
#         self.atac_metadata = self.atac_metadata.set_index('SampleID')
#         print(f"      {self.atac_metadata.shape[0]} samples")
#         print(f"      Columns: {list(self.atac_metadata.columns)}")
        
#         # 4. Create merged metadata and mapping
#         print("\n[4/9] Creating ATAC-to-RNA sample mapping...")
#         self._create_sample_mapping()
        
#         # 5. Remove outlier samples
#         if self.exclude_samples:
#             print(f"\n[5/9] Removing outlier samples...")
#             samples_to_remove = [s for s in self.exclude_samples if s in self.counts.columns]
#             if samples_to_remove:
#                 self.counts = self.counts.drop(columns=samples_to_remove)
#                 self.metadata = self.metadata.drop(index=[s for s in samples_to_remove if s in self.metadata.index])
#                 print(f"      Removed: {samples_to_remove}")
#                 print(f"      Remaining: {self.counts.shape[1]} samples")
#             else:
#                 print(f"      No matching samples found to remove")
#         else:
#             print("\n[5/9] No outlier samples to remove")
        
#         # 6. Filter low-count genes
#         print("\n[6/9] Filtering low-count genes...")
#         self._filter_low_counts(min_counts=10, min_samples=3, group_col='Stage')
        
#         # 7. Load overlap_df
#         print("\n[7/9] Loading overlap_df (ChIP ∩ ATAC)...")
#         self.overlap_df = pd.read_csv(self.paths['overlap_df'], sep='\t')
#         print(f"      {self.overlap_df.shape[0]} overlaps")
#         print(f"      Columns: {list(self.overlap_df.columns)}")
        
#         # 8. Load chip_annotated
#         print("\n[8/9] Loading chip_annotated...")
#         self.chip_annotated = pd.read_csv(self.paths['chip_annotated'], sep='\t')
#         print(f"      {self.chip_annotated.shape[0]} peaks")
        
#         # 9. Load atac_annotated
#         print("\n[9/9] Loading atac_annotated...")
#         self.atac_annotated = pd.read_csv(self.paths['atac_annotated'], sep='\t')
#         print(f"      {self.atac_annotated.shape[0]} peaks")
        
#         # 10. Load TF list
#         print("\n[10/10] Loading TF list...")
#         tf_df = pd.read_excel(self.paths['tf_list'])
#         tf_df = tf_df.dropna()
#         tf_df.columns = tf_df.iloc[0, :]
#         tf_df = tf_df.iloc[1:, :].reset_index(drop=True)
#         self.tf_df = tf_df
#         self.tf_symbols = set(tf_df['Gene Symbol'].str.strip().tolist())
#         self.tf_ensembl = set(tf_df['Ensembl ID'].str.strip().tolist())
#         print(f"      {len(self.tf_symbols)} TFs")
        
#         # Check TFs remaining after filtering
#         tfs_in_counts = len([g for g in self.counts.index if g in self.tf_ensembl])
#         print(f"      TFs in filtered counts: {tfs_in_counts}")
        
#         # 11. Parse GTF for gene mappings
#         print("\n[11/11] Parsing GTF for gene symbol mappings...")
#         self.ensembl_to_symbol = self._parse_gtf_gene_mapping()
        
#         mapped = sum(1 for g in self.counts.index if g in self.ensembl_to_symbol)
#         print(f"      Counts genes with mapping: {mapped} / {len(self.counts)}")
        
#         print("\n" + "=" * 60)
#         print("Data loaded successfully!")
#         print("=" * 60)

#         # 12. Generate promoters from GTF
#         print("\n[12/13] Generating promoters from GTF...")
#         self.promoters = self._get_promoters_from_gtf(window=2000)
#         print(f"      {len(self.promoters)} promoters")
        
#         # 13. Load ChIP peaks (if provided)
#         if self.paths['chip_peaks'] and self.paths['chip_peaks'].exists():
#             print("\n[13/13] Loading ChIP-seq peaks...")
#             self.chip_peaks = self._load_chip_peaks()
#             print(f"      {len(self.chip_peaks)} peaks")
#             print(f"      {self.chip_peaks['TF'].nunique()} unique TFs")
#         else:
#             print("\n[13/13] No ChIP peaks file provided, skipping...")
#             self.chip_peaks = None
        
#         print("\n" + "=" * 60)
#         print("All data loaded successfully!")
#         print("=" * 60)

# ###############################################################################################################
# ############################### ALL LOADING AND REGULATORY REGION CREATION HERE ###############################
# ###############################################################################################################  

# ######## Defining our regulatory regions to be used in Base GRN #########
# ######## 2 steps
# ######## 1. Load GTF file and enhancer atlas (downloaded)
# ######## 2. annotate them into promoters, enhancers and proximal promoters 
# ######## NOTE: Many enhancers could be promoters for different genes 
# ######## Proximal promoters are defined as those with enhancer and promoter overlap for the same Gene only. 
# ######## Finallly we would get a dataframe of
# # 'chr': prom['chr'],
# # 'start': prom['start'],
# # 'end': prom['end'],
# # 'gene': prom['gene'],
# # 'region_type': 'promoter'
# ##############################################################################################################
#     def _parse_gtf_gene_mapping(self) -> Dict[str, str]:
#         """Parse GTF file to create Ensembl ID -> Gene Symbol mapping."""
#         print("      Parsing GTF for gene mappings...")
        
#         ensembl_to_symbol = {}
        
#         with open(self.paths['gtf'], 'r') as f:
#             for line in f:
#                 if line.startswith('#'):
#                     continue
                
#                 fields = line.strip().split('\t')
#                 if len(fields) < 9:
#                     continue
                
#                 # Only parse gene entries
#                 if fields[2] != 'gene':
#                     continue
                
#                 attributes = fields[8]
                
#                 # Extract gene_id and gene_name
#                 gene_id = None
#                 gene_name = None
                
#                 for attr in attributes.split(';'):
#                     attr = attr.strip()
#                     if attr.startswith('gene_id'):
#                         # gene_id "ENSMUSG00000000001.5"
#                         gene_id = attr.split('"')[1].split('.')[0]  # Remove version
#                     elif attr.startswith('gene_name'):
#                         # gene_name "Gnai3"
#                         gene_name = attr.split('"')[1]
                
#                 if gene_id and gene_name:
#                     ensembl_to_symbol[gene_id] = gene_name
        
#         print(f"      Parsed {len(ensembl_to_symbol)} gene mappings from GTF")
#         return ensembl_to_symbol
    
#     def _get_promoters_from_gtf(self, window=2000):
#         """
#         Extract promoter regions (UPSTREAM of TSS) from GTF.
        
#         Promoter = window bp UPSTREAM of TSS
#         """
#         promoters = []
        
#         with open(self.paths['gtf']) as f:
#             for line in f:
#                 if line.startswith('#'):
#                     continue
                
#                 fields = line.strip().split('\t')
#                 if fields[2] != 'gene':
#                     continue
                
#                 chrom = fields[0]
#                 start = int(fields[3])
#                 end = int(fields[4])
#                 strand = fields[6]
                
#                 # Extract gene symbol
#                 attrs = {}
#                 for item in fields[8].split(';'):
#                     if not item.strip():
#                         continue
#                     key_val = item.strip().split(' ', 1)
#                     if len(key_val) == 2:
#                         attrs[key_val[0]] = key_val[1].strip('"')
                
#                 gene_name = attrs.get('gene_name', '')
                
#                 # Get TSS and promoter based on strand
#                 if strand == '+':
#                     tss = start
#                     # Promoter is UPSTREAM (before gene start)
#                     promoter_start = max(0, tss - window)
#                     promoter_end = tss
#                 else:  # - strand
#                     tss = end
#                     # Promoter is UPSTREAM (after gene end in coordinates)
#                     promoter_start = tss
#                     promoter_end = tss + window
                
#                 promoters.append({
#                     'chr': chrom,
#                     'start': promoter_start,
#                     'end': promoter_end,
#                     'gene': gene_name,
#                     'strand': strand,
#                     'tss': tss
#                 })
        
#         return pd.DataFrame(promoters)
# ##############################################################################################################
# ##############################################################################################################
#     def create_merged_regulatory_regions(self, promoters, enhancers):
#         """
#         Merge promoters and enhancers into unified regulatory regions.
        
#         SIMPLE LOGIC:
#         1. Same-gene overlap → Merge into "proximal_regulatory"
#         2. Everything else → Keep as-is ("promoter" or "enhancer")
        
#         Different-gene overlaps are FINE - both regions stay separate.
#         """
#         print("\n" + "="*80)
#         print("CREATING MERGED REGULATORY REGIONS")
#         print("="*80)
        
#         print(f"Input:")
#         print(f"  Promoters: {len(promoters):,}")
#         print(f"  Enhancers: {len(enhancers):,}")
        
#         # Convert to PyRanges
#         promoters_pr = pr.PyRanges(promoters.rename(
#             columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}
#         ))
        
#         enhancers_pr = pr.PyRanges(enhancers.rename(
#             columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End', 'symbol': 'gene'}
#         ))
        
#         # Find overlaps
#         all_overlaps = promoters_pr.join(enhancers_pr).df
        
#         # Get SAME-GENE overlaps
#         same_gene_overlaps = all_overlaps[
#             all_overlaps['gene'] == all_overlaps['gene_b']
#         ].copy()
        
#         print(f"\nSame-gene promoter-enhancer overlaps: {len(same_gene_overlaps):,}")
        
#         # Track which promoters/enhancers are in same-gene overlaps
#         promoters_in_proximal = set(
#             zip(same_gene_overlaps['gene'], 
#                 same_gene_overlaps['Chromosome'],
#                 same_gene_overlaps['Start'], 
#                 same_gene_overlaps['End'])
#         )
        
#         enhancers_in_proximal = set(
#             zip(same_gene_overlaps['gene_b'],
#                 same_gene_overlaps['Chromosome'],
#                 same_gene_overlaps['Start_b'], 
#                 same_gene_overlaps['End_b'])
#         )
        
#         # 1. Create proximal regulatory regions (merge same-gene overlaps)
#         proximal_regions = []
#         for _, row in same_gene_overlaps.iterrows():
#             merged_start = min(row['Start'], row['Start_b'])
#             merged_end = max(row['End'], row['End_b'])
            
#             proximal_regions.append({
#                 'chr': row['Chromosome'],
#                 'start': merged_start,
#                 'end': merged_end,
#                 'gene': row['gene'],
#                 'region_type': 'proximal_regulatory'
#             })
        
#         proximal_df = pd.DataFrame(proximal_regions).drop_duplicates()
        
#         # 2. Keep promoters that DON'T have same-gene overlap
#         promoter_regions = []
#         for _, prom in promoters.iterrows():
#             prom_key = (prom['gene'], prom['chr'], prom['start'], prom['end'])
#             if prom_key not in promoters_in_proximal:
#                 promoter_regions.append({
#                     'chr': prom['chr'],
#                     'start': prom['start'],
#                     'end': prom['end'],
#                     'gene': prom['gene'],
#                     'region_type': 'promoter'
#                 })
        
#         promoter_df = pd.DataFrame(promoter_regions)
        
#         # 3. Keep enhancers that DON'T have same-gene overlap
#         enhancer_regions = []
#         for _, enh in enhancers.iterrows():
#             enh_key = (enh['symbol'], enh['chr'], enh['start'], enh['end'])
#             if enh_key not in enhancers_in_proximal:
#                 enhancer_regions.append({
#                     'chr': enh['chr'],
#                     'start': enh['start'],
#                     'end': enh['end'],
#                     'gene': enh['symbol'],
#                     'region_type': 'enhancer'
#                 })
        
#         enhancer_df = pd.DataFrame(enhancer_regions)
        
#         # Combine
#         merged_regulatory = pd.concat([
#             promoter_df,
#             proximal_df,
#             enhancer_df
#         ], ignore_index=True)
        
#         print(f"\nOutput:")
#         print(f"  Promoter: {len(promoter_df):,}")
#         print(f"  Proximal regulatory: {len(proximal_df):,}")
#         print(f"  Enhancer: {len(enhancer_df):,}")
#         print(f"  Total: {len(merged_regulatory):,}")
        
#         return merged_regulatory

#     def _load_chip_peaks(self):
#         """Load ChIP-seq peaks from BED file."""
#         # Read file, skip track line
#         chip_peaks = pd.read_csv(self.paths['chip_peaks'], sep='\t', header=None, skiprows=1,
#                                  names=['chr', 'start', 'end', 'metadata', 'score', 'strand', 
#                                         'thickStart', 'thickEnd', 'itemRgb'])
        
#         # Extract TF name from metadata (Name=TF%20(@%20cell))
#         chip_peaks['TF'] = chip_peaks['metadata'].str.extract(r'Name=([^%]+)')[0]
#         chip_peaks['TF'] = chip_peaks['TF'].str.strip()
        
#         # Keep only needed columns
#         chip_peaks = chip_peaks[['chr', 'start', 'end', 'TF', 'score']].copy()
        
#         return chip_peaks
# ###############################################################################################################
# ############################### RNA-SEQ DATA NORMALIZATION METHODS ARE HERE ###################################
# ###############################################################################################################   
#     def _filter_low_counts(
#         self,
#         min_counts: int = 10,
#         min_samples: int = None,
#         min_samples_pct: float = None,
#         min_cpm: float = None,
#         group_col: str = None
#     ) -> pd.DataFrame:
#         """
#         Filter out lowly expressed genes.
        
#         Args:
#             min_counts: Minimum count threshold
#             min_samples: Minimum number of samples meeting threshold
#             min_samples_pct: Minimum percentage of samples (alternative to min_samples)
#             min_cpm: If set, use CPM threshold instead of raw counts
#             group_col: If set, require threshold in at least one group
            
#         Returns:
#             Filtered count matrix
#         """
#         print(f"\n{'='*60}")
#         print("Filtering low-count genes")
#         print(f"{'='*60}")
        
#         n_genes_before = len(self.counts)
#         n_samples = len(self.counts.columns)
        
#         # Determine min_samples
#         if min_samples is None and min_samples_pct is None:
#             # Default: smallest group size
#             if group_col and group_col in self.metadata.columns:
#                 min_samples = self.metadata[group_col].value_counts().min()
#             else:
#                 min_samples = max(3, int(n_samples * 0.1))  # 10% or at least 3
#         elif min_samples_pct is not None:
#             min_samples = max(1, int(n_samples * min_samples_pct))
        
#         print(f"Threshold: {'CPM > ' + str(min_cpm) if min_cpm else 'counts >= ' + str(min_counts)} in >= {min_samples} samples")
        
#         if group_col and group_col in self.metadata.columns:
#             # Filter per group: keep gene if it passes in ANY group
#             groups = self.metadata[group_col].unique()
#             keep_genes = set()
            
#             for group in groups:
#                 group_samples = self.metadata[self.metadata[group_col] == group].index
#                 group_samples = [s for s in group_samples if s in self.counts.columns]
#                 group_counts = self.counts[group_samples]
                
#                 if min_cpm:
#                     # CPM filtering
#                     lib_sizes = group_counts.sum(axis=0)
#                     cpm = group_counts.div(lib_sizes, axis=1) * 1e6
#                     passes = (cpm > min_cpm).sum(axis=1) >= min(min_samples, len(group_samples))
#                 else:
#                     # Count filtering
#                     passes = (group_counts >= min_counts).sum(axis=1) >= min(min_samples, len(group_samples))
                
#                 keep_genes.update(self.counts.index[passes])
#                 print(f"  {group}: {passes.sum()} genes pass filter")
            
#             keep_mask = self.counts.index.isin(keep_genes)
#         else:
#             # Simple filtering across all samples
#             if min_cpm:
#                 lib_sizes = self.counts.sum(axis=0)
#                 cpm = self.counts.div(lib_sizes, axis=1) * 1e6
#                 keep_mask = (cpm > min_cpm).sum(axis=1) >= min_samples
#             else:
#                 keep_mask = (self.counts >= min_counts).sum(axis=1) >= min_samples
        
#         # Apply filter
#         self.counts_unfiltered = self.counts.copy()  # Keep original
#         self.counts = self.counts[keep_mask]
        
#         n_genes_after = len(self.counts)
#         n_removed = n_genes_before - n_genes_after
        
#         print(f"\nGenes before: {n_genes_before}")
#         print(f"Genes after:  {n_genes_after}")
#         print(f"Removed:      {n_removed} ({100*n_removed/n_genes_before:.1f}%)")
        
#         # Update TF list if exists
#         if hasattr(self, 'tf_ensembl') and self.tf_ensembl is not None:
#             tfs_before = len([g for g in self.counts_unfiltered.index if g in self.tf_ensembl])
#             tfs_after = len([g for g in self.counts.index if g in self.tf_ensembl])
#             print(f"TFs before: {tfs_before}, after: {tfs_after}")

#     def _create_sample_mapping(self):
#         """
#         Create mapping between ATAC and RNA sample IDs based on 
#         Stage, Region, Marker, and Replicate.
#         """
#         # Standardize ATAC metadata columns to match RNA naming
#         atac = self.atac_metadata.copy()
        
#         # Map ATAC column names to RNA column names
#         # ATAC: Tissue, Factor, Condition, Replicate
#         # RNA: Region, Marker, Stage, Replicate
        
#         # Tissue -> Region
#         atac['Region'] = atac['Tissue'].map({'Cortex': 'Ctx', 'LGE': 'LGE'})
        
#         # Factor -> Marker
#         atac['Marker'] = atac['Factor'].map({'CD133': 'Progenitors', 'PSA-NCAM': 'PSANCAM'})
        
#         # Condition -> Stage
#         atac['Stage'] = atac['Condition']
        
#         # Build mapping
#         self.atac_to_rna_map = {}
#         unmatched = []
        
#         for atac_id in atac.index:
#             atac_row = atac.loc[atac_id]
            
#             match = self.metadata[
#                 (self.metadata['Stage'] == atac_row['Stage']) &
#                 (self.metadata['Region'] == atac_row['Region']) &
#                 (self.metadata['Marker'] == atac_row['Marker']) &
#                 (self.metadata['Replicate'] == atac_row['Replicate'])
#             ]
            
#             if len(match) == 1:
#                 self.atac_to_rna_map[atac_id] = match.index[0]
#             elif len(match) > 1:
#                 print(f"      Warning: Multiple matches for {atac_id}")
#                 self.atac_to_rna_map[atac_id] = match.index[0]  # Take first
#             else:
#                 unmatched.append(atac_id)
        
#         print(f"      Mapped: {len(self.atac_to_rna_map)} / {len(atac)} ATAC samples")
#         if unmatched:
#             print(f"      Unmatched ATAC samples: {unmatched}")
        
#         # Create merged metadata
#         atac['RNA_SampleID'] = atac.index.map(self.atac_to_rna_map)
#         self.merged_metadata = atac
        
#         # Store reverse mapping too
#         self.rna_to_atac_map = {v: k for k, v in self.atac_to_rna_map.items()}
    
#     def summary(self):
#         """Print summary of loaded data."""
#         print("\n=== NSCAnalysis Summary ===")
#         print(f"Counts: {self.counts.shape[0]} genes x {self.counts.shape[1]} samples")
#         print(f"Metadata: {self.metadata.shape[0]} samples")
#         print(f"  - Stages: {self.metadata['Stage'].unique().tolist()}")
#         print(f"  - Regions: {self.metadata['Region'].unique().tolist()}")
#         print(f"  - Markers: {self.metadata['Marker'].unique().tolist()}")
#         print(f"Overlap_df: {self.overlap_df.shape[0]} functional TF-gene bindings")
#         print(f"  - Unique TFs: {self.overlap_df['TF'].nunique()}")
#         print(f"  - Unique genes: {self.overlap_df['gene'].nunique()}")
#         print(f"ChIP peaks: {self.chip_annotated.shape[0]}")
#         print(f"ATAC peaks: {self.atac_annotated.shape[0]}")
#         print(f"TF list: {len(self.tf_symbols)} TFs")

#     @staticmethod
#     def calculate_tpm(
#         counts: pd.DataFrame,
#         gene_lengths: pd.Series
#     ) -> pd.DataFrame:
#         """
#         Calculate TPM (Transcripts Per Million).
        
#         TPM = (reads / gene_length_kb) / sum(reads / gene_length_kb) * 1e6
        
#         Args:
#             counts: Raw count matrix (genes x samples)
#             gene_lengths: Gene lengths in bp (index = gene_id)
            
#         Returns:
#             TPM normalized matrix
#         """
#         # Align gene lengths to counts
#         common_genes = counts.index.intersection(gene_lengths.index)
#         counts_aligned = counts.loc[common_genes]
#         lengths_aligned = gene_lengths.loc[common_genes]
        
#         print(f"Calculating TPM for {len(common_genes)} genes...")
        
#         # Convert to kb
#         lengths_kb = lengths_aligned / 1000
        
#         # RPK (reads per kilobase)
#         rpk = counts_aligned.div(lengths_kb, axis=0)
        
#         # Scale to million
#         tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
        
#         print(f"TPM range: {tpm.min().min():.2f} - {tpm.max().max():.2f}")
#         print(f"TPM sum per sample (should be ~1e6): {tpm.sum().mean():.0f}")
        
#         return tpm

#     @staticmethod
#     def calculate_fpkm(
#         counts: pd.DataFrame,
#         gene_lengths: pd.Series,
#         library_sizes: pd.Series = None
#     ) -> pd.DataFrame:
#         """
#         Calculate FPKM (Fragments Per Kilobase per Million mapped reads).
        
#         FPKM = (reads * 1e9) / (gene_length * total_reads)
        
#         Args:
#             counts: Raw count matrix (genes x samples)
#             gene_lengths: Gene lengths in bp (index = gene_id)
#             library_sizes: Total reads per sample (if None, calculated from counts)
            
#         Returns:
#             FPKM normalized matrix
#         """
#         # Align gene lengths to counts
#         common_genes = counts.index.intersection(gene_lengths.index)
#         counts_aligned = counts.loc[common_genes]
#         lengths_aligned = gene_lengths.loc[common_genes]
        
#         print(f"Calculating FPKM for {len(common_genes)} genes...")
        
#         # Library sizes
#         if library_sizes is None:
#             library_sizes = counts_aligned.sum(axis=0)
        
#         # FPKM calculation
#         fpkm = counts_aligned.div(lengths_aligned, axis=0).div(library_sizes, axis=1) * 1e9
        
#         print(f"FPKM range: {fpkm.min().min():.2f} - {fpkm.max().max():.2f}")
        
#         return fpkm

#     @staticmethod
#     def get_gene_lengths_from_gtf(gtf_path: str) -> pd.Series:
#         """
#         Extract gene lengths from GTF file.
#         Uses the sum of exon lengths (non-overlapping) for each gene.
        
#         Args:
#             gtf_path: Path to GTF file
            
#         Returns:
#             Series with gene_id as index and length in bp
#         """
#         import gzip
#         from collections import defaultdict
        
#         print(f"Extracting gene lengths from GTF...")
        
#         gene_exons = defaultdict(list)  # gene_id -> list of (start, end) tuples
        
#         opener = gzip.open if gtf_path.endswith('.gz') else open
        
#         with opener(gtf_path, 'rt') as f:
#             for line in f:
#                 if line.startswith('#'):
#                     continue
                
#                 fields = line.strip().split('\t')
#                 if len(fields) < 9:
#                     continue
                
#                 feature_type = fields[2]
#                 if feature_type != 'exon':
#                     continue
                
#                 start = int(fields[3])
#                 end = int(fields[4])
                
#                 # Parse attributes
#                 attrs = fields[8]
#                 gene_id = None
#                 for attr in attrs.split(';'):
#                     attr = attr.strip()
#                     if attr.startswith('gene_id'):
#                         gene_id = attr.split('"')[1].split('.')[0]  # Remove version
#                         break
                
#                 if gene_id:
#                     gene_exons[gene_id].append((start, end))
        
#         # Calculate merged exon length for each gene
#         gene_lengths = {}
#         for gene_id, exons in gene_exons.items():
#             # Merge overlapping exons
#             exons_sorted = sorted(exons)
#             merged = []
#             for start, end in exons_sorted:
#                 if merged and start <= merged[-1][1]:
#                     merged[-1] = (merged[-1][0], max(merged[-1][1], end))
#                 else:
#                     merged.append((start, end))
            
#             # Sum lengths
#             total_length = sum(end - start + 1 for start, end in merged)
#             gene_lengths[gene_id] = total_length
        
#         lengths = pd.Series(gene_lengths, name='length')
#         print(f"Extracted lengths for {len(lengths)} genes")
#         print(f"Length range: {lengths.min()} - {lengths.max()} bp")
        
#         return lengths

# ###############################################################################################################
# ############################### RNA-SEQ DATA ANALYSIS METHODS ARE HERE ########################################
# ###############################################################################################################  
#     def run_deseq(
#         self, 
#         group1: str, 
#         group2: str, 
#         group_col: str = 'Stage',
#         padj_threshold: float = 0.05,
#         log2fc_threshold: float = 1.0
#     ) -> pd.DataFrame:
#         """
#         Run differential expression analysis using DESeq2.
        
#         Args:
#             group1: Reference group (e.g., 'E14')
#             group2: Comparison group (e.g., 'E18')
#             group_col: Column in metadata to use for grouping
#             padj_threshold: Adjusted p-value cutoff
#             log2fc_threshold: Log2 fold change cutoff
            
#         Returns:
#             DataFrame with DESeq2 results, annotated with TF/Gene status
#         """
#         print(f"\n{'='*60}")
#         print(f"Running DESeq2: {group1} vs {group2}")
#         print(f"{'='*60}")
        
#         # Validate groups
#         if group_col not in self.metadata.columns:
#             raise ValueError(f"Column '{group_col}' not in metadata. Available: {list(self.metadata.columns)}")
        
#         unique_groups = self.metadata[group_col].unique()
#         if group1 not in unique_groups or group2 not in unique_groups:
#             raise ValueError(f"Groups must be in {unique_groups}")
        
#         # Filter samples for the two groups
#         samples_mask = self.metadata[group_col].isin([group1, group2])
#         samples = self.metadata[samples_mask].index.tolist()
        
#         # Subset counts and metadata
#         counts_subset = self.counts[samples].T  # DESeq2 wants samples as rows
#         metadata_subset = self.metadata.loc[samples, [group_col]].copy()
        
#         print(f"\nSamples: {len(samples)} ({group1}: {sum(metadata_subset[group_col]==group1)}, {group2}: {sum(metadata_subset[group_col]==group2)})")
        
#         # Ensure counts are integers
#         counts_subset = counts_subset.astype(int)
        
#         # Create DESeq dataset
#         print("\nCreating DESeq2 dataset...")
#         dds = DeseqDataSet(
#             counts=counts_subset,
#             metadata=metadata_subset,
#             design_factors=group_col,
#             n_cpus=1
            
#         )
    

#         # # VST-normalized counts (recommended for heatmaps/visualization)
#         # dds.vst_fit()
#         # dds.vst_transform()
#         # # See what's available
#         # print(dds.layers.keys()) 
#         # self.vst_counts = dds.layers['vst']  # Returns DataFrame (genes x samples)

#         # # Or as a DataFrame explicitly
#         # self.vst_df = pd.DataFrame(
#         #     dds.layers['vst'],
#         #     index=dds.var_names,
#         #     columns=dds.obs_names
#         # )

#         # Run DESeq2
#         print("Running DESeq2...")
#         dds.deseq2()
        
#         # Get results
#         print(f"Extracting results ({group1} vs {group2})...")
#         stat_res = DeseqStats(dds, contrast=[group_col, group1, group2], n_cpus=1)
#         stat_res.summary()

#         # In run_deseq, replace the symbol/is_TF assignment section with:

#         self.normed_counts = dds.layers['normed_counts']
#         # Get results as DataFrame
#         results = stat_res.results_df.copy()
#         results['gene_id'] = results.index

#         # Map Ensembl ID to gene symbol using GTF mapping
#         results['symbol'] = results['gene_id'].map(self.ensembl_to_symbol)
#         results['symbol'] = results['symbol'].fillna(results['gene_id'])  # Keep ID if no mapping

#         # Annotate as TF (check both Ensembl ID and symbol)
#         results['is_TF'] = (
#             results['gene_id'].isin(self.tf_ensembl) | 
#             results['symbol'].isin(self.tf_symbols)
#         )
#         results['gene_type'] = results['is_TF'].map({True: 'TF', False: 'Gene'})
#         # Add gene symbols (extract from gene_id if needed)
#         # For now, gene_id is the symbol since counts already use symbols

        
#         # Add direction
#         results['direction'] = np.where(
#             results['log2FoldChange'] > 0, 
#             group2,  # Positive = higher in group2
#             group1   # Negative = higher in group1
#         )
        
#         # Filter significant
#         sig_mask = (results['padj'] < padj_threshold) & (results['log2FoldChange'].abs() > log2fc_threshold)
#         results['significant'] = sig_mask
        
#         # Sort by adjusted p-value
#         results = results.sort_values('padj')
        
#         # Store results
#         self.deseq_results = results
#         self.deseq_params = {
#             'group1': group1,
#             'group2': group2,
#             'group_col': group_col,
#             'padj_threshold': padj_threshold,
#             'log2fc_threshold': log2fc_threshold
#         }
        
#         # Summary
#         n_sig = sig_mask.sum()
#         n_sig_tf = results[sig_mask & results['is_TF']].shape[0]
#         n_sig_gene = results[sig_mask & ~results['is_TF']].shape[0]
        
#         n_up = results[sig_mask & (results['log2FoldChange'] > 0)].shape[0]
#         n_down = results[sig_mask & (results['log2FoldChange'] < 0)].shape[0]
        
#         print(f"\n{'='*60}")
#         print("DESeq2 Results Summary")
#         print(f"{'='*60}")
#         print(f"Total genes tested: {len(results)}")
#         print(f"Significant (padj < {padj_threshold}, |log2FC| > {log2fc_threshold}): {n_sig}")
#         print(f"  - TFs: {n_sig_tf}")
#         print(f"  - Genes: {n_sig_gene}")
#         print(f"  - Up in {group2}: {n_up}")
#         print(f"  - Up in {group1}: {n_down}")
        
#         return results


 

#     def get_deseq_markers(
#             self, 
#             gene_type: str = 'both'
#         ) -> Dict[str, pd.DataFrame]:
#             """
#             Get differential markers categorized by direction.
            
#             Args:
#                 gene_type: 'TF', 'Gene', or 'both'
                
#             Returns:
#                 Dict with group1_high, group2_high, all_significant
#             """
#             if self.deseq_results is None:
#                 raise ValueError("Run run_deseq() first")
            
#             results = self.deseq_results.copy()
#             sig_results = results[results['significant']].copy()
            
#             if gene_type == 'TF':
#                 sig_results = sig_results[sig_results['is_TF']]
#             elif gene_type == 'Gene':
#                 sig_results = sig_results[~sig_results['is_TF']]
            
#             group1 = self.deseq_params['group1']
#             group2 = self.deseq_params['group2']
            
#             # Negative log2FC = higher in group1 (reference)
#             # Positive log2FC = higher in group2
#             group1_high = sig_results[sig_results['log2FoldChange'] < 0].sort_values('log2FoldChange')
#             group2_high = sig_results[sig_results['log2FoldChange'] > 0].sort_values('log2FoldChange', ascending=False)
            
#             markers = {
#                 f'{group1}_high': group1_high,
#                 f'{group2}_high': group2_high,
#                 'all_significant': sig_results
#             }
            
#             print(f"\n=== DEG Markers ({gene_type}) ===")
#             print(f"{group1}-high: {len(group1_high)}")
#             print(f"{group2}-high: {len(group2_high)}")
            
#             return markers
    
#     def compare_conditions(
#         self,
#         results1: pd.DataFrame,
#         results2: pd.DataFrame,
#         name1: str = "Condition1",
#         name2: str = "Condition2",
#         padj_threshold: float = 0.05,
#         log2fc_threshold: float = 1.0,
#         nonsig_pval: float = 0.5
#     ) -> Dict[str, pd.DataFrame]:
#         """
#         Compare DEGs across two conditions to find shared, contrasting, and specific genes.
        
#         Based on:
#         1. Shared DEGs: Significant in both with concordant direction
#         2. Contrasting DEGs: Significant in both with discordant direction  
#         3. Condition-specific DEGs: Significant in one, not approaching significance in other
        
#         Args:
#             results1: DESeq results from condition 1
#             results2: DESeq results from condition 2
#             name1: Name for condition 1
#             name2: Name for condition 2
#             padj_threshold: FDR threshold for significance
#             log2fc_threshold: |log2FC| threshold for significance
#             nonsig_pval: Nominal p-value threshold for "not approaching significance"
            
#         Returns:
#             Dict with 'shared', 'contrasting', 'specific_cond1', 'specific_cond2'
#         """
#         # Merge results on gene
#         merged = results1[['symbol', 'log2FoldChange', 'padj', 'pvalue']].merge(
#             results2[['symbol', 'log2FoldChange', 'padj', 'pvalue']],
#             on='symbol',
#             suffixes=(f'_{name1}', f'_{name2}')
#         )
        
#         # Define significance in each condition
#         merged[f'sig_{name1}'] = (
#             (merged[f'padj_{name1}'] < padj_threshold) & 
#             (merged[f'log2FoldChange_{name1}'].abs() > log2fc_threshold)
#         )
#         merged[f'sig_{name2}'] = (
#             (merged[f'padj_{name2}'] < padj_threshold) & 
#             (merged[f'log2FoldChange_{name2}'].abs() > log2fc_threshold)
#         )
        
#         # Direction concordance
#         merged['same_direction'] = (
#             np.sign(merged[f'log2FoldChange_{name1}']) == 
#             np.sign(merged[f'log2FoldChange_{name2}'])
#         )
        
#         # Not approaching significance (nominal p > nonsig_pval)
#         merged[f'nonsig_{name1}'] = merged[f'pvalue_{name1}'] > nonsig_pval
#         merged[f'nonsig_{name2}'] = merged[f'pvalue_{name2}'] > nonsig_pval
        
#         # Categorize
#         # 1. Shared: Significant in both, same direction
#         shared = merged[
#             merged[f'sig_{name1}'] & 
#             merged[f'sig_{name2}'] & 
#             merged['same_direction']
#         ].copy()
        
#         # 2. Contrasting: Significant in both, opposite direction
#         contrasting = merged[
#             merged[f'sig_{name1}'] & 
#             merged[f'sig_{name2}'] & 
#             ~merged['same_direction']
#         ].copy()
        
#         # 3. Condition1-specific: Significant in cond1, not approaching in cond2
#         specific_cond1 = merged[
#             merged[f'sig_{name1}'] & 
#             merged[f'nonsig_{name2}']
#         ].copy()
        
#         # 4. Condition2-specific: Significant in cond2, not approaching in cond1
#         specific_cond2 = merged[
#             merged[f'sig_{name2}'] & 
#             merged[f'nonsig_{name1}']
#         ].copy()
        
#         # Summary
#         print(f"\n{'='*60}")
#         print(f"DEG Comparison: {name1} vs {name2}")
#         print(f"{'='*60}")
#         print(f"Shared (concordant):     {len(shared)}")
#         print(f"Contrasting (discordant): {len(contrasting)}")
#         print(f"{name1}-specific:         {len(specific_cond1)}")
#         print(f"{name2}-specific:         {len(specific_cond2)}")
        
#         return {
#             'shared': shared,
#             'contrasting': contrasting,
#             f'specific_{name1}': specific_cond1,
#             f'specific_{name2}': specific_cond2,
#             'merged': merged
#         }
    


#     def compare_degs(
#         self,
#         results1: pd.DataFrame,
#         results2: pd.DataFrame,
#         name1: str = "DEG1",
#         name2: str = "DEG2",
#         padj_threshold: float = 0.05,
#         log2fc_threshold: float = 1.0,
#         nonsig_pval: float = 0.5
#     ) -> Dict[str, pd.DataFrame]:
#         """
#         Compare two DEG results to find shared, contrasting, and specific genes.
        
#         1. Shared: Significant in both, same direction
#         2. Contrasting: Significant in both, opposite direction
#         3. Specific: Significant in one, not significant in other
        
#         Args:
#             results1: First DESeq results DataFrame
#             results2: Second DESeq results DataFrame
#             name1: Label for first analysis
#             name2: Label for second analysis
            
#         Returns:
#             Dict with 'shared', 'contrasting', 'specific_<name1>', 'specific_<name2>'
#         """
#         # Merge on symbol
#         merged = results1[['symbol', 'log2FoldChange', 'padj', 'pvalue']].merge(
#             results2[['symbol', 'log2FoldChange', 'padj', 'pvalue']],
#             on='symbol',
#             suffixes=(f'_{name1}', f'_{name2}')
#         )
        
#         # Significance flags
#         sig1 = (merged[f'padj_{name1}'] < padj_threshold) & (merged[f'log2FoldChange_{name1}'].abs() > log2fc_threshold)
#         sig2 = (merged[f'padj_{name2}'] < padj_threshold) & (merged[f'log2FoldChange_{name2}'].abs() > log2fc_threshold)
        
#         # Direction
#         same_dir = np.sign(merged[f'log2FoldChange_{name1}']) == np.sign(merged[f'log2FoldChange_{name2}'])
        
#         # Not significant (nominal p > nonsig_pval)
#         nonsig1 = merged[f'pvalue_{name1}'] > nonsig_pval
#         nonsig2 = merged[f'pvalue_{name2}'] > nonsig_pval
        
#         # Categorize
#         shared = merged[sig1 & sig2 & same_dir].copy()
#         contrasting = merged[sig1 & sig2 & ~same_dir].copy()
#         specific1 = merged[sig1 & nonsig2].copy()
#         specific2 = merged[sig2 & nonsig1].copy()
        
#         print(f"\n=== DEG Comparison: {name1} vs {name2} ===")
#         print(f"Shared (same direction):     {len(shared)}")
#         print(f"Contrasting (opposite dir):  {len(contrasting)}")
#         print(f"Specific to {name1}:          {len(specific1)}")
#         print(f"Specific to {name2}:          {len(specific2)}")
        
#         return {
#             'shared': shared,
#             'contrasting': contrasting,
#             f'specific_{name1}': specific1,
#             f'specific_{name2}': specific2
#         }

# ###############################################################################################################
# ############################### OLD BULK GJSD METHODS ARE HERE ################################################
# ###############################################################################################################  

#     def run_gjsd(
#         self,
#         group1: str,
#         group2: str,
#         group_col: str = 'Stage',
#         method: str = 'bidirectional_gjsd',
#         padj_threshold: float = 0.05,
#         log2fc_threshold: float = 1.0,
#         gene_type: str = 'both'
#     ) -> Dict[str, pd.DataFrame]:
#         """
#         Run gJSD analysis on significant DEGs using BulkGJSD class.
        
#         Args:
#             group1: First group (e.g., 'E14') - will be "target" 
#             group2: Second group (e.g., 'E18') - will be "other"
#             group_col: Column in metadata for grouping
#             method: gJSD method - 'bidirectional_gjsd', 'asymmetric_gjsd', 'kl_directional',
#                     'gaussian_gjsd', 'geometric_jsd', etc.
#             padj_threshold: Filter to DEGs with padj < threshold
#             log2fc_threshold: Filter to DEGs with |log2FC| > threshold
#             gene_type: 'TF', 'Gene', or 'both'
            
#         Returns:
#             Dict with 'all', 'group1_specific', 'group2_specific' DataFrames
#         """
#         print(f"\n{'='*60}")
#         print(f"Running gJSD: {group1} vs {group2} (method={method})")
#         print(f"{'='*60}")
        
#         # Check DESeq results exist
#         if self.deseq_results is None:
#             raise ValueError("Run run_deseq() first")
        
#         # Get significant genes from DESeq
#         sig_genes = self.deseq_results[
#             (self.deseq_results['padj'] < padj_threshold) &
#             (self.deseq_results['log2FoldChange'].abs() > log2fc_threshold)
#         ].copy()
        
#         # Filter by gene type
#         if gene_type == 'TF':
#             sig_genes = sig_genes[sig_genes['is_TF']]
#             print(f"Filtering to TFs only")
#         elif gene_type == 'Gene':
#             sig_genes = sig_genes[~sig_genes['is_TF']]
#             print(f"Filtering to non-TF genes only")
        
#         sig_gene_ids = set(sig_genes.index)
        
#         n_up = (sig_genes['log2FoldChange'] > 0).sum()
#         n_down = (sig_genes['log2FoldChange'] < 0).sum()
#         print(f"Significant genes: {len(sig_genes)} ({n_up} {group1}-high, {n_down} {group2}-high)")
        
#         if len(sig_genes) == 0:
#             print("No significant genes found!")
#             return None
        
#         # Get samples for each group
#         group1_samples = self.metadata[self.metadata[group_col] == group1].index.tolist()
#         group2_samples = self.metadata[self.metadata[group_col] == group2].index.tolist()
        
#         # Filter to samples in counts
#         group1_samples = [s for s in group1_samples if s in self.counts.columns]
#         group2_samples = [s for s in group2_samples if s in self.counts.columns]
        
#         print(f"Samples: {group1}={len(group1_samples)}, {group2}={len(group2_samples)}")
        
#         # Subset counts to significant genes
        
#         expr_sig = np.log2(self.counts.loc[self.counts.index.isin(sig_gene_ids)] + 1)
#         print(f"Expression matrix: {expr_sig.shape[0]} genes x {expr_sig.shape[1]} samples")
        
#         # Run BulkGJSD
#         gjsd = BulkGJSD(expr_sig, n_processes=4)
        
#         # Methods that support bidirectional (have 'direction' column)
#         bidirectional_methods = {'asymmetric_gjsd', 'bidirectional_gjsd', 'kl_directional'}
        
#         if method in bidirectional_methods:
#             # Use compare_bidirectional for methods with direction
#             results = gjsd.compare_bidirectional(group1_samples, group2_samples, method=method)
#         else:
#             # Use compare for symmetric methods, then manually split by log2FC
#             all_results = gjsd.compare(group1_samples, group2_samples, method=method)
            
#             # Add DESeq2 info to determine direction
#             deseq_info = self.deseq_results[['log2FoldChange', 'padj', 'baseMean', 'is_TF', 'symbol']]
#             all_results = all_results.join(deseq_info, how='left')
            
#             # Add direction based on log2FC (positive = group1-high, negative = group2-high)
#             all_results['direction'] = np.where(all_results['log2FoldChange'] > 0, 1, -1)
#             all_results['signed_specificity'] = all_results['gjsd_score'] * all_results['direction']
            
#             # Split by direction
#             group1_specific = all_results[all_results['log2FoldChange'] > 0].copy()
#             group2_specific = all_results[all_results['log2FoldChange'] < 0].copy()
            
#             # Sort appropriately
#             group1_specific = group1_specific.sort_values('gjsd_score', ascending=False)
#             group2_specific = group2_specific.sort_values('gjsd_score', ascending=False)
            
#             results = {
#                 'all': all_results,
#                 'group1_specific': group1_specific,
#                 'group2_specific': group2_specific
#             }
        
#         # Add DESeq2 info to results (for bidirectional methods)
#         if method in bidirectional_methods:
#             deseq_info = self.deseq_results[['log2FoldChange', 'padj', 'baseMean', 'is_TF', 'symbol']]
#             for key in ['all', 'group1_specific', 'group2_specific']:
#                 results[key] = results[key].join(deseq_info, how='left')
        
#         # Sort by signed_specificity
#         results['group1_specific'] = results['group1_specific'].sort_values(
#             'signed_specificity' if 'signed_specificity' in results['group1_specific'].columns else 'gjsd_score', 
#             ascending=False
#         )
#         results['group2_specific'] = results['group2_specific'].sort_values(
#             'signed_specificity' if 'signed_specificity' in results['group2_specific'].columns else 'gjsd_score', 
#             ascending=False
#         )
        
#         # Store results
#         self.gjsd_results = results
#         self.gjsd_params = {
#             'group1': group1, 
#             'group2': group2, 
#             'group_col': group_col,
#             'method': method,
#             'gene_type': gene_type
#         }
        
#         # Summary
#         print(f"\n=== gJSD Results ===")
#         print(f"Total scored: {len(results['all'])}")
#         print(f"{group1}-specific: {len(results['group1_specific'])}")
#         print(f"{group2}-specific: {len(results['group2_specific'])}")
        
#         # Top specific for each group
#         print(f"\nTop 10 {group1}-specific:")
#         top_g1 = results['group1_specific'].head(10)
#         cols = ['symbol', 'gjsd_score', 'log2FoldChange', 'is_TF']
#         if 'signed_specificity' in top_g1.columns:
#             cols.insert(2, 'signed_specificity')
#         print(top_g1[[c for c in cols if c in top_g1.columns]].to_string())
        
#         print(f"\nTop 10 {group2}-specific:")
#         top_g2 = results['group2_specific'].head(10)
#         print(top_g2[[c for c in cols if c in top_g2.columns]].to_string())
        
#         return results


#     def get_gjsd_markers(
#         self,
#         min_specificity: float = 0.0,
#         gene_type: str = 'both'
#     ) -> Dict[str, pd.DataFrame]:
#         """
#         Get gJSD-filtered markers.
        
#         Args:
#             min_specificity: Minimum |signed_specificity| threshold
#             gene_type: 'TF', 'Gene', or 'both'
            
#         Returns:
#             Dict with group1_specific, group2_specific DataFrames
#         """
#         if self.gjsd_results is None:
#             raise ValueError("Run run_gjsd() first")
        
#         group1 = self.gjsd_params['group1']
#         group2 = self.gjsd_params['group2']
        
#         g1_df = self.gjsd_results['group1_specific'].copy()
#         g2_df = self.gjsd_results['group2_specific'].copy()
        
#         # Filter by specificity threshold
#         if min_specificity > 0:
#             g1_df = g1_df[g1_df['signed_specificity'].abs() >= min_specificity]
#             g2_df = g2_df[g2_df['signed_specificity'].abs() >= min_specificity]
        
#         # Filter by gene type
#         if gene_type == 'TF':
#             g1_df = g1_df[g1_df['is_TF']]
#             g2_df = g2_df[g2_df['is_TF']]
#         elif gene_type == 'Gene':
#             g1_df = g1_df[~g1_df['is_TF']]
#             g2_df = g2_df[~g2_df['is_TF']]
        
#         print(f"\n=== gJSD Markers (|specificity| >= {min_specificity}, type={gene_type}) ===")
#         print(f"{group1}-specific: {len(g1_df)}")
#         print(f"{group2}-specific: {len(g2_df)}")
        
#         return {
#             f'{group1}_specific': g1_df,
#             f'{group2}_specific': g2_df
#         }

#     # def get_candidates_with_binding(
#     #         self, 
#     #         gjsd_results: Dict[str, pd.DataFrame],
#     #         overlap_df: pd.DataFrame,
#     #         ensembl_to_symbol: Dict[str, str],
#     #         padj_thresh: float = 0.05,
#     #         lfc_thresh: float = 1.0,
#     #         gjsd_percentile: float = 95,
#     #         group1: str = 'E14',
#     #         group2: str = 'E18'
#     #     ) -> Dict[str, pd.DataFrame]:
#     #         """
#     #         Step 2: Get candidate TFs and genes, then filter by binding evidence.
#     #         """
#     #         print(f"\n{'='*60}")
#     #         print("Step 2: Filter candidates by binding evidence")
#     #         print(f"{'='*60}")
            
#     #         all_results = gjsd_results['all'].copy()
            
#     #         # Add symbol if not present
#     #         if 'symbol' not in all_results.columns:
#     #             all_results['symbol'] = [ensembl_to_symbol.get(g, g) for g in all_results.index]
            
#     #         # Calculate gJSD thresholds - handle empty cases
#     #         tf_scores = all_results[all_results['is_TF']]['gjsd_score'].dropna()
#     #         gene_scores = all_results[~all_results['is_TF']]['gjsd_score'].dropna()
            
#     #         gjsd_thresh_tf = np.percentile(tf_scores, gjsd_percentile) if len(tf_scores) > 0 else 0
#     #         gjsd_thresh_gene = np.percentile(gene_scores, gjsd_percentile) if len(gene_scores) > 0 else 0
            
#     #         print(f"\nThresholds:")
#     #         print(f"  padj < {padj_thresh}")
#     #         print(f"  |log2FC| > {lfc_thresh}")
#     #         print(f"  gJSD (TF) > {gjsd_thresh_tf:.4f} (p{gjsd_percentile}) [{len(tf_scores)} TFs]")
#     #         print(f"  gJSD (Gene) > {gjsd_thresh_gene:.4f} (p{gjsd_percentile}) [{len(gene_scores)} genes]")
            
#     #         # =========================================================================
#     #         # STEP 2a: Get candidate TFs and genes by expression
#     #         # =========================================================================
            
#     #         # Positive log2FC = group1-high (E14)
#     #         # Negative log2FC = group2-high (E18)
            
#     #         g1_tfs = all_results[
#     #             (all_results['padj'] < padj_thresh) & 
#     #             (all_results['log2FoldChange'] > lfc_thresh) & 
#     #             (all_results['is_TF'] == True) &
#     #             (all_results['gjsd_score'] >= gjsd_thresh_tf)
#     #         ].copy()
            
#     #         g2_tfs = all_results[
#     #             (all_results['padj'] < padj_thresh) & 
#     #             (all_results['log2FoldChange'] < -lfc_thresh) & 
#     #             (all_results['is_TF'] == True) &
#     #             (all_results['gjsd_score'] >= gjsd_thresh_tf)
#     #         ].copy()
            
#     #         # Genes (handle empty case)
#     #         if len(gene_scores) > 0:
#     #             g1_genes = all_results[
#     #                 (all_results['padj'] < padj_thresh) & 
#     #                 (all_results['log2FoldChange'] > lfc_thresh) & 
#     #                 (all_results['is_TF'] == False) &
#     #                 (all_results['gjsd_score'] >= gjsd_thresh_gene)
#     #             ].copy()
                
#     #             g2_genes = all_results[
#     #                 (all_results['padj'] < padj_thresh) & 
#     #                 (all_results['log2FoldChange'] < -lfc_thresh) & 
#     #                 (all_results['is_TF'] == False) &
#     #                 (all_results['gjsd_score'] >= gjsd_thresh_gene)
#     #             ].copy()
#     #         else:
#     #             g1_genes = pd.DataFrame()
#     #             g2_genes = pd.DataFrame()
#     #             print("\nWarning: No non-TF genes in gJSD results. Run with gene_type='both'")
            
#     #         print(f"\n=== Candidates by expression + gJSD ===")
#     #         print(f"{group1}-high TFs: {len(g1_tfs)}")
#     #         print(f"{group2}-high TFs: {len(g2_tfs)}")
#     #         print(f"{group1}-high genes: {len(g1_genes)}")
#     #         print(f"{group2}-high genes: {len(g2_genes)}")
            
#     #         # =========================================================================
#     #         # STEP 2b: Get unique TFs and genes from overlap_df
#     #         # =========================================================================
            
#     #         tfs_in_overlap = set(overlap_df['TF'].unique())
#     #         genes_in_overlap = set(overlap_df['gene'].unique())
            
#     #         g1_binding_tfs = set(overlap_df[overlap_df['condition'] == group1]['TF'].unique())
#     #         g2_binding_tfs = set(overlap_df[overlap_df['condition'] == group2]['TF'].unique())
            
#     #         g1_binding_genes = set(overlap_df[overlap_df['condition'] == group1]['gene'].unique())
#     #         g2_binding_genes = set(overlap_df[overlap_df['condition'] == group2]['gene'].unique())
            
#     #         print(f"\n=== Binding evidence in overlap_df ===")
#     #         print(f"Total TFs with ChIP evidence: {len(tfs_in_overlap)}")
#     #         print(f"Total target genes with ATAC evidence: {len(genes_in_overlap)}")
#     #         print(f"TFs with {group1} binding: {len(g1_binding_tfs)}")
#     #         print(f"TFs with {group2} binding: {len(g2_binding_tfs)}")
            
#     #         # =========================================================================
#     #         # STEP 2c: Filter candidates by binding evidence
#     #         # =========================================================================
            
#     #         g1_tfs_with_binding = g1_tfs[g1_tfs['symbol'].isin(tfs_in_overlap)].copy() if len(g1_tfs) > 0 else pd.DataFrame()
#     #         g2_tfs_with_binding = g2_tfs[g2_tfs['symbol'].isin(tfs_in_overlap)].copy() if len(g2_tfs) > 0 else pd.DataFrame()
            
#     #         g1_tfs_with_g1_binding = g1_tfs[g1_tfs['symbol'].isin(g1_binding_tfs)].copy() if len(g1_tfs) > 0 else pd.DataFrame()
#     #         g2_tfs_with_g2_binding = g2_tfs[g2_tfs['symbol'].isin(g2_binding_tfs)].copy() if len(g2_tfs) > 0 else pd.DataFrame()
            
#     #         g1_genes_with_binding = g1_genes[g1_genes['symbol'].isin(genes_in_overlap)].copy() if len(g1_genes) > 0 else pd.DataFrame()
#     #         g2_genes_with_binding = g2_genes[g2_genes['symbol'].isin(genes_in_overlap)].copy() if len(g2_genes) > 0 else pd.DataFrame()
            
#     #         g1_genes_with_g1_access = g1_genes[g1_genes['symbol'].isin(g1_binding_genes)].copy() if len(g1_genes) > 0 else pd.DataFrame()
#     #         g2_genes_with_g2_access = g2_genes[g2_genes['symbol'].isin(g2_binding_genes)].copy() if len(g2_genes) > 0 else pd.DataFrame()
            
#     #         print(f"\n=== Candidates with binding evidence ===")
#     #         print(f"{group1}-high TFs with ChIP evidence: {len(g1_tfs_with_binding)}")
#     #         print(f"{group1}-high TFs with {group1} binding: {len(g1_tfs_with_g1_binding)}")
#     #         print(f"{group2}-high TFs with ChIP evidence: {len(g2_tfs_with_binding)}")
#     #         print(f"{group2}-high TFs with {group2} binding: {len(g2_tfs_with_g2_binding)}")
#     #         print(f"{group1}-high genes with ATAC evidence: {len(g1_genes_with_binding)}")
#     #         print(f"{group2}-high genes with ATAC evidence: {len(g2_genes_with_binding)}")
            
#     #         # =========================================================================
#     #         # STEP 2d: Show top candidates
#     #         # =========================================================================
            
#     #         print(f"\n=== Top {group1}-high TFs (to UPREGULATE for neurogenesis) ===")
#     #         if len(g1_tfs_with_binding) > 0:
#     #             top_g1 = g1_tfs_with_binding.nlargest(10, 'gjsd_score')
#     #             print(top_g1[['symbol', 'log2FoldChange', 'padj', 'gjsd_score']].to_string())
#     #         else:
#     #             print("None found")
            
#     #         print(f"\n=== Top {group2}-high TFs (to DOWNREGULATE for neurogenesis) ===")
#     #         if len(g2_tfs_with_binding) > 0:
#     #             top_g2 = g2_tfs_with_binding.nlargest(10, 'gjsd_score')
#     #             print(top_g2[['symbol', 'log2FoldChange', 'padj', 'gjsd_score']].to_string())
#     #         else:
#     #             print("None found")
            
#     #         return {
#     #             f'{group1}_TFs': g1_tfs,
#     #             f'{group2}_TFs': g2_tfs,
#     #             f'{group1}_genes': g1_genes,
#     #             f'{group2}_genes': g2_genes,
#     #             f'{group1}_TFs_with_binding': g1_tfs_with_binding,
#     #             f'{group2}_TFs_with_binding': g2_tfs_with_binding,
#     #             f'{group1}_genes_with_binding': g1_genes_with_binding,
#     #             f'{group2}_genes_with_binding': g2_genes_with_binding,
#     #             f'{group1}_TFs_with_{group1}_binding': g1_tfs_with_g1_binding,
#     #             f'{group2}_TFs_with_{group2}_binding': g2_tfs_with_g2_binding,
#     #             f'{group1}_genes_with_{group1}_access': g1_genes_with_g1_access,
#     #             f'{group2}_genes_with_{group2}_access': g2_genes_with_g2_access,
#     #             'tfs_in_overlap': tfs_in_overlap,
#     #             'genes_in_overlap': genes_in_overlap,
#     #             'thresholds': {
#     #                 'padj': padj_thresh,
#     #                 'lfc': lfc_thresh,
#     #                 'gjsd_tf': gjsd_thresh_tf,
#     #                 'gjsd_gene': gjsd_thresh_gene
#     #             }
#     #         }
        
# ###############################################################################################################################################################
# ############################### OLD CONSENSUS PEAKS AND DA USING NARRO PEAKS ARE HERE (REDUNDANT)- NEED TO REMOVE #############################################
# ############################################################################################################################################################### 
#     # def _create_consensus_peaks(
#     #     self,
#     #     merge_distance: int = 10
#     # ) -> pd.DataFrame:
#     #     """
#     #     Create consensus peak set by merging overlapping peaks across samples.
#     #     Also maps genes from atac_annotated to each consensus peak.
#     #     """
#     #     peaks = self.atac_annotated[['chr', 'start', 'end']].drop_duplicates()
#     #     print(f"      Unique peaks before merging: {len(peaks)}")
        
#     #     peaks = peaks.sort_values(['chr', 'start', 'end'])
        
#     #     bed = pybedtools.BedTool.from_dataframe(peaks)
#     #     merged = bed.merge(d=merge_distance)
        
#     #     consensus = merged.to_dataframe(names=['chr', 'start', 'end'])
#     #     consensus['peak_id'] = [f"peak_{i}" for i in range(len(consensus))]
        
#     #     print(f"      Consensus peaks after merging: {len(consensus)}")
        
#     #     # Map genes from atac_annotated to consensus peaks
#     #     print(f"      Mapping genes to consensus peaks...")
        
#     #     atac_genes = self.atac_annotated[['chr', 'start', 'end', 'gene']].drop_duplicates()
        
#     #     consensus_bed = pybedtools.BedTool.from_dataframe(
#     #         consensus[['chr', 'start', 'end', 'peak_id']]
#     #     )
#     #     atac_genes_bed = pybedtools.BedTool.from_dataframe(atac_genes)
        
#     #     # Intersect to find overlapping genes
#     #     intersect = consensus_bed.intersect(atac_genes_bed, wa=True, wb=True)
        
#     #     # Build peak_id -> genes mapping
#     #     peak_genes = {}
#     #     for interval in intersect:
#     #         peak_id = interval[3]
#     #         gene = interval[7]
#     #         if peak_id not in peak_genes:
#     #             peak_genes[peak_id] = set()
#     #         peak_genes[peak_id].add(gene)
        
#     #     # Convert sets to lists
#     #     consensus['genes'] = consensus['peak_id'].map(
#     #         lambda x: list(peak_genes.get(x, set()))
#     #     )
        
#     #     n_with_genes = (consensus['genes'].apply(len) > 0).sum()
#     #     print(f"      Peaks with gene mappings: {n_with_genes} / {len(consensus)}")
        
#     #     return consensus

#     # def _build_atac_signal_matrix(
#     #     self,
#     #     consensus_peaks: pd.DataFrame
#     # ) -> pd.DataFrame:
#     #     """
#     #     Build signal matrix: consensus_peaks × samples.
#     #     """
#     #     samples = self.atac_annotated['sample'].unique()
#     #     print(f"      Building signal matrix for {len(samples)} samples...")
        
#     #     consensus_bed = pybedtools.BedTool.from_dataframe(
#     #         consensus_peaks[['chr', 'start', 'end', 'peak_id']]
#     #     )
        
#     #     signal_matrix = pd.DataFrame(index=consensus_peaks['peak_id'])
        
#     #     for sample in samples:
#     #         sample_peaks = self.atac_annotated[self.atac_annotated['sample'] == sample][
#     #             ['chr', 'start', 'end', 'signalValue']
#     #         ].drop_duplicates()
            
#     #         if len(sample_peaks) == 0:
#     #             signal_matrix[sample] = 0
#     #             continue
            
#     #         sample_bed = pybedtools.BedTool.from_dataframe(sample_peaks)
#     #         intersect = consensus_bed.intersect(sample_bed, wa=True, wb=True)
            
#     #         peak_signals = {}
#     #         for interval in intersect:
#     #             peak_id = interval[3]
#     #             signal = float(interval[7])
#     #             if peak_id not in peak_signals:
#     #                 peak_signals[peak_id] = signal
#     #             else:
#     #                 peak_signals[peak_id] = max(peak_signals[peak_id], signal)
            
#     #         signal_matrix[sample] = signal_matrix.index.map(peak_signals).fillna(0)
        
#     #     print(f"      Signal matrix shape: {signal_matrix.shape}")
#     #     print(f"      Non-zero entries: {(signal_matrix > 0).sum().sum()}")
        
#     #     return signal_matrix
#     # def _filter_atac_by_factor(
#     #     self,
#     #     factor: str = 'CD133'
#     # ) -> None:
#     #     """
#     #     Filter atac_annotated and overlap_df to samples with specific Factor.
        
#     #     Args:
#     #         factor: Factor to keep (e.g., 'CD133' for progenitors)
#     #     """
#     #     # Get sample IDs with this factor
#     #     factor_samples = self.merged_metadata[
#     #         self.merged_metadata['Factor'] == factor
#     #     ].index.tolist()
        
#     #     print(f"      {factor} samples: {factor_samples}")
        
#     #     # Filter atac_annotated
#     #     n_before = len(self.atac_annotated)
#     #     self.atac_annotated = self.atac_annotated[
#     #         self.atac_annotated['sample'].isin(factor_samples)
#     #     ]
#     #     n_after = len(self.atac_annotated)
#     #     print(f"      atac_annotated: {n_before} → {n_after} rows")
        
#     #     # Filter overlap_df
#     #     n_before = len(self.overlap_df)
#     #     self.overlap_df = self.overlap_df[
#     #         self.overlap_df['sample'].isin(factor_samples)
#     #     ]
#     #     n_after = len(self.overlap_df)
#     #     print(f"      overlap_df: {n_before} → {n_after} rows")
        
#     #     # Verify samples
#     #     print(f"      Unique samples in atac_annotated: {self.atac_annotated['sample'].nunique()}")

#     # def _filter_peaks_by_presence(
#     #     self,
#     #     signal_matrix: pd.DataFrame,
#     #     group_col: str = 'Stage',
#     #     min_samples_per_group: int = 3
#     # ) -> pd.DataFrame:
#     #     """
#     #     Filter peaks to those with signal in >= min_samples in EITHER group.
#     #     Uses UNION to preserve condition-specific peaks.
        
#     #     Args:
#     #         signal_matrix: Peak signal matrix (peaks x samples)
#     #         group_col: Column for grouping in merged_metadata
#     #         min_samples_per_group: Minimum samples with signal per group
            
#     #     Returns:
#     #         Filtered signal matrix
#     #     """
#     #     groups = self.merged_metadata[group_col].unique()
        
#     #     keep_peaks = set()
        
#     #     for group in groups:
#     #         group_samples = [
#     #             s for s in self.merged_metadata[self.merged_metadata[group_col] == group].index 
#     #             if s in signal_matrix.columns
#     #         ]
            
#     #         # Count non-zero samples per peak
#     #         nonzero_count = (signal_matrix[group_samples] > 0).sum(axis=1)
            
#     #         # Peaks with signal in >= min_samples in this group
#     #         group_peaks = set(signal_matrix.index[nonzero_count >= min_samples_per_group])
            
#     #         print(f"      {group}: {len(group_samples)} samples, {len(group_peaks)} peaks with signal in >= {min_samples_per_group}")
            
#     #         # UNION - keep peaks from either group
#     #         keep_peaks = keep_peaks.union(group_peaks)
        
#     #     filtered = signal_matrix.loc[list(keep_peaks)]
#     #     print(f"      Total peaks after filter (union): {len(filtered)}")
        
#     #     # Report zeros in filtered matrix
#     #     zeros_per_peak = (filtered == 0).sum(axis=1)
#     #     print(f"      Peaks with some missing data: {(zeros_per_peak > 0).sum()}")
#     #     print(f"      Peaks with complete data: {(zeros_per_peak == 0).sum()}")
        
#     #     return filtered

#     # def _run_dar_stats(
#     #     self,
#     #     signal_matrix: pd.DataFrame,
#     #     group1: str,
#     #     group2: str,
#     #     group_col: str = 'Stage',
#     #     min_present: int = 3,
#     #     method: str = 'ttest',
#     #     padj_thresh: float = 0.05,
#     #     lfc_thresh: float = 1.0
#     # ) -> pd.DataFrame:
#     #     """
#     #     Run differential accessibility analysis with proper handling of missing data.
        
#     #     Tests for both:
#     #     1. Signal differences (among samples where peak is present)
#     #     2. Presence differences (Fisher's exact test)
        
#     #     Args:
#     #         signal_matrix: Peak signal matrix (peaks x samples)
#     #         group1: First group name
#     #         group2: Second group name  
#     #         group_col: Column for grouping
#     #         min_present: Minimum samples with signal for signal test
#     #         method: Statistical method ('ttest' or 'wilcoxon')
#     #         padj_thresh: P-value threshold for significance
#     #         lfc_thresh: Log2FC threshold for significance
            
#     #     Returns:
#     #         DataFrame with DAR statistics
#     #     """
#     #     # Get samples per group
#     #     g1_samples = [
#     #         s for s in self.merged_metadata[self.merged_metadata[group_col] == group1].index 
#     #         if s in signal_matrix.columns
#     #     ]
#     #     g2_samples = [
#     #         s for s in self.merged_metadata[self.merged_metadata[group_col] == group2].index 
#     #         if s in signal_matrix.columns
#     #     ]
        
#     #     print(f"      Samples: {group1}={len(g1_samples)}, {group2}={len(g2_samples)}")
        
#     #     results = []
        
#     #     for peak_id in signal_matrix.index:
#     #         # Raw signal values
#     #         g1_raw = signal_matrix.loc[peak_id, g1_samples]
#     #         g2_raw = signal_matrix.loc[peak_id, g2_samples]
            
#     #         # Count presence (non-zero)
#     #         g1_present = (g1_raw > 0).sum()
#     #         g2_present = (g2_raw > 0).sum()
#     #         g1_absent = len(g1_samples) - g1_present
#     #         g2_absent = len(g2_samples) - g2_present
            
#     #         # Log transform non-zero values only
#     #         g1_vals = np.log2(g1_raw[g1_raw > 0] + 1)
#     #         g2_vals = np.log2(g2_raw[g2_raw > 0] + 1)
            
#     #         # Calculate mean signal (only from present samples)
#     #         mean_g1 = g1_vals.mean() if len(g1_vals) > 0 else 0
#     #         mean_g2 = g2_vals.mean() if len(g2_vals) > 0 else 0
#     #         log2fc = mean_g1 - mean_g2
            
#     #         # Signal test (only if enough non-zero in both groups)
#     #         if len(g1_vals) >= min_present and len(g2_vals) >= min_present:
#     #             if method == 'ttest':
#     #                 _, pval_signal = stats.ttest_ind(g1_vals, g2_vals)
#     #             elif method == 'wilcoxon':
#     #                 try:
#     #                     _, pval_signal = stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
#     #                 except:
#     #                     pval_signal = 1.0
#     #             else:
#     #                 raise ValueError(f"Unknown method: {method}")
#     #         else:
#     #             pval_signal = 1.0
            
#     #         # Presence test (Fisher's exact)
#     #         try:
#     #             _, pval_presence = stats.fisher_exact([
#     #                 [g1_present, g1_absent],
#     #                 [g2_present, g2_absent]
#     #             ])
#     #         except:
#     #             pval_presence = 1.0
            
#     #         # Combined p-value (minimum of signal and presence tests)
#     #         pval_combined = min(pval_signal, pval_presence)
            
#     #         # Determine test type that drove significance
#     #         if pval_signal < pval_presence:
#     #             sig_test = 'signal'
#     #         elif pval_presence < pval_signal:
#     #             sig_test = 'presence'
#     #         else:
#     #             sig_test = 'both'
            
#     #         results.append({
#     #             'peak_id': peak_id,
#     #             'g1_present': g1_present,
#     #             'g2_present': g2_present,
#     #             'g1_n': len(g1_samples),
#     #             'g2_n': len(g2_samples),
#     #             'mean_log2_g1': mean_g1,
#     #             'mean_log2_g2': mean_g2,
#     #             'log2FoldChange': log2fc,
#     #             'pval_signal': pval_signal,
#     #             'pval_presence': pval_presence,
#     #             'pvalue': pval_combined,
#     #             'sig_test': sig_test
#     #         })
        
#     #     results_df = pd.DataFrame(results).set_index('peak_id')
        
#     #     # FDR correction
#     #     results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
        
#     #     # Significance
#     #     results_df['significant'] = (
#     #         (results_df['padj'] < padj_thresh) & 
#     #         (results_df['log2FoldChange'].abs() > lfc_thresh)
#     #     )
        
#     #     # Direction
#     #     results_df['direction'] = np.where(
#     #         results_df['log2FoldChange'] > 0, group1, group2
#     #     )
        
#     #     # Summary statistics
#     #     n_sig = results_df['significant'].sum()
#     #     n_g1 = ((results_df['significant']) & (results_df['direction'] == group1)).sum()
#     #     n_g2 = ((results_df['significant']) & (results_df['direction'] == group2)).sum()
        
#     #     print(f"\n      === DAR Results ===")
#     #     print(f"      Total peaks tested: {len(results_df)}")
#     #     print(f"      Significant DARs: {n_sig} (padj<{padj_thresh}, |log2FC|>{lfc_thresh})")
#     #     print(f"        {group1}-specific (more open): {n_g1}")
#     #     print(f"        {group2}-specific (more open): {n_g2}")
        
#     #     # Break down by test type
#     #     if n_sig > 0:
#     #         sig_dars = results_df[results_df['significant']]
#     #         print(f"\n      Significance driven by:")
#     #         print(f"        Signal differences: {(sig_dars['sig_test'] == 'signal').sum()}")
#     #         print(f"        Presence differences: {(sig_dars['sig_test'] == 'presence').sum()}")
#     #         print(f"        Both: {(sig_dars['sig_test'] == 'both').sum()}")
        
#     #     return results_df

#     # def _create_differential_overlap(
#     #     self,
#     #     dar_results: pd.DataFrame,
#     #     consensus_peaks: pd.DataFrame,
#     #     group1: str,
#     #     group2: str,
#     #     padj_thresh: float = 0.05,
#     #     lfc_thresh: float = 1.0
#     # ) -> pd.DataFrame:
#     #     """
#     #     Create overlap_df subset with only differentially accessible TF-gene bindings.
#     #     """
#     #     # Get significant DARs with coordinates
#     #     sig_dars = dar_results[dar_results['significant']].reset_index()
#     #     sig_dars = sig_dars.merge(consensus_peaks, on='peak_id', how='left')
        
#     #     g1_dars = sig_dars[sig_dars['log2FoldChange'] > lfc_thresh]
#     #     g2_dars = sig_dars[sig_dars['log2FoldChange'] < -lfc_thresh]
        
#     #     print(f"      {group1}-specific DARs: {len(g1_dars)}")
#     #     print(f"      {group2}-specific DARs: {len(g2_dars)}")
        
#     #     # Get unique ATAC coordinates from overlap_df
#     #     overlap_regions = self.overlap_df[['atac_chr', 'atac_start', 'atac_end']].drop_duplicates()
#     #     overlap_regions.columns = ['chr', 'start', 'end']
        
#     #     if len(overlap_regions) == 0:
#     #         print("      No overlap regions available")
#     #         return pd.DataFrame()
        
#     #     overlap_bed = pybedtools.BedTool.from_dataframe(overlap_regions)
        
#     #     def get_dar_overlapping_coords(dar_coords, condition):
#     #         if len(dar_coords) == 0:
#     #             return set()
            
#     #         dar_bed = pybedtools.BedTool.from_dataframe(
#     #             dar_coords[['chr', 'start', 'end']]
#     #         )
            
#     #         intersect = overlap_bed.intersect(dar_bed, wa=True, u=True)
            
#     #         overlapping = set()
#     #         for interval in intersect:
#     #             overlapping.add((interval[0], int(interval[1]), int(interval[2])))
            
#     #         return overlapping
        
#     #     g1_overlapping = get_dar_overlapping_coords(g1_dars, group1)
#     #     g2_overlapping = get_dar_overlapping_coords(g2_dars, group2)
        
#     #     print(f"\n      Overlap_df regions in {group1} DARs: {len(g1_overlapping)}")
#     #     print(f"      Overlap_df regions in {group2} DARs: {len(g2_overlapping)}")
        
#     #     # Subset overlap_df
#     #     self.overlap_df['atac_coords'] = list(zip(
#     #         self.overlap_df['atac_chr'],
#     #         self.overlap_df['atac_start'],
#     #         self.overlap_df['atac_end']
#     #     ))
        
#     #     g1_overlap = self.overlap_df[self.overlap_df['atac_coords'].isin(g1_overlapping)].copy()
#     #     g2_overlap = self.overlap_df[self.overlap_df['atac_coords'].isin(g2_overlapping)].copy()
        
#     #     g1_overlap['dar_condition'] = group1
#     #     g2_overlap['dar_condition'] = group2
        
#     #     diff_overlap = pd.concat([g1_overlap, g2_overlap], ignore_index=True)
        
#     #     # Clean up temp column
#     #     if 'atac_coords' in self.overlap_df.columns:
#     #         self.overlap_df = self.overlap_df.drop(columns=['atac_coords'])
#     #     if 'atac_coords' in diff_overlap.columns:
#     #         diff_overlap = diff_overlap.drop(columns=['atac_coords'])
        
#     #     print(f"\n      === Differential Overlap ===")
#     #     print(f"      Total rows: {len(diff_overlap)}")
#     #     print(f"      {group1}-specific TF-gene bindings: {len(g1_overlap)}")
#     #     print(f"      {group2}-specific TF-gene bindings: {len(g2_overlap)}")
#     #     if len(diff_overlap) > 0:
#     #         print(f"      Unique TFs: {diff_overlap['TF'].nunique()}")
#     #         print(f"      Unique genes: {diff_overlap['gene'].nunique()}")
        
#     #     return diff_overlap

#     # def run_differential_atac(
#     #     self,
#     #     group1: str = 'E14',
#     #     group2: str = 'E18',
#     #     group_col: str = 'Stage',
#     #     factor_filter: Optional[str] = 'CD133',
#     #     method: str = 'ttest',
#     #     merge_distance: int = 100,
#     #     min_samples_per_group: int = 3,
#     #     padj_thresh: float = 0.05,
#     #     lfc_thresh: float = 1.0
#     # ) -> Dict[str, pd.DataFrame]:
#     #     """
#     #     Run differential ATAC-seq analysis pipeline.
        
#     #     Args:
#     #         group1, group2: Groups to compare
#     #         group_col: Metadata column for grouping
#     #         factor_filter: If set, filter to this Factor (e.g., 'CD133')
#     #         method: 'ttest' or 'wilcoxon'
#     #         merge_distance: Distance to merge nearby peaks
#     #         min_samples_per_group: Min samples with signal per group
#     #         padj_thresh: Adjusted p-value threshold
#     #         lfc_thresh: Log2 fold change threshold
            
#     #     Returns:
#     #         Dict with dar_results, diff_overlap, consensus_peaks, signal_matrix
#     #     """
#     #     print(f"\n{'='*60}")
#     #     print(f"Differential ATAC: {group1} vs {group2}")
#     #     print(f"{'='*60}")
        
#     #     # 0. Filter samples if needed
#     #     if factor_filter:
#     #         print(f"\n[0/5] Filtering to {factor_filter} samples...")
#     #         self._filter_atac_by_factor(factor_filter)
        
#     #     # 1. Create consensus peaks
#     #     print("\n[1/5] Creating consensus peaks...")
#     #     self.consensus_peaks = self._create_consensus_peaks(merge_distance=merge_distance)
        
#     #     # 2. Build signal matrix
#     #     print("\n[2/5] Building signal matrix...")
#     #     self.atac_signal_matrix = self._build_atac_signal_matrix(self.consensus_peaks)
        
#     #     # 3. Filter peaks by presence
#     #     print(f"\n[3/5] Filtering peaks by presence (min {min_samples_per_group} per group)...")
#     #     self.atac_signal_matrix_filtered = self._filter_peaks_by_presence(
#     #         self.atac_signal_matrix,
#     #         group_col=group_col,
#     #         min_samples_per_group=min_samples_per_group
#     #     )
        
#     #     # 4. Run differential analysis
#     #     print("\n[4/5] Running differential analysis...")
#     #     self.dar_results = self._run_dar_stats(
#     #         self.atac_signal_matrix_filtered,
#     #         group1=group1,
#     #         group2=group2,
#     #         group_col=group_col,
#     #         min_present=min_samples_per_group,
#     #         method=method,
#     #         padj_thresh=padj_thresh,
#     #         lfc_thresh=lfc_thresh
#     #     )
        
#     #     # 5. Create differential overlap with ChIP
#     #     print("\n[5/5] Creating differential overlap...")
#     #     self.diff_overlap_df = self._create_differential_overlap(
#     #         self.dar_results,
#     #         self.consensus_peaks,
#     #         group1=group1,
#     #         group2=group2,
#     #         padj_thresh=padj_thresh,
#     #         lfc_thresh=lfc_thresh
#     #     )
        
#     #     # Store parameters
#     #     self.dar_params = {
#     #         'group1': group1,
#     #         'group2': group2,
#     #         'group_col': group_col,
#     #         'factor_filter': factor_filter,
#     #         'min_samples_per_group': min_samples_per_group,
#     #         'method': method,
#     #         'padj_thresh': padj_thresh,
#     #         'lfc_thresh': lfc_thresh
#     #     }
        
#     #     print(f"\n{'='*60}")
#     #     print("Differential ATAC complete!")
#     #     print(f"{'='*60}")
#     #     # After step 5, merge genes into dar_results
#     #     self.dar_results = self.dar_results.merge(
#     #         self.consensus_peaks[['peak_id', 'genes']].set_index('peak_id'),
#     #         left_index=True,
#     #         right_index=True,
#     #         how='left'
#     #     )
        
#     #     print(f"\n{'='*60}")
#     #     print("Differential ATAC complete!")
#     #     print(f"{'='*60}")
        
#     #     return {
#     #         'dar_results': self.dar_results,
#     #         'diff_overlap': self.diff_overlap_df,
#     #         'consensus_peaks': self.consensus_peaks,
#     #         'signal_matrix': self.atac_signal_matrix,
#     #         'signal_matrix_filtered': self.atac_signal_matrix_filtered
#     #     }
    
#     # def quantify_atac_at_enhancers(self, enhancer_df, consensus_peaks, atac_metadata, 
#     #                                 group_col='Stage', condition1='E14', condition2='E18'):
#     #     """
#     #     Quantify ATAC-seq signal at enhancers for two conditions
        
#     #     Parameters:
#     #     -----------
#     #     enhancer_df : pd.DataFrame
#     #         Enhancer database from load_enhancer_database()
#     #     consensus_peaks : pd.DataFrame
#     #         Consensus peaks with columns: chr, start, end, and condition-specific columns
#     #     atac_metadata : pd.DataFrame
#     #         ATAC metadata with condition labels
        
#     #     Returns:
#     #     --------
#     #     tuple of (enhancer_atac_e14, enhancer_atac_e18, enhancer_df_with_atac)
#     #         - Series with ATAC signal per enhancer for each condition
#     #         - enhancer_df with added atac_signal_e14 and atac_signal_e18 columns
#     #     """
        
#     #     print(f"\nQuantifying ATAC-seq signal at enhancers...")
        
#     #     # Get unique enhancers
#     #     unique_enhancers = enhancer_df[['enhancer_id', 'enhancer_chr', 
#     #                                     'enhancer_start', 'enhancer_end']].drop_duplicates()
        
#     #     print(f"Unique enhancers: {len(unique_enhancers)}")
        

        
#     #     enhancer_bed = pybedtools.BedTool.from_dataframe(
#     #         unique_enhancers.rename(columns={
#     #             'enhancer_chr': 'chr',
#     #             'enhancer_start': 'start', 
#     #             'enhancer_end': 'end'
#     #         })[['chr', 'start', 'end', 'enhancer_id']]
#     #     )
        
#     #     consensus_bed = pybedtools.BedTool.from_dataframe(
#     #         consensus_peaks[['chr', 'start', 'end']]
#     #     )
        
#     #     # Find overlaps
#     #     overlaps = enhancer_bed.intersect(consensus_bed, wa=True, wb=True)
        
#     #     # Parse overlaps and extract sample-specific signals
#     #     overlap_data = []
#     #     for interval in overlaps:
#     #         enh_chr = interval.fields[0]
#     #         enh_start = int(interval.fields[1])
#     #         enh_end = int(interval.fields[2])
#     #         enh_id = interval.fields[3]
            
#     #         peak_chr = interval.fields[4]
#     #         peak_start = int(interval.fields[5])
#     #         peak_end = int(interval.fields[6])
            
#     #         # Find this peak in consensus_peaks to get signal values
#     #         peak_match = consensus_peaks[
#     #             (consensus_peaks['chr'] == peak_chr) &
#     #             (consensus_peaks['start'] == peak_start) &
#     #             (consensus_peaks['end'] == peak_end)
#     #         ]
            
#     #         if len(peak_match) > 0:
#     #             overlap_data.append({
#     #                 'enhancer_id': enh_id,
#     #                 'peak_idx': peak_match.index[0]
#     #             })
        
#     #     overlap_df = pd.DataFrame(overlap_data)
        
#     #     # Get sample columns from consensus peaks
#     #     sample_cols = [col for col in consensus_peaks.columns 
#     #                 if col not in ['chr', 'start', 'end', 'peak_id']]
        
#     #     # Separate E14 and E18 samples
#     #     e14_samples = atac_metadata[atac_metadata[group_col] == condition1].index.tolist()
#     #     e18_samples = atac_metadata[atac_metadata[group_col] == condition2].index.tolist()
        
#     #     # Filter to available sample columns
#     #     e14_cols = [col for col in sample_cols if any(s in col for s in e14_samples)]
#     #     e18_cols = [col for col in sample_cols if any(s in col for s in e18_samples)]
        
#     #     print(f"E14 samples: {len(e14_cols)}")
#     #     print(f"E18 samples: {len(e18_cols)}")
        
#     #     # Calculate mean ATAC signal per enhancer
#     #     enhancer_atac = {}
        
#     #     for enh_id in unique_enhancers['enhancer_id']:
#     #         # Get all peaks overlapping this enhancer
#     #         peak_indices = overlap_df[overlap_df['enhancer_id'] == enh_id]['peak_idx'].tolist()
            
#     #         if len(peak_indices) > 0:
#     #             # Mean signal across overlapping peaks, then across samples
#     #             e14_signal = consensus_peaks.loc[peak_indices, e14_cols].mean().mean()
#     #             e18_signal = consensus_peaks.loc[peak_indices, e18_cols].mean().mean()
#     #         else:
#     #             e14_signal = 0
#     #             e18_signal = 0
            
#     #         enhancer_atac[enh_id] = {
#     #             'atac_e14': e14_signal,
#     #             'atac_e18': e18_signal
#     #         }
        
#     #     # Convert to Series
#     #     enhancer_atac_e14 = pd.Series({k: v['atac_e14'] for k, v in enhancer_atac.items()})
#     #     enhancer_atac_e18 = pd.Series({k: v['atac_e18'] for k, v in enhancer_atac.items()})
        
#     #     # Add to enhancer_df
#     #     enhancer_df_with_atac = enhancer_df.copy()
#     #     enhancer_df_with_atac['atac_signal_e14'] = enhancer_df_with_atac['enhancer_id'].map(
#     #         enhancer_atac_e14
#     #     )
#     #     enhancer_df_with_atac['atac_signal_e18'] = enhancer_df_with_atac['enhancer_id'].map(
#     #         enhancer_atac_e18
#     #     )
        
#     #     # Fill NaN with 0
#     #     enhancer_df_with_atac['atac_signal_e14'].fillna(0, inplace=True)
#     #     enhancer_df_with_atac['atac_signal_e18'].fillna(0, inplace=True)
        
#     #     print(f"\nEnhancers with ATAC signal > 0 (E14): {(enhancer_atac_e14 > 0).sum()}")
#     #     print(f"Enhancers with ATAC signal > 0 (E18): {(enhancer_atac_e18 > 0).sum()}")
        
#     #     return enhancer_atac_e14, enhancer_atac_e18, enhancer_df_with_atac