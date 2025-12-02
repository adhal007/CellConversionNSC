# ==============================================================================
# NSCAnalysis Class - Step 1: Data Loading
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
# Add these imports at the top
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
# Add this import at the top of the class file
from src.bulk_gjsd import BulkGJSD


class NSCAnalysis:
    """
    Analysis pipeline for NSC cell conversion project.
    
    Questions addressed:
        1. What are pro-neural and pro-glial signatures?
        2. What factors can convert glial to neuronal?
    """
    
    def __init__(
        self,
        counts_path: str,
        metadata_path: str,
        overlap_df_path: str,
        chip_annotated_path: str,
        atac_annotated_path: str,
        tf_list_path: str,
        gtf_path: str
    ):
        # Store paths
        self.paths = {
            'counts': Path(counts_path),
            'metadata': Path(metadata_path),
            'overlap_df': Path(overlap_df_path),
            'chip_annotated': Path(chip_annotated_path),
            'atac_annotated': Path(atac_annotated_path),
            'tf_list': Path(tf_list_path),
            'gtf': Path(gtf_path)
        }
        
        # Validate paths exist
        for name, path in self.paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name}: {path}")
        
        # Data containers (populated by _load_data)
        self.counts = None
        self.metadata = None
        self.overlap_df = None
        self.chip_annotated = None
        self.atac_annotated = None
        self.tf_symbols = None
        self.gene_id_to_symbol = None
        
        # Results containers
        self.deseq_results = None
        self.gjsd_results = None
        self.diff_atac = None
        self.grn = None
        
        # Load data
        self._load_data()
    
    # Update _load_data method:

    def _load_data(self):
        """Load all input data."""
        print("=" * 60)
        print("Loading data...")
        print("=" * 60)
        
        # 1. Load counts (semicolon-separated, gene IDs in column 1)
        print("\n[1/6] Loading counts...")
        self.counts = pd.read_csv(self.paths['counts'], sep=';', index_col=1)
        self.counts = self.counts.drop(columns=['Unnamed: 0'])
        
        # Drop genes with NaN values
        nan_genes = self.counts.isna().any(axis=1).sum()
        if nan_genes > 0:
            print(f"      Dropping {nan_genes} genes with NaN values")
            self.counts = self.counts.dropna()
        
        print(f"      {self.counts.shape[0]} genes x {self.counts.shape[1]} samples")
        
        # 2. Load metadata (semicolon-separated)
        print("\n[2/6] Loading metadata...")
        self.metadata = pd.read_csv(self.paths['metadata'], sep=';')
        self.metadata = self.metadata.set_index('SampleID')
        print(f"      {self.metadata.shape[0]} samples")
        print(f"      Columns: {list(self.metadata.columns)}")
        
        # 3. Load overlap_df
        print("\n[3/6] Loading overlap_df (ChIP ∩ ATAC)...")
        self.overlap_df = pd.read_csv(self.paths['overlap_df'], sep='\t')
        print(f"      {self.overlap_df.shape[0]} overlaps")
        print(f"      Columns: {list(self.overlap_df.columns)}")
        
        # 4. Load chip_annotated
        print("\n[4/6] Loading chip_annotated...")
        self.chip_annotated = pd.read_csv(self.paths['chip_annotated'], sep='\t')
        print(f"      {self.chip_annotated.shape[0]} peaks")
        
        # 5. Load atac_annotated
        print("\n[5/6] Loading atac_annotated...")
        self.atac_annotated = pd.read_csv(self.paths['atac_annotated'], sep='\t')
        print(f"      {self.atac_annotated.shape[0]} peaks")
        
        # 6. Load TF list
        print("\n[6/6] Loading TF list...")
        tf_df = pd.read_excel(self.paths['tf_list'])
        tf_df = tf_df.dropna()
        tf_df.columns = tf_df.iloc[0, :]
        tf_df = tf_df.iloc[1:, :].reset_index(drop=True)
        self.tf_df = tf_df
        self.tf_symbols = set(tf_df['Gene Symbol'].str.strip().tolist())
        self.tf_ensembl = set(tf_df['Ensembl ID'].str.strip().tolist())
        print(f"      {len(self.tf_symbols)} TFs")
        
        # 7. Parse GTF for gene mappings
        print("\n[7/7] Parsing GTF for gene symbol mappings...")
        self.ensembl_to_symbol = self._parse_gtf_gene_mapping()
        
        # Check coverage
        mapped = sum(1 for g in self.counts.index if g in self.ensembl_to_symbol)
        print(f"      Counts genes with mapping: {mapped} / {len(self.counts)}")
        
        print("\n" + "=" * 60)
        print("Data loaded successfully!")
        print("=" * 60)
    
    def summary(self):
        """Print summary of loaded data."""
        print("\n=== NSCAnalysis Summary ===")
        print(f"Counts: {self.counts.shape[0]} genes x {self.counts.shape[1]} samples")
        print(f"Metadata: {self.metadata.shape[0]} samples")
        print(f"  - Stages: {self.metadata['Stage'].unique().tolist()}")
        print(f"  - Regions: {self.metadata['Region'].unique().tolist()}")
        print(f"  - Markers: {self.metadata['Marker'].unique().tolist()}")
        print(f"Overlap_df: {self.overlap_df.shape[0]} functional TF-gene bindings")
        print(f"  - Unique TFs: {self.overlap_df['TF'].nunique()}")
        print(f"  - Unique genes: {self.overlap_df['gene'].nunique()}")
        print(f"ChIP peaks: {self.chip_annotated.shape[0]}")
        print(f"ATAC peaks: {self.atac_annotated.shape[0]}")
        print(f"TF list: {len(self.tf_symbols)} TFs")

    def _parse_gtf_gene_mapping(self) -> Dict[str, str]:
        """Parse GTF file to create Ensembl ID -> Gene Symbol mapping."""
        print("      Parsing GTF for gene mappings...")
        
        ensembl_to_symbol = {}
        
        with open(self.paths['gtf'], 'r') as f:
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
# ==============================================================================
# NSCAnalysis Class - Step 2: Add DESeq2 method
# ==============================================================================
    def run_deseq(
        self, 
        group1: str, 
        group2: str, 
        group_col: str = 'Stage',
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Run differential expression analysis using DESeq2.
        
        Args:
            group1: Reference group (e.g., 'E14')
            group2: Comparison group (e.g., 'E18')
            group_col: Column in metadata to use for grouping
            padj_threshold: Adjusted p-value cutoff
            log2fc_threshold: Log2 fold change cutoff
            
        Returns:
            DataFrame with DESeq2 results, annotated with TF/Gene status
        """
        print(f"\n{'='*60}")
        print(f"Running DESeq2: {group1} vs {group2}")
        print(f"{'='*60}")
        
        # Validate groups
        if group_col not in self.metadata.columns:
            raise ValueError(f"Column '{group_col}' not in metadata. Available: {list(self.metadata.columns)}")
        
        unique_groups = self.metadata[group_col].unique()
        if group1 not in unique_groups or group2 not in unique_groups:
            raise ValueError(f"Groups must be in {unique_groups}")
        
        # Filter samples for the two groups
        samples_mask = self.metadata[group_col].isin([group1, group2])
        samples = self.metadata[samples_mask].index.tolist()
        
        # Subset counts and metadata
        counts_subset = self.counts[samples].T  # DESeq2 wants samples as rows
        metadata_subset = self.metadata.loc[samples, [group_col]].copy()
        
        print(f"\nSamples: {len(samples)} ({group1}: {sum(metadata_subset[group_col]==group1)}, {group2}: {sum(metadata_subset[group_col]==group2)})")
        
        # Ensure counts are integers
        counts_subset = counts_subset.astype(int)
        
        # Create DESeq dataset
        print("\nCreating DESeq2 dataset...")
        dds = DeseqDataSet(
            counts=counts_subset,
            metadata=metadata_subset,
            design_factors=group_col,
            n_cpus=1
            
        )
        
        # Run DESeq2
        print("Running DESeq2...")
        dds.deseq2()
        
        # Get results
        print(f"Extracting results ({group1} vs {group2})...")
        stat_res = DeseqStats(dds, contrast=[group_col, group1, group2], n_cpus=1)
        stat_res.summary()
        

        # In run_deseq, replace the symbol/is_TF assignment section with:

        # Get results as DataFrame
        results = stat_res.results_df.copy()
        results['gene_id'] = results.index

        # Map Ensembl ID to gene symbol using GTF mapping
        results['symbol'] = results['gene_id'].map(self.ensembl_to_symbol)
        results['symbol'] = results['symbol'].fillna(results['gene_id'])  # Keep ID if no mapping

        # Annotate as TF (check both Ensembl ID and symbol)
        results['is_TF'] = (
            results['gene_id'].isin(self.tf_ensembl) | 
            results['symbol'].isin(self.tf_symbols)
        )
        results['gene_type'] = results['is_TF'].map({True: 'TF', False: 'Gene'})
        # Add gene symbols (extract from gene_id if needed)
        # For now, gene_id is the symbol since counts already use symbols

        
        # Add direction
        results['direction'] = np.where(
            results['log2FoldChange'] > 0, 
            group2,  # Positive = higher in group2
            group1   # Negative = higher in group1
        )
        
        # Filter significant
        sig_mask = (results['padj'] < padj_threshold) & (results['log2FoldChange'].abs() > log2fc_threshold)
        results['significant'] = sig_mask
        
        # Sort by adjusted p-value
        results = results.sort_values('padj')
        
        # Store results
        self.deseq_results = results
        self.deseq_params = {
            'group1': group1,
            'group2': group2,
            'group_col': group_col,
            'padj_threshold': padj_threshold,
            'log2fc_threshold': log2fc_threshold
        }
        
        # Summary
        n_sig = sig_mask.sum()
        n_sig_tf = results[sig_mask & results['is_TF']].shape[0]
        n_sig_gene = results[sig_mask & ~results['is_TF']].shape[0]
        
        n_up = results[sig_mask & (results['log2FoldChange'] > 0)].shape[0]
        n_down = results[sig_mask & (results['log2FoldChange'] < 0)].shape[0]
        
        print(f"\n{'='*60}")
        print("DESeq2 Results Summary")
        print(f"{'='*60}")
        print(f"Total genes tested: {len(results)}")
        print(f"Significant (padj < {padj_threshold}, |log2FC| > {log2fc_threshold}): {n_sig}")
        print(f"  - TFs: {n_sig_tf}")
        print(f"  - Genes: {n_sig_gene}")
        print(f"  - Up in {group2}: {n_up}")
        print(f"  - Up in {group1}: {n_down}")
        
        return results


    
    def get_deseq_markers(
            self, 
            gene_type: str = 'both'
        ) -> Dict[str, pd.DataFrame]:
            """
            Get differential markers categorized by direction.
            
            Args:
                gene_type: 'TF', 'Gene', or 'both'
                
            Returns:
                Dict with group1_high, group2_high, all_significant
            """
            if self.deseq_results is None:
                raise ValueError("Run run_deseq() first")
            
            results = self.deseq_results.copy()
            sig_results = results[results['significant']].copy()
            
            if gene_type == 'TF':
                sig_results = sig_results[sig_results['is_TF']]
            elif gene_type == 'Gene':
                sig_results = sig_results[~sig_results['is_TF']]
            
            group1 = self.deseq_params['group1']
            group2 = self.deseq_params['group2']
            
            # Negative log2FC = higher in group1 (reference)
            # Positive log2FC = higher in group2
            group1_high = sig_results[sig_results['log2FoldChange'] < 0].sort_values('log2FoldChange')
            group2_high = sig_results[sig_results['log2FoldChange'] > 0].sort_values('log2FoldChange', ascending=False)
            
            markers = {
                f'{group1}_high': group1_high,
                f'{group2}_high': group2_high,
                'all_significant': sig_results
            }
            
            print(f"\n=== DEG Markers ({gene_type}) ===")
            print(f"{group1}-high: {len(group1_high)}")
            print(f"{group2}-high: {len(group2_high)}")
            
            return markers
    
    def compare_conditions(
        self,
        results1: pd.DataFrame,
        results2: pd.DataFrame,
        name1: str = "Condition1",
        name2: str = "Condition2",
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        nonsig_pval: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare DEGs across two conditions to find shared, contrasting, and specific genes.
        
        Based on:
        1. Shared DEGs: Significant in both with concordant direction
        2. Contrasting DEGs: Significant in both with discordant direction  
        3. Condition-specific DEGs: Significant in one, not approaching significance in other
        
        Args:
            results1: DESeq results from condition 1
            results2: DESeq results from condition 2
            name1: Name for condition 1
            name2: Name for condition 2
            padj_threshold: FDR threshold for significance
            log2fc_threshold: |log2FC| threshold for significance
            nonsig_pval: Nominal p-value threshold for "not approaching significance"
            
        Returns:
            Dict with 'shared', 'contrasting', 'specific_cond1', 'specific_cond2'
        """
        # Merge results on gene
        merged = results1[['symbol', 'log2FoldChange', 'padj', 'pvalue']].merge(
            results2[['symbol', 'log2FoldChange', 'padj', 'pvalue']],
            on='symbol',
            suffixes=(f'_{name1}', f'_{name2}')
        )
        
        # Define significance in each condition
        merged[f'sig_{name1}'] = (
            (merged[f'padj_{name1}'] < padj_threshold) & 
            (merged[f'log2FoldChange_{name1}'].abs() > log2fc_threshold)
        )
        merged[f'sig_{name2}'] = (
            (merged[f'padj_{name2}'] < padj_threshold) & 
            (merged[f'log2FoldChange_{name2}'].abs() > log2fc_threshold)
        )
        
        # Direction concordance
        merged['same_direction'] = (
            np.sign(merged[f'log2FoldChange_{name1}']) == 
            np.sign(merged[f'log2FoldChange_{name2}'])
        )
        
        # Not approaching significance (nominal p > nonsig_pval)
        merged[f'nonsig_{name1}'] = merged[f'pvalue_{name1}'] > nonsig_pval
        merged[f'nonsig_{name2}'] = merged[f'pvalue_{name2}'] > nonsig_pval
        
        # Categorize
        # 1. Shared: Significant in both, same direction
        shared = merged[
            merged[f'sig_{name1}'] & 
            merged[f'sig_{name2}'] & 
            merged['same_direction']
        ].copy()
        
        # 2. Contrasting: Significant in both, opposite direction
        contrasting = merged[
            merged[f'sig_{name1}'] & 
            merged[f'sig_{name2}'] & 
            ~merged['same_direction']
        ].copy()
        
        # 3. Condition1-specific: Significant in cond1, not approaching in cond2
        specific_cond1 = merged[
            merged[f'sig_{name1}'] & 
            merged[f'nonsig_{name2}']
        ].copy()
        
        # 4. Condition2-specific: Significant in cond2, not approaching in cond1
        specific_cond2 = merged[
            merged[f'sig_{name2}'] & 
            merged[f'nonsig_{name1}']
        ].copy()
        
        # Summary
        print(f"\n{'='*60}")
        print(f"DEG Comparison: {name1} vs {name2}")
        print(f"{'='*60}")
        print(f"Shared (concordant):     {len(shared)}")
        print(f"Contrasting (discordant): {len(contrasting)}")
        print(f"{name1}-specific:         {len(specific_cond1)}")
        print(f"{name2}-specific:         {len(specific_cond2)}")
        
        return {
            'shared': shared,
            'contrasting': contrasting,
            f'specific_{name1}': specific_cond1,
            f'specific_{name2}': specific_cond2,
            'merged': merged
        }
    


    def compare_degs(
        self,
        results1: pd.DataFrame,
        results2: pd.DataFrame,
        name1: str = "DEG1",
        name2: str = "DEG2",
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        nonsig_pval: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare two DEG results to find shared, contrasting, and specific genes.
        
        1. Shared: Significant in both, same direction
        2. Contrasting: Significant in both, opposite direction
        3. Specific: Significant in one, not significant in other
        
        Args:
            results1: First DESeq results DataFrame
            results2: Second DESeq results DataFrame
            name1: Label for first analysis
            name2: Label for second analysis
            
        Returns:
            Dict with 'shared', 'contrasting', 'specific_<name1>', 'specific_<name2>'
        """
        # Merge on symbol
        merged = results1[['symbol', 'log2FoldChange', 'padj', 'pvalue']].merge(
            results2[['symbol', 'log2FoldChange', 'padj', 'pvalue']],
            on='symbol',
            suffixes=(f'_{name1}', f'_{name2}')
        )
        
        # Significance flags
        sig1 = (merged[f'padj_{name1}'] < padj_threshold) & (merged[f'log2FoldChange_{name1}'].abs() > log2fc_threshold)
        sig2 = (merged[f'padj_{name2}'] < padj_threshold) & (merged[f'log2FoldChange_{name2}'].abs() > log2fc_threshold)
        
        # Direction
        same_dir = np.sign(merged[f'log2FoldChange_{name1}']) == np.sign(merged[f'log2FoldChange_{name2}'])
        
        # Not significant (nominal p > nonsig_pval)
        nonsig1 = merged[f'pvalue_{name1}'] > nonsig_pval
        nonsig2 = merged[f'pvalue_{name2}'] > nonsig_pval
        
        # Categorize
        shared = merged[sig1 & sig2 & same_dir].copy()
        contrasting = merged[sig1 & sig2 & ~same_dir].copy()
        specific1 = merged[sig1 & nonsig2].copy()
        specific2 = merged[sig2 & nonsig1].copy()
        
        print(f"\n=== DEG Comparison: {name1} vs {name2} ===")
        print(f"Shared (same direction):     {len(shared)}")
        print(f"Contrasting (opposite dir):  {len(contrasting)}")
        print(f"Specific to {name1}:          {len(specific1)}")
        print(f"Specific to {name2}:          {len(specific2)}")
        
        return {
            'shared': shared,
            'contrasting': contrasting,
            f'specific_{name1}': specific1,
            f'specific_{name2}': specific2
        }
# ==============================================================================
# NSCAnalysis Class - Step 3: Add gJSD method

    # Add this method to NSCAnalysis class:

    def run_gjsd(
        self,
        group1: str,
        group2: str,
        group_col: str = 'Stage',
        method: str = 'bidirectional_gjsd',
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        gene_type: str = 'both'
    ) -> Dict[str, pd.DataFrame]:
        """
        Run gJSD analysis on significant DEGs using BulkGJSD class.
        
        Args:
            group1: First group (e.g., 'E14') - will be "target" 
            group2: Second group (e.g., 'E18') - will be "other"
            group_col: Column in metadata for grouping
            method: gJSD method ('bidirectional_gjsd', 'asymmetric_gjsd', etc.)
            padj_threshold: Filter to DEGs with padj < threshold
            log2fc_threshold: Filter to DEGs with |log2FC| > threshold
            gene_type: 'TF', 'Gene', or 'both'
            
        Returns:
            Dict with 'all', 'group1_specific', 'group2_specific' DataFrames
        """
        print(f"\n{'='*60}")
        print(f"Running gJSD: {group1} vs {group2}")
        print(f"{'='*60}")
        
        # Check DESeq results exist
        if self.deseq_results is None:
            raise ValueError("Run run_deseq() first")
        
        # Get significant genes from DESeq
        sig_genes = self.deseq_results[
            (self.deseq_results['padj'] < padj_threshold) &
            (self.deseq_results['log2FoldChange'].abs() > log2fc_threshold)
        ].copy()
        
        # Filter by gene type
        if gene_type == 'TF':
            sig_genes = sig_genes[sig_genes['is_TF']]
            print(f"Filtering to TFs only")
        elif gene_type == 'Gene':
            sig_genes = sig_genes[~sig_genes['is_TF']]
            print(f"Filtering to non-TF genes only")
        
        sig_gene_ids = set(sig_genes.index)
        
        n_up = (sig_genes['log2FoldChange'] > 0).sum()
        n_down = (sig_genes['log2FoldChange'] < 0).sum()
        print(f"Significant genes: {len(sig_genes)} ({n_up} {group2}-high, {n_down} {group1}-high)")
        
        if len(sig_genes) == 0:
            print("No significant genes found!")
            return None
        
        # Get samples for each group
        group1_samples = self.metadata[self.metadata[group_col] == group1].index.tolist()
        group2_samples = self.metadata[self.metadata[group_col] == group2].index.tolist()
        
        # Filter to samples in counts
        group1_samples = [s for s in group1_samples if s in self.counts.columns]
        group2_samples = [s for s in group2_samples if s in self.counts.columns]
        
        print(f"Samples: {group1}={len(group1_samples)}, {group2}={len(group2_samples)}")
        
        # Subset counts to significant genes
        expr_sig = np.log1p(self.counts.loc[self.counts.index.isin(sig_gene_ids)] + 1)  # log1p transform
        print(f"Expression matrix: {expr_sig.shape[0]} genes x {expr_sig.shape[1]} samples")
        
        # Run BulkGJSD
        gjsd = BulkGJSD(expr_sig, n_processes=4)
        results = gjsd.compare_bidirectional(group1_samples, group2_samples, method=method)
        
        # Add DESeq2 info to results
        deseq_info = self.deseq_results[['log2FoldChange', 'padj', 'baseMean', 'is_TF', 'symbol']]
        
        for key in ['all', 'group1_specific', 'group2_specific']:
            results[key] = results[key].join(deseq_info, how='left')


        
        # Sort by signed_specificity
        results['group1_specific'] = results['group1_specific'].sort_values(
            'signed_specificity', ascending=False
        )
        results['group2_specific'] = results['group2_specific'].sort_values(
            'signed_specificity', ascending=True
        )
        
        # Store results
        self.gjsd_results = results
        self.gjsd_params = {
            'group1': group1, 
            'group2': group2, 
            'group_col': group_col,
            'method': method,
            'gene_type': gene_type
        }
        
        # Summary
        print(f"\n=== gJSD Results ===")
        print(f"Total scored: {len(results['all'])}")
        print(f"{group1}-specific: {len(results['group1_specific'])}")
        print(f"{group2}-specific: {len(results['group2_specific'])}")
        
        # Top specific for each group
        print(f"\nTop 10 {group1}-specific:")
        top_g1 = results['group1_specific'].head(10)
        print(top_g1[['symbol', 'signed_specificity', 'log2FoldChange', 'is_TF']].to_string())
        
        print(f"\nTop 10 {group2}-specific:")
        top_g2 = results['group2_specific'].head(10)
        print(top_g2[['symbol', 'signed_specificity', 'log2FoldChange', 'is_TF']].to_string())
        
        return results


    def get_gjsd_markers(
        self,
        min_specificity: float = 0.0,
        gene_type: str = 'both'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get gJSD-filtered markers.
        
        Args:
            min_specificity: Minimum |signed_specificity| threshold
            gene_type: 'TF', 'Gene', or 'both'
            
        Returns:
            Dict with group1_specific, group2_specific DataFrames
        """
        if self.gjsd_results is None:
            raise ValueError("Run run_gjsd() first")
        
        group1 = self.gjsd_params['group1']
        group2 = self.gjsd_params['group2']
        
        g1_df = self.gjsd_results['group1_specific'].copy()
        g2_df = self.gjsd_results['group2_specific'].copy()
        
        # Filter by specificity threshold
        if min_specificity > 0:
            g1_df = g1_df[g1_df['signed_specificity'].abs() >= min_specificity]
            g2_df = g2_df[g2_df['signed_specificity'].abs() >= min_specificity]
        
        # Filter by gene type
        if gene_type == 'TF':
            g1_df = g1_df[g1_df['is_TF']]
            g2_df = g2_df[g2_df['is_TF']]
        elif gene_type == 'Gene':
            g1_df = g1_df[~g1_df['is_TF']]
            g2_df = g2_df[~g2_df['is_TF']]
        
        print(f"\n=== gJSD Markers (|specificity| >= {min_specificity}, type={gene_type}) ===")
        print(f"{group1}-specific: {len(g1_df)}")
        print(f"{group2}-specific: {len(g2_df)}")
        
        return {
            f'{group1}_specific': g1_df,
            f'{group2}_specific': g2_df
        }