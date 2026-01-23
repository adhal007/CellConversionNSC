# ==============================================================================
# NSCAnalysis Class - Refactored with modular imports
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

# Import our modular io functions
from src.io.gtf_parser import parse_gtf_gene_mapping, get_promoters_from_gtf

# NOTE: Heavy imports (pydeseq2, pybedtools, pyranges, scipy, statsmodels, BulkGJSD)
# are imported lazily inside methods that use them to speed up module loading


class NSCAnalysis:
    """
    Analysis pipeline for NSC cell conversion project.
    
    Questions addressed:
        1. What are pro-neural and pro-glial signatures?
        2. What factors can convert glial to neuronal?

    Enhanced with loading of the Chip seq bed files
    """
    
    def __init__(
        self,
        counts_path: str,
        metadata_path: str,
        atac_metadata_path: str,
        overlap_df_path: str,
        chip_annotated_path: str,
        atac_annotated_path: str,
        tf_list_path: str,
        gtf_path: str,
        chip_peaks_path: str = None,
        exclude_samples: List[str] = None
    ):
        # Store paths
        self.paths = {
            'counts': Path(counts_path),
            'metadata': Path(metadata_path),
            'atac_metadata': Path(atac_metadata_path),
            'overlap_df': Path(overlap_df_path),
            'chip_annotated': Path(chip_annotated_path),
            'atac_annotated': Path(atac_annotated_path),
            'tf_list': Path(tf_list_path),
            'gtf': Path(gtf_path),
            'chip_peaks': Path(chip_peaks_path) if chip_peaks_path else None
        }
        
        # Validate paths exist
        for name, path in self.paths.items():
            if path is not None and not path.exists():
                raise FileNotFoundError(f"{name}: {path}")
        
        # Store exclude list
        self.exclude_samples = exclude_samples or []
        
        # Data containers
        self.counts = None
        self.counts_unfiltered = None
        self.metadata = None
        self.atac_metadata = None
        self.merged_metadata = None
        self.atac_to_rna_map = None
        self.overlap_df = None
        self.chip_annotated = None
        self.atac_annotated = None
        self.tf_symbols = None
        self.tf_ensembl = None
        self.ensembl_to_symbol = None
        
        # Results containers
        self.deseq_results = None
        self.gjsd_results = None
        self.dar_results = None
        self.consensus_peaks = None
        self.atac_signal_matrix = None
        self.diff_overlap_df = None
        self.gene_lengths = None
        self.tpm = None
        self.fpkm = None
        
        # NEW containers
        self.promoters = None
        self.chip_peaks = None
        
        # Load data
        self._load_data()

    def _load_data(self):
        """Load all input data using repositories."""
        from src.io.data_loaders import create_repositories
        
        print("=" * 60)
        print("Loading data...")
        print("=" * 60)
        
        # Create repository factory
        repos = create_repositories()
        
        # 1. Load counts
        print("\n[1/9] Loading counts...")
        self.counts = repos.rna.load_counts(str(self.paths['counts']))
        
        # 2. Load RNA-seq metadata
        print("\n[2/9] Loading RNA-seq metadata...")
        self.metadata = repos.rna.load_metadata(str(self.paths['metadata']))
        
        # 3. Load ATAC-seq metadata
        print("\n[3/9] Loading ATAC-seq metadata...")
        self.atac_metadata = repos.atac.load_metadata(str(self.paths['atac_metadata']))
        
        # 4. Create merged metadata and mapping
        print("\n[4/9] Creating ATAC-to-RNA sample mapping...")
        self._create_sample_mapping()
        
        # 5. Remove outlier samples
        if self.exclude_samples:
            print(f"\n[5/9] Removing outlier samples...")
            samples_to_remove = [s for s in self.exclude_samples if s in self.counts.columns]
            if samples_to_remove:
                self.counts = self.counts.drop(columns=samples_to_remove)
                self.metadata = self.metadata.drop(index=[s for s in samples_to_remove if s in self.metadata.index])
                print(f"      Removed: {samples_to_remove}")
                print(f"      Remaining: {self.counts.shape[1]} samples")
            else:
                print(f"      No matching samples found to remove")
        else:
            print("\n[5/9] No outlier samples to remove")
        
        # 6. Filter low-count genes
        print("\n[6/9] Filtering low-count genes...")
        self._filter_low_counts(min_counts=10, min_samples=3, group_col='Stage')
        
        # 7. Load overlap_df
        print("\n[7/9] Loading overlap_df (ChIP ∩ ATAC)...")
        self.overlap_df = repos.integrated.load_overlap_data(str(self.paths['overlap_df']))
        
        # 8. Load chip_annotated
        print("\n[8/9] Loading chip_annotated...")
        self.chip_annotated = repos.chip.load_annotated_peaks(str(self.paths['chip_annotated']))
        
        # 9. Load atac_annotated
        print("\n[9/9] Loading atac_annotated...")
        self.atac_annotated = repos.atac.load_annotated_peaks(str(self.paths['atac_annotated']))
        
        # 10. Load TF list
        print("\n[10/10] Loading TF list...")
        self.tf_df, self.tf_symbols, self.tf_ensembl = repos.annotations.load_tf_list(str(self.paths['tf_list']))
        
        # Check TFs remaining after filtering
        tfs_in_counts = len([g for g in self.counts.index if g in self.tf_ensembl])
        print(f"      TFs in filtered counts: {tfs_in_counts}")
        
        # 11. Parse GTF for gene mappings using modular function
        print("\n[11/11] Parsing GTF for gene symbol mappings...")
        self.ensembl_to_symbol = parse_gtf_gene_mapping(str(self.paths['gtf']))
        
        mapped = sum(1 for g in self.counts.index if g in self.ensembl_to_symbol)
        print(f"      Counts genes with mapping: {mapped} / {len(self.counts)}")
        
        print("\n" + "=" * 60)
        print("Data loaded successfully!")
        print("=" * 60)

        # 12. Generate promoters from GTF using modular function
        print("\n[12/13] Generating promoters from GTF...")
        self.promoters = get_promoters_from_gtf(str(self.paths['gtf']), window=2000)
        print(f"      {len(self.promoters)} promoters")
        
        # 13. Load ChIP peaks (if provided)
        if self.paths['chip_peaks'] and self.paths['chip_peaks'].exists():
            print("\n[13/13] Loading ChIP-seq peaks...")
            self.chip_peaks = repos.chip.load_chip_peaks_bed(str(self.paths['chip_peaks']))
            print(f"      {len(self.chip_peaks)} peaks")
            print(f"      {self.chip_peaks['TF'].nunique()} unique TFs")
        else:
            print("\n[13/13] No ChIP peaks file provided, skipping...")
            self.chip_peaks = None
        
        print("\n" + "=" * 60)
        print("All data loaded successfully!")
        print("=" * 60)


    def _filter_low_counts(
        self,
        min_counts: int = 10,
        min_samples: int = None,
        min_samples_pct: float = None,
        min_cpm: float = None,
        group_col: str = None
    ) -> pd.DataFrame:
        """
        Filter out lowly expressed genes.
        
        Args:
            min_counts: Minimum count threshold
            min_samples: Minimum number of samples meeting threshold
            min_samples_pct: Minimum percentage of samples (alternative to min_samples)
            min_cpm: If set, use CPM threshold instead of raw counts
            group_col: If set, require threshold in at least one group
            
        Returns:
            Filtered count matrix
        """
        print(f"\n{'='*60}")
        print("Filtering low-count genes")
        print(f"{'='*60}")
        
        n_genes_before = len(self.counts)
        n_samples = len(self.counts.columns)
        
        # Determine min_samples
        if min_samples is None and min_samples_pct is None:
            # Default: smallest group size
            if group_col and group_col in self.metadata.columns:
                min_samples = self.metadata[group_col].value_counts().min()
            else:
                min_samples = max(3, int(n_samples * 0.1))  # 10% or at least 3
        elif min_samples_pct is not None:
            min_samples = max(1, int(n_samples * min_samples_pct))
        
        print(f"Threshold: {'CPM > ' + str(min_cpm) if min_cpm else 'counts >= ' + str(min_counts)} in >= {min_samples} samples")
        
        if group_col and group_col in self.metadata.columns:
            # Filter per group: keep gene if it passes in ANY group
            groups = self.metadata[group_col].unique()
            keep_genes = set()
            
            for group in groups:
                group_samples = self.metadata[self.metadata[group_col] == group].index
                group_samples = [s for s in group_samples if s in self.counts.columns]
                group_counts = self.counts[group_samples]
                
                if min_cpm:
                    # CPM filtering
                    lib_sizes = group_counts.sum(axis=0)
                    cpm = group_counts.div(lib_sizes, axis=1) * 1e6
                    passes = (cpm > min_cpm).sum(axis=1) >= min(min_samples, len(group_samples))
                else:
                    # Count filtering
                    passes = (group_counts >= min_counts).sum(axis=1) >= min(min_samples, len(group_samples))
                
                keep_genes.update(self.counts.index[passes])
                print(f"  {group}: {passes.sum()} genes pass filter")
            
            keep_mask = self.counts.index.isin(keep_genes)
        else:
            # Simple filtering across all samples
            if min_cpm:
                lib_sizes = self.counts.sum(axis=0)
                cpm = self.counts.div(lib_sizes, axis=1) * 1e6
                keep_mask = (cpm > min_cpm).sum(axis=1) >= min_samples
            else:
                keep_mask = (self.counts >= min_counts).sum(axis=1) >= min_samples
        
        # Apply filter
        self.counts_unfiltered = self.counts.copy()  # Keep original
        self.counts = self.counts[keep_mask]
        
        n_genes_after = len(self.counts)
        n_removed = n_genes_before - n_genes_after
        
        print(f"\nGenes before: {n_genes_before}")
        print(f"Genes after:  {n_genes_after}")
        print(f"Removed:      {n_removed} ({100*n_removed/n_genes_before:.1f}%)")
        
        # Update TF list if exists
        if hasattr(self, 'tf_ensembl') and self.tf_ensembl is not None:
            tfs_before = len([g for g in self.counts_unfiltered.index if g in self.tf_ensembl])
            tfs_after = len([g for g in self.counts.index if g in self.tf_ensembl])
            print(f"TFs before: {tfs_before}, after: {tfs_after}")

    def _create_sample_mapping(self):
        """
        Create mapping between ATAC and RNA sample IDs based on 
        Stage, Region, Marker, and Replicate.
        """
        # Standardize ATAC metadata columns to match RNA naming
        atac = self.atac_metadata.copy()
        
        # Map ATAC column names to RNA column names
        # ATAC: Tissue, Factor, Condition, Replicate
        # RNA: Region, Marker, Stage, Replicate
        
        # Tissue -> Region
        atac['Region'] = atac['Tissue'].map({'Cortex': 'Ctx', 'LGE': 'LGE'})
        
        # Factor -> Marker
        atac['Marker'] = atac['Factor'].map({'CD133': 'Progenitors', 'PSA-NCAM': 'PSANCAM'})
        
        # Condition -> Stage
        atac['Stage'] = atac['Condition']
        
        # Build mapping
        self.atac_to_rna_map = {}
        unmatched = []
        
        for atac_id in atac.index:
            atac_row = atac.loc[atac_id]
            
            match = self.metadata[
                (self.metadata['Stage'] == atac_row['Stage']) &
                (self.metadata['Region'] == atac_row['Region']) &
                (self.metadata['Marker'] == atac_row['Marker']) &
                (self.metadata['Replicate'] == atac_row['Replicate'])
            ]
            
            if len(match) == 1:
                self.atac_to_rna_map[atac_id] = match.index[0]
            elif len(match) > 1:
                print(f"      Warning: Multiple matches for {atac_id}")
                self.atac_to_rna_map[atac_id] = match.index[0]  # Take first
            else:
                unmatched.append(atac_id)
        
        print(f"      Mapped: {len(self.atac_to_rna_map)} / {len(atac)} ATAC samples")
        if unmatched:
            print(f"      Unmatched ATAC samples: {unmatched}")
        
        # Create merged metadata
        atac['RNA_SampleID'] = atac.index.map(self.atac_to_rna_map)
        self.merged_metadata = atac

    def create_merged_regulatory_regions(self, promoters, enhancers):
        """
        Merge promoters and enhancers into unified regulatory regions.
        
        SIMPLE LOGIC:
        1. Same-gene overlap → Merge into "proximal_regulatory"
        2. Everything else → Keep as-is ("promoter" or "enhancer")
        
        Different-gene overlaps are FINE - both regions stay separate.
        """
        # Lazy import - only load when this method is called
        import pyranges as pr
        
        print("\n" + "="*80)
        print("CREATING MERGED REGULATORY REGIONS")
        print("="*80)
        
        print(f"Input:")
        print(f"  Promoters: {len(promoters):,}")
        print(f"  Enhancers: {len(enhancers):,}")
        
        # Convert to PyRanges
        promoters_pr = pr.PyRanges(promoters.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}
        ))
        
        enhancers_pr = pr.PyRanges(enhancers.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End', 'symbol': 'gene'}
        ))
        
        # Find overlaps
        all_overlaps = promoters_pr.join(enhancers_pr).df
        
        # Get SAME-GENE overlaps
        same_gene_overlaps = all_overlaps[
            all_overlaps['gene'] == all_overlaps['gene_b']
        ].copy()
        
        print(f"\nSame-gene promoter-enhancer overlaps: {len(same_gene_overlaps):,}")
        
        # Track which promoters/enhancers are in same-gene overlaps
        promoters_in_proximal = set(
            zip(same_gene_overlaps['gene'], 
                same_gene_overlaps['Chromosome'],
                same_gene_overlaps['Start'], 
                same_gene_overlaps['End'])
        )
        
        enhancers_in_proximal = set(
            zip(same_gene_overlaps['gene_b'],
                same_gene_overlaps['Chromosome'],
                same_gene_overlaps['Start_b'], 
                same_gene_overlaps['End_b'])
        )
        
        # 1. Create proximal regulatory regions (merge same-gene overlaps)
        proximal_regions = []
        for _, row in same_gene_overlaps.iterrows():
            merged_start = min(row['Start'], row['Start_b'])
            merged_end = max(row['End'], row['End_b'])
            
            proximal_regions.append({
                'chr': row['Chromosome'],
                'start': merged_start,
                'end': merged_end,
                'gene': row['gene'],
                'region_type': 'proximal_regulatory'
            })
        
        proximal_df = pd.DataFrame(proximal_regions).drop_duplicates()
        
        # 2. Keep promoters that DON'T have same-gene overlap
        promoter_regions = []
        for _, prom in promoters.iterrows():
            prom_key = (prom['gene'], prom['chr'], prom['start'], prom['end'])
            if prom_key not in promoters_in_proximal:
                promoter_regions.append({
                    'chr': prom['chr'],
                    'start': prom['start'],
                    'end': prom['end'],
                    'gene': prom['gene'],
                    'region_type': 'promoter'
                })
        
        promoter_df = pd.DataFrame(promoter_regions)
        
        # 3. Keep enhancers that DON'T have same-gene overlap
        enhancer_regions = []
        for _, enh in enhancers.iterrows():
            enh_key = (enh['symbol'], enh['chr'], enh['start'], enh['end'])
            if enh_key not in enhancers_in_proximal:
                enhancer_regions.append({
                    'chr': enh['chr'],
                    'start': enh['start'],
                    'end': enh['end'],
                    'gene': enh['symbol'],
                    'region_type': 'enhancer'
                })
        
        enhancer_df = pd.DataFrame(enhancer_regions)
        
        # Combine
        merged_regulatory = pd.concat([
            promoter_df,
            proximal_df,
            enhancer_df
        ], ignore_index=True)
        
        print(f"\nOutput:")
        print(f"  Promoter: {len(promoter_df):,}")
        print(f"  Proximal regulatory: {len(proximal_df):,}")
        print(f"  Enhancer: {len(enhancer_df):,}")
        print(f"  Total: {len(merged_regulatory):,}")
        
        return merged_regulatory


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
        # Lazy imports - only load when DESeq2 is actually run
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        
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

        self.normed_counts = dds.layers['normed_counts']
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


# The rest of the pipeline methods will be kept for now
# We'll continue refactoring in the next steps
