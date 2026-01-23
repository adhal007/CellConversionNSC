"""
Strategy Pattern for Differential Expression Analysis.

This module implements the Strategy pattern to make differential expression
algorithms interchangeable. Based on refactoring.guru/design-patterns/strategy
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class DifferentialExpressionStrategy(ABC):
    """
    Abstract base class for differential expression strategies.
    
    This defines the interface that all DE strategies must implement.
    Following the Strategy pattern from refactoring.guru.
    """
    
    @abstractmethod
    def analyze(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        group1: str,
        group2: str,
        group_col: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform differential expression analysis.
        
        Parameters
        ----------
        counts : pd.DataFrame
            Gene expression counts (genes x samples)
        metadata : pd.DataFrame
            Sample metadata with group information
        group1 : str
            Reference group name
        group2 : str
            Comparison group name
        group_col : str
            Metadata column containing group assignments
        **kwargs
            Algorithm-specific parameters
            
        Returns
        -------
        pd.DataFrame
            Results with columns: gene_id, log2FoldChange, pvalue, padj, etc.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this strategy."""
        pass


class DESeq2Strategy(DifferentialExpressionStrategy):
    """
    DESeq2 differential expression strategy.
    
    Uses the PyDESeq2 implementation of DESeq2.
    """
    
    def get_name(self) -> str:
        return "DESeq2"
    
    def analyze(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        group1: str,
        group2: str,
        group_col: str,
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        n_cpus: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run DESeq2 differential expression analysis.
        
        Parameters
        ----------
        counts : pd.DataFrame
            Gene expression counts (genes x samples)
        metadata : pd.DataFrame
            Sample metadata
        group1 : str
            Reference group
        group2 : str
            Comparison group
        group_col : str
            Column in metadata with groups
        padj_threshold : float
            Adjusted p-value threshold
        log2fc_threshold : float
            Log2 fold change threshold
        n_cpus : int
            Number of CPUs to use
            
        Returns
        -------
        pd.DataFrame
            DESeq2 results
        """
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        
        print(f"\n{'='*60}")
        print(f"Running {self.get_name()}: {group1} vs {group2}")
        print(f"{'='*60}")
        
        # Validate groups
        if group_col not in metadata.columns:
            raise ValueError(f"Column '{group_col}' not in metadata")
        
        unique_groups = metadata[group_col].unique()
        if group1 not in unique_groups or group2 not in unique_groups:
            raise ValueError(f"Groups must be in {unique_groups}")
        
        # Filter samples for the two groups
        samples_mask = metadata[group_col].isin([group1, group2])
        samples = metadata[samples_mask].index.tolist()
        
        # Subset counts and metadata
        counts_subset = counts[samples].T  # DESeq2 wants samples as rows
        metadata_subset = metadata.loc[samples, [group_col]].copy()
        
        print(f"\nSamples: {len(samples)} ({group1}: {sum(metadata_subset[group_col]==group1)}, {group2}: {sum(metadata_subset[group_col]==group2)})")
        
        # Ensure counts are integers
        counts_subset = counts_subset.astype(int)
        
        # Create DESeq dataset
        print("\nCreating DESeq2 dataset...")
        dds = DeseqDataSet(
            counts=counts_subset,
            metadata=metadata_subset,
            design_factors=group_col,
            n_cpus=n_cpus
        )
        
        # Run DESeq2
        print("Running DESeq2...")
        dds.deseq2()
        
        # Get results
        print(f"Extracting results ({group1} vs {group2})...")
        stat_res = DeseqStats(dds, contrast=[group_col, group1, group2], n_cpus=n_cpus)
        stat_res.summary()
        
        # Get results as DataFrame
        results = stat_res.results_df.copy()
        results['gene_id'] = results.index
        
        # Add significance flag
        sig_mask = (results['padj'] < padj_threshold) & (results['log2FoldChange'].abs() > log2fc_threshold)
        results['significant'] = sig_mask
        
        # Add direction
        results['direction'] = np.where(
            results['log2FoldChange'] > 0,
            group2,  # Positive = higher in group2
            group1   # Negative = higher in group1
        )
        
        # Sort by adjusted p-value
        results = results.sort_values('padj')
        
        # Store normalized counts
        self.normed_counts = dds.layers['normed_counts']
        
        print(f"\n{'='*60}")
        print(f"{self.get_name()} Results Summary")
        print(f"{'='*60}")
        print(f"Total genes tested: {len(results)}")
        print(f"Significant (padj < {padj_threshold}, |log2FC| > {log2fc_threshold}): {sig_mask.sum()}")
        
        return results


class GJSDStrategy(DifferentialExpressionStrategy):
    """
    GJSD (Geometric Jensen-Shannon Divergence) strategy.
    
    Uses the BulkGJSD implementation for differential specificity.
    """
    
    def get_name(self) -> str:
        return "GJSD"
    
    def analyze(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        group1: str,
        group2: str,
        group_col: str,
        method: str = 'gaussian_gjsd',
        alpha: float = 0.5,
        n_processes: int = 4,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run GJSD differential specificity analysis.
        
        Parameters
        ----------
        counts : pd.DataFrame
            Gene expression counts (genes x samples)
        metadata : pd.DataFrame
            Sample metadata
        group1 : str
            Reference group
        group2 : str
            Comparison group
        group_col : str
            Column in metadata with groups
        method : str
            GJSD method to use
        alpha : float
            Skew parameter for skewed GJSD
        n_processes : int
            Number of parallel processes
            
        Returns
        -------
        pd.DataFrame
            GJSD results
        """
        from src.models.stats.bulk_gjsd import BulkGJSD
        
        print(f"\n{'='*60}")
        print(f"Running {self.get_name()}: {group1} vs {group2}")
        print(f"{'='*60}")
        
        # Validate groups
        if group_col not in metadata.columns:
            raise ValueError(f"Column '{group_col}' not in metadata")
        
        # Get samples for each group
        target_samples = metadata[metadata[group_col] == group2].index.tolist()
        other_samples = metadata[metadata[group_col] == group1].index.tolist()
        
        print(f"\nSamples: {len(target_samples)} {group2} vs {len(other_samples)} {group1}")
        
        # Initialize GJSD
        gjsd = BulkGJSD(counts, n_processes=n_processes)
        
        # Run comparison
        print(f"\nRunning {method}...")
        results = gjsd.compare(
            target_samples=target_samples,
            other_samples=other_samples,
            method=method,
            alpha=alpha,
            return_full=True
        )
        
        # Add gene_id column
        results['gene_id'] = results.index
        
        # Add direction (for compatibility)
        results['direction'] = np.where(
            results.get('specificity_target', results.get('score', 0)) > 0,
            group2,
            group1
        )
        
        print(f"\n{'='*60}")
        print(f"{self.get_name()} Results Summary")
        print(f"{'='*60}")
        print(f"Total genes tested: {len(results)}")
        
        return results


class DifferentialExpressionContext:
    """
    Context class that uses a DifferentialExpressionStrategy.
    
    This is the main class users interact with. It delegates the actual
    analysis to the strategy object.
    """
    
    def __init__(self, strategy: Optional[DifferentialExpressionStrategy] = None):
        """
        Initialize with a strategy.
        
        Parameters
        ----------
        strategy : DifferentialExpressionStrategy, optional
            The strategy to use. Defaults to DESeq2Strategy.
        """
        self._strategy = strategy or DESeq2Strategy()
    
    @property
    def strategy(self) -> DifferentialExpressionStrategy:
        """Get the current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: DifferentialExpressionStrategy):
        """Set a new strategy."""
        self._strategy = strategy
        print(f"Strategy changed to: {strategy.get_name()}")
    
    def analyze(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        group1: str,
        group2: str,
        group_col: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run differential expression using the current strategy.
        
        Parameters
        ----------
        counts : pd.DataFrame
            Gene expression counts
        metadata : pd.DataFrame
            Sample metadata
        group1 : str
            Reference group
        group2 : str
            Comparison group
        group_col : str
            Metadata column with groups
        **kwargs
            Strategy-specific parameters
            
        Returns
        -------
        pd.DataFrame
            Analysis results
        """
        return self._strategy.analyze(
            counts, metadata, group1, group2, group_col, **kwargs
        )
