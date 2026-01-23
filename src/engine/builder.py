"""
Builder pattern for NSCAnalysis construction.

Provides a fluent interface for building NSCAnalysis objects with
clear validation and optional components.
"""

from pathlib import Path
from typing import List, Optional
from src.engine.pipeline import NSCAnalysis


class NSCAnalysisBuilder:
    """
    Builder for NSCAnalysis objects.
    
    Provides a fluent interface for constructing NSCAnalysis with validation
    and clear separation of required vs optional components.
    
    Examples
    --------
    >>> builder = NSCAnalysisBuilder()
    >>> analysis = (builder
    ...     .with_rna_data(counts_path, metadata_path)
    ...     .with_atac_data(atac_metadata_path)
    ...     .with_annotations(gtf_path, tf_list_path)
    ...     .with_chip_seq(chip_peaks_path)
    ...     .with_integrated_data(overlap_path, chip_annotated_path, atac_annotated_path)
    ...     .exclude_samples(['MUC9939', 'MUC9940'])
    ...     .build()
    ... )
    """
    
    def __init__(self):
        """Initialize builder with empty configuration."""
        self._counts_path: Optional[str] = None
        self._metadata_path: Optional[str] = None
        self._atac_metadata_path: Optional[str] = None
        self._overlap_df_path: Optional[str] = None
        self._chip_annotated_path: Optional[str] = None
        self._atac_annotated_path: Optional[str] = None
        self._tf_list_path: Optional[str] = None
        self._gtf_path: Optional[str] = None
        self._chip_peaks_path: Optional[str] = None
        self._exclude_samples: List[str] = []
    
    def with_rna_data(self, counts_path: str, metadata_path: str) -> 'NSCAnalysisBuilder':
        """
        Add RNA-seq data paths.
        
        Parameters
        ----------
        counts_path : str
            Path to RNA-seq counts CSV
        metadata_path : str
            Path to RNA-seq metadata CSV
            
        Returns
        -------
        NSCAnalysisBuilder
            Self for method chaining
        """
        self._counts_path = counts_path
        self._metadata_path = metadata_path
        return self
    
    def with_atac_data(self, atac_metadata_path: str) -> 'NSCAnalysisBuilder':
        """
        Add ATAC-seq metadata path.
        
        Parameters
        ----------
        atac_metadata_path : str
            Path to ATAC-seq metadata CSV
            
        Returns
        -------
        NSCAnalysisBuilder
            Self for method chaining
        """
        self._atac_metadata_path = atac_metadata_path
        return self
    
    def with_annotations(self, gtf_path: str, tf_list_path: str) -> 'NSCAnalysisBuilder':
        """
        Add annotation file paths.
        
        Parameters
        ----------
        gtf_path : str
            Path to GTF annotation file
        tf_list_path : str
            Path to transcription factor list (Excel)
            
        Returns
        -------
        NSCAnalysisBuilder
            Self for method chaining
        """
        self._gtf_path = gtf_path
        self._tf_list_path = tf_list_path
        return self
    
    def with_chip_seq(self, chip_peaks_path: Optional[str] = None) -> 'NSCAnalysisBuilder':
        """
        Add ChIP-seq peaks path (optional).
        
        Parameters
        ----------
        chip_peaks_path : str, optional
            Path to ChIP-seq peaks BED file
            
        Returns
        -------
        NSCAnalysisBuilder
            Self for method chaining
        """
        self._chip_peaks_path = chip_peaks_path
        return self
    
    def with_integrated_data(
        self, 
        overlap_df_path: str,
        chip_annotated_path: str,
        atac_annotated_path: str
    ) -> 'NSCAnalysisBuilder':
        """
        Add integrated ChIP-ATAC data paths.
        
        Parameters
        ----------
        overlap_df_path : str
            Path to ChIP ∩ ATAC overlap data
        chip_annotated_path : str
            Path to annotated ChIP peaks
        atac_annotated_path : str
            Path to annotated ATAC peaks
            
        Returns
        -------
        NSCAnalysisBuilder
            Self for method chaining
        """
        self._overlap_df_path = overlap_df_path
        self._chip_annotated_path = chip_annotated_path
        self._atac_annotated_path = atac_annotated_path
        return self
    
    def exclude_samples(self, sample_ids: List[str]) -> 'NSCAnalysisBuilder':
        """
        Specify samples to exclude from analysis.
        
        Parameters
        ----------
        sample_ids : List[str]
            List of sample IDs to exclude
            
        Returns
        -------
        NSCAnalysisBuilder
            Self for method chaining
        """
        self._exclude_samples = sample_ids
        return self
    
    def validate(self) -> None:
        """
        Validate that all required paths are set.
        
        Raises
        ------
        ValueError
            If any required path is missing
        FileNotFoundError
            If any specified path does not exist
        """
        required = {
            'counts_path': self._counts_path,
            'metadata_path': self._metadata_path,
            'atac_metadata_path': self._atac_metadata_path,
            'overlap_df_path': self._overlap_df_path,
            'chip_annotated_path': self._chip_annotated_path,
            'atac_annotated_path': self._atac_annotated_path,
            'tf_list_path': self._tf_list_path,
            'gtf_path': self._gtf_path,
        }
        
        # Check all required paths are set
        missing = [name for name, path in required.items() if path is None]
        if missing:
            raise ValueError(f"Missing required paths: {', '.join(missing)}")
        
        # Check all paths exist
        all_paths = {**required}
        if self._chip_peaks_path:
            all_paths['chip_peaks_path'] = self._chip_peaks_path
        
        for name, path in all_paths.items():
            if path and not Path(path).exists():
                raise FileNotFoundError(f"{name}: {path}")
    
    def build(self) -> NSCAnalysis:
        """
        Build and return NSCAnalysis object.
        
        Returns
        -------
        NSCAnalysis
            Configured analysis object with data loaded
            
        Raises
        ------
        ValueError
            If required configuration is missing
        FileNotFoundError
            If any file path does not exist
        """
        self.validate()
        
        return NSCAnalysis(
            counts_path=self._counts_path,
            metadata_path=self._metadata_path,
            atac_metadata_path=self._atac_metadata_path,
            overlap_df_path=self._overlap_df_path,
            chip_annotated_path=self._chip_annotated_path,
            atac_annotated_path=self._atac_annotated_path,
            tf_list_path=self._tf_list_path,
            gtf_path=self._gtf_path,
            chip_peaks_path=self._chip_peaks_path,
            exclude_samples=self._exclude_samples
        )
    
    def build_from_dict(self, config: dict) -> NSCAnalysis:
        """
        Build NSCAnalysis from a configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with keys matching builder methods
            
        Returns
        -------
        NSCAnalysis
            Configured analysis object
            
        Examples
        --------
        >>> config = {
        ...     'counts_path': 'data/counts.csv',
        ...     'metadata_path': 'data/metadata.csv',
        ...     'gtf_path': 'annotations/genome.gtf',
        ...     # ... other paths
        ... }
        >>> analysis = NSCAnalysisBuilder().build_from_dict(config)
        """
        if 'counts_path' in config and 'metadata_path' in config:
            self.with_rna_data(config['counts_path'], config['metadata_path'])
        
        if 'atac_metadata_path' in config:
            self.with_atac_data(config['atac_metadata_path'])
        
        if 'gtf_path' in config and 'tf_list_path' in config:
            self.with_annotations(config['gtf_path'], config['tf_list_path'])
        
        if 'chip_peaks_path' in config:
            self.with_chip_seq(config['chip_peaks_path'])
        
        if all(k in config for k in ['overlap_df_path', 'chip_annotated_path', 'atac_annotated_path']):
            self.with_integrated_data(
                config['overlap_df_path'],
                config['chip_annotated_path'],
                config['atac_annotated_path']
            )
        
        if 'exclude_samples' in config:
            self.exclude_samples(config['exclude_samples'])
        
        return self.build()


# Convenience function
def build_analysis(**kwargs) -> NSCAnalysis:
    """
    Convenience function to build NSCAnalysis from keyword arguments.
    
    Parameters
    ----------
    **kwargs
        Keyword arguments matching NSCAnalysisBuilder methods
        
    Returns
    -------
    NSCAnalysis
        Configured analysis object
        
    Examples
    --------
    >>> analysis = build_analysis(
    ...     counts_path='data/counts.csv',
    ...     metadata_path='data/metadata.csv',
    ...     gtf_path='annotations/genome.gtf',
    ...     # ... other arguments
    ... )
    """
    return NSCAnalysisBuilder().build_from_dict(kwargs)
