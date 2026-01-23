"""
IO utilities for loading and parsing genomic data files.
"""

from .gtf_parser import (
    parse_gtf_gene_mapping,
    get_promoters_from_gtf,
    parse_gtf_attributes
)
from .reg_region_integrator import (
    ConsensusPeakBuilder,
    EnhancerIntegrator,
    RegulatoryRegionMaker
)
from .data_loaders import (
    RNASeqRepository,
    ATACSeqRepository,
    ChIPSeqRepository,
    IntegratedDataRepository,
    AnnotationRepository,
    DataRepositoryFactory,
    create_repositories
)

__all__ = [
    'parse_gtf_gene_mapping',
    'get_promoters_from_gtf',
    'parse_gtf_attributes',
    'ConsensusPeakBuilder',
    'EnhancerIntegrator',
    'RegulatoryRegionMaker',
    'RNASeqRepository',
    'ATACSeqRepository',
    'ChIPSeqRepository',
    'IntegratedDataRepository',
    'AnnotationRepository',
    'DataRepositoryFactory',
    'create_repositories',
]
