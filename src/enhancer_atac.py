"""
Backward compatibility wrapper for enhancer_atac imports.
This allows old code using 'import src.enhancer_atac' to work with the new modular structure.
"""

# Import from the new location
from src.io.reg_region_integrator import ConsensusPeakBuilder, EnhancerIntegrator

# Make them available at the old import path
__all__ = ['ConsensusPeakBuilder', 'EnhancerIntegrator']
