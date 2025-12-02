# %% Extended gJSD class for bulk RNA-seq with bidirectional specificity
# Includes asymmetric variants based on Nielsen's geometric JSD formulation
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class AsymmetricResult:
    """Container for asymmetric gJSD results"""
    magnitude: float          # Symmetric magnitude (original gJSD)
    direction: float          # Signed direction score (+ = target-specific, - = other-specific)
    signed_specificity: float # magnitude * sign(direction)
    kl_target_to_other: float # KL(target || other)
    kl_other_to_target: float # KL(other || target)
    specificity_target: float # How specific to target population
    specificity_other: float  # How specific to other population


class BulkGJSD:
    """
    Geometric Jensen-Shannon Divergence for bulk RNA-seq differential specificity.
    
    Implements multiple methods including:
    - gaussian_gjsd: Symmetric G-JSD (Nielsen 2019/2025)
    - asymmetric_gjsd: G-JSD with directional decomposition
    - skew_gjsd: Skewed G-JSD with tunable alpha parameter
    - bidirectional_gjsd: Full bidirectional specificity scores
    
    For comparing two populations (target vs other), returns:
    - Magnitude of differential expression/specificity
    - Direction indicating which population the feature is specific to
    """
    
    GAUSSIAN_METHODS = (
        'gaussian_gjsd', 
        'extended_gjsd', 
        'asymmetric_gjsd',
        'skew_gjsd',
        'bidirectional_gjsd',
        'kl_directional'
    )
    
    def __init__(self, counts_df: pd.DataFrame, n_processes: int = 4):
        """
        Initialize with expression counts DataFrame.
        
        Parameters
        ----------
        counts_df : pd.DataFrame
            Gene expression matrix (genes x samples), ideally log-transformed
        n_processes : int
            Number of parallel processes for computation
        """
        self.counts = counts_df
        self.n_processes = n_processes
    
    def compare(self, target_samples: List[str], other_samples: List[str], 
                method: str = 'gaussian_gjsd', alpha: float = 0.5,
                return_full: bool = False) -> pd.DataFrame:
        """
        Compare two sample groups and compute specificity scores.
        
        Parameters
        ----------
        target_samples : List[str]
            Sample names for target/foreground group
        other_samples : List[str]
            Sample names for other/background group
        method : str
            One of: 'gaussian_gjsd', 'asymmetric_gjsd', 'skew_gjsd', 
                    'bidirectional_gjsd', 'kl_directional', or discrete methods
        alpha : float
            Skew parameter for skew_gjsd (0.5 = symmetric, <0.5 biases toward other,
            >0.5 biases toward target)
        return_full : bool
            If True, return all component scores for asymmetric methods
            
        Returns
        -------
        pd.DataFrame
            DataFrame with specificity scores, indexed by gene
        """
        X_target = self.counts[target_samples].values.T
        X_other = self.counts[other_samples].values.T
        
        n_target = len(target_samples)
        n_other = len(other_samples)
        
        # Calculate statistics for both groups
        mu_other = X_other.mean(axis=0)
        var_other = X_other.var(axis=0, ddof=1) if n_other > 1 else np.full_like(mu_other, 0.1)
        
        mu_target = X_target.mean(axis=0)
        var_target = X_target.var(axis=0, ddof=1) if n_target > 1 else np.full_like(mu_target, 0.1)
        
        # Calculate fallback variances
        var_fallback_target = self._compute_fallback_variance(mu_target, var_target)
        var_fallback_other = self._compute_fallback_variance(mu_other, var_other)
        
        print(f"Variance fallback - Target: {var_fallback_target:.4f}, Other: {var_fallback_other:.4f}")
        
        # Prepare gene data
        gene_data_list = []
        for i, gene in enumerate(self.counts.index):
            if method in self.GAUSSIAN_METHODS:
                gene_data_list.append((
                    gene,
                    mu_target[i], var_target[i],
                    mu_other[i], var_other[i],
                    var_fallback_target, var_fallback_other,
                    alpha
                ))
            else:
                # Legacy discrete methods
                gene_data_list.append((
                    gene, X_target[:, i], n_target, mu_other[i]
                ))
        
        # Compute scores
        compute_func = partial(self._compute_gjsd_single, method=method)
        
        with Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(compute_func, gene_data_list, chunksize=100),
                total=len(gene_data_list),
                desc=f"gJSD ({method})"
            ))
        
        # Build results DataFrame based on method
        df = self._build_results_df(results, method, target_samples, other_samples, return_full)
        
        return df
    
    def _compute_fallback_variance(self, mu: np.ndarray, var: np.ndarray) -> float:
        """Compute fallback variance from expressed genes."""
        expressed_mask = mu > 0
        if expressed_mask.sum() > 0:
            return float(np.mean(var[expressed_mask]))
        return 0.1
    
    def _build_results_df(self, results: List, method: str, 
                          target_samples: List[str], other_samples: List[str],
                          return_full: bool) -> pd.DataFrame:
        """Build results DataFrame with appropriate columns for method."""
        
        if method in ('asymmetric_gjsd', 'bidirectional_gjsd', 'kl_directional'):
            # Full asymmetric results
            df = pd.DataFrame(results, columns=[
                'gene', 'gjsd_score', 'direction', 'signed_specificity',
                'kl_target_other', 'kl_other_target', 
                'specificity_target', 'specificity_other'
            ])
        elif method == 'skew_gjsd':
            df = pd.DataFrame(results, columns=[
                'gene', 'gjsd_score', 'gjsd_alpha', 'gjsd_1_minus_alpha'
            ])
        else:
            df = pd.DataFrame(results, columns=['gene', 'gjsd_score'])
        
        # Add expression statistics
        df['mean_target'] = self.counts[target_samples].mean(axis=1).values
        df['mean_other'] = self.counts[other_samples].mean(axis=1).values
        # df['log2FC'] = np.log2((df['mean_target'] + 1) / (df['mean_other'] + 1))
        df = df.set_index('gene')
        
        # Sort by appropriate column
        if 'signed_specificity' in df.columns:
            # For bidirectional, sort by absolute signed specificity
            df['abs_specificity'] = df['signed_specificity'].abs()
            df = df.sort_values('abs_specificity', ascending=False)
        else:
            df = df.sort_values('gjsd_score', ascending=False)
        
        return df
    
    @staticmethod
    def _compute_gjsd_single(gene_data: Tuple, method: str = 'gaussian_gjsd') -> Tuple:
        """
        Compute gJSD for a single gene.
        
        Returns tuple with results depending on method.
        """
        eps = 1e-12
        
        # Parse gene data based on method
        if len(gene_data) == 8:
            # Gaussian methods with full data
            (gene_name, mu_target, var_target, mu_other, var_other,
             var_fallback_target, var_fallback_other, alpha) = gene_data
        elif len(gene_data) == 4:
            # Legacy discrete methods
            gene_name, target_expr, n_target, mu_other = gene_data
            return BulkGJSD._compute_discrete_single(
                gene_name, target_expr, n_target, mu_other, method
            )
        else:
            raise ValueError(f"Unexpected gene_data length: {len(gene_data)}")
        
        # Prepare values
        mu1, mu2 = float(mu_target), float(mu_other)
        v1 = float(var_target) if var_target > eps else var_fallback_target
        v2 = float(var_other) if var_other > eps else var_fallback_other
        
        # Both means near zero = not expressed anywhere
        if abs(mu1) < eps and abs(mu2) < eps:
            if method in ('asymmetric_gjsd', 'bidirectional_gjsd', 'kl_directional'):
                return (gene_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif method == 'skew_gjsd':
                return (gene_name, 0.0, 0.0, 0.0)
            else:
                return (gene_name, 0.0)
        
        # =====================================================================
        # SYMMETRIC GAUSSIAN G-JSD (Nielsen 2019/2025)
        # =====================================================================
        if method in ('gaussian_gjsd', 'extended_gjsd'):
            gjsd = BulkGJSD._gaussian_gjsd_symmetric(mu1, v1, mu2, v2)
            return (gene_name, gjsd)
        
        # =====================================================================
        # ASYMMETRIC G-JSD with KL decomposition
        # =====================================================================
        elif method in ('asymmetric_gjsd', 'bidirectional_gjsd'):
            return BulkGJSD._gaussian_gjsd_asymmetric(gene_name, mu1, v1, mu2, v2)
        
        # =====================================================================
        # KL DIRECTIONAL (pure KL-based directionality)
        # =====================================================================
        elif method == 'kl_directional':
            return BulkGJSD._kl_directional(gene_name, mu1, v1, mu2, v2)
        
        # =====================================================================
        # SKEW G-JSD with alpha parameter
        # =====================================================================
        elif method == 'skew_gjsd':
            return BulkGJSD._gaussian_gjsd_skew(gene_name, mu1, v1, mu2, v2, alpha)
        
        else:
            raise ValueError(f"Unknown Gaussian method: {method}")
    
    @staticmethod
    def _gaussian_gjsd_symmetric(mu1: float, v1: float, mu2: float, v2: float) -> float:
        """
        Symmetric G-JSD using Jeffreys + Bhattacharyya (Nielsen formulation).
        
        G-JSD = (1/4) * Jeffreys - Bhattacharyya
        """
        # Jeffreys Divergence: J(P1, P2) = KL(P1||P2) + KL(P2||P1)
        variance_ratio_term = 0.5 * (v1/v2 + v2/v1 - 2.0)
        mean_diff_term = 0.5 * (mu1 - mu2)**2 * (1.0/v1 + 1.0/v2)
        jeffreys = variance_ratio_term + mean_diff_term
        
        # Bhattacharyya Distance
        var_sum = v1 + v2
        bhattacharyya = (
            (mu1 - mu2)**2 / (4.0 * var_sum) +
            0.5 * np.log(var_sum / (2.0 * np.sqrt(v1 * v2)))
        )
        
        # G-JSD = (1/4)*Jeffreys - Bhattacharyya
        gjsd = 0.25 * jeffreys - bhattacharyya
        gjsd = max(0.0, gjsd)
        
        return gjsd if np.isfinite(gjsd) else 0.0
    
    @staticmethod
    def _gaussian_gjsd_asymmetric(gene_name: str, mu1: float, v1: float, 
                                   mu2: float, v2: float) -> Tuple:
        """
        Asymmetric G-JSD with full directional decomposition.
        
        Decomposes JSD into KL components to determine direction:
        - KL(P1 || P2): surprise of seeing P2 when expecting P1
        - KL(P2 || P1): surprise of seeing P1 when expecting P2
        
        Direction = KL(P1||P2) - KL(P2||P1)
        Positive = feature specific to P1 (target)
        Negative = feature specific to P2 (other)
        
        Returns
        -------
        Tuple with: gene, magnitude, direction, signed_specificity,
                    kl_1_2, kl_2_1, specificity_target, specificity_other
        """
        # Compute symmetric magnitude first
        magnitude = BulkGJSD._gaussian_gjsd_symmetric(mu1, v1, mu2, v2)
        
        # KL divergences for Gaussians (closed form)
        # KL(P1 || P2) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        kl_1_2 = (np.log(np.sqrt(v2/v1)) + 
                  (v1 + (mu1 - mu2)**2) / (2*v2) - 0.5)
        
        kl_2_1 = (np.log(np.sqrt(v1/v2)) + 
                  (v2 + (mu2 - mu1)**2) / (2*v1) - 0.5)
        
        # Direction score: positive = target-specific, negative = other-specific
        direction = kl_1_2 - kl_2_1
        
        # Signed specificity combines magnitude with direction
        signed_specificity = np.sign(direction) * magnitude
        
        # Normalized specificity scores for each population
        # Using softmax-like normalization of KL terms
        kl_sum = kl_1_2 + kl_2_1 + 1e-12
        specificity_target = kl_1_2 / kl_sum  # Higher when target is "outlier"
        specificity_other = kl_2_1 / kl_sum   # Higher when other is "outlier"
        
        # Ensure finite values
        results = [magnitude, direction, signed_specificity, 
                   kl_1_2, kl_2_1, specificity_target, specificity_other]
        results = [x if np.isfinite(x) else 0.0 for x in results]
        
        return (gene_name, *results)
    
    @staticmethod
    def _kl_directional(gene_name: str, mu1: float, v1: float,
                        mu2: float, v2: float) -> Tuple:
        """
        Pure KL-based directional score without G-JSD magnitude.
        
        Useful when you want raw KL asymmetry without the Bhattacharyya correction.
        """
        # KL divergences
        kl_1_2 = (np.log(np.sqrt(v2/v1)) + 
                  (v1 + (mu1 - mu2)**2) / (2*v2) - 0.5)
        
        kl_2_1 = (np.log(np.sqrt(v1/v2)) + 
                  (v2 + (mu2 - mu1)**2) / (2*v1) - 0.5)
        
        # Jeffreys (symmetric magnitude)
        jeffreys = kl_1_2 + kl_2_1
        
        # Direction
        direction = kl_1_2 - kl_2_1
        signed_specificity = np.sign(direction) * jeffreys
        
        kl_sum = kl_1_2 + kl_2_1 + 1e-12
        specificity_target = kl_1_2 / kl_sum
        specificity_other = kl_2_1 / kl_sum
        
        results = [jeffreys, direction, signed_specificity,
                   kl_1_2, kl_2_1, specificity_target, specificity_other]
        results = [x if np.isfinite(x) else 0.0 for x in results]
        
        return (gene_name, *results)
    
    @staticmethod
    def _gaussian_gjsd_skew(gene_name: str, mu1: float, v1: float,
                            mu2: float, v2: float, alpha: float) -> Tuple:
        """
        Skewed G-JSD with tunable alpha parameter.
        
        Uses weighted KL terms:
        JSD_alpha = alpha * KL(P1||M_alpha) + (1-alpha) * KL(P2||M_alpha)
        
        where M_alpha is the alpha-weighted mixture.
        
        alpha = 0.5: symmetric (standard G-JSD)
        alpha > 0.5: biased toward detecting target-specific features
        alpha < 0.5: biased toward detecting other-specific features
        
        Returns both the alpha-skewed score and its complement for comparison.
        """
        # For Gaussians, we approximate using weighted KL decomposition
        # of the Jeffreys component
        
        # KL divergences
        kl_1_2 = (np.log(np.sqrt(v2/v1)) + 
                  (v1 + (mu1 - mu2)**2) / (2*v2) - 0.5)
        
        kl_2_1 = (np.log(np.sqrt(v1/v2)) + 
                  (v2 + (mu2 - mu1)**2) / (2*v1) - 0.5)
        
        # Bhattacharyya (symmetric, unchanged)
        var_sum = v1 + v2
        bhattacharyya = (
            (mu1 - mu2)**2 / (4.0 * var_sum) +
            0.5 * np.log(var_sum / (2.0 * np.sqrt(v1 * v2)))
        )
        
        # Skewed Jeffreys: weight the KL terms
        jeffreys_alpha = alpha * kl_1_2 + (1 - alpha) * kl_2_1
        jeffreys_1_minus_alpha = (1 - alpha) * kl_1_2 + alpha * kl_2_1
        
        # Skewed G-JSD approximation
        # Note: This is an approximation; true skew-G-JSD would use skewed geometric mixture
        gjsd_alpha = 0.5 * jeffreys_alpha - bhattacharyya
        gjsd_1_minus_alpha = 0.5 * jeffreys_1_minus_alpha - bhattacharyya
        
        gjsd_alpha = max(0.0, gjsd_alpha)
        gjsd_1_minus_alpha = max(0.0, gjsd_1_minus_alpha)
        
        # Symmetric version for reference
        gjsd_symmetric = BulkGJSD._gaussian_gjsd_symmetric(mu1, v1, mu2, v2)
        
        results = [gjsd_symmetric, gjsd_alpha, gjsd_1_minus_alpha]
        results = [x if np.isfinite(x) else 0.0 for x in results]
        
        return (gene_name, *results)
    
    @staticmethod
    def _compute_discrete_single(gene_name: str, target_expr: np.ndarray,
                                  n_target: int, mu_other: float,
                                  method: str) -> Tuple[str, float]:
        """Legacy discrete per-sample methods."""
        eps = 1e-12
        target_expr = np.asarray(target_expr, dtype=float)
        
        score_sum = 0.0
        for i in range(n_target):
            x = float(target_expr[i])
            denom = x + mu_other
            
            if denom == 0.0:
                continue
            
            p0, p1 = x / denom, mu_other / denom
            q0, q1 = 1.0, 0.0
            
            p0_safe, p1_safe = max(p0, eps), max(p1, eps)
            q0_safe, q1_safe = max(q0, eps), max(q1, eps)
            
            if method == 'jsd':
                m0, m1 = 0.5 * (p0 + q0), 0.5 * (p1 + q1)
                m0, m1 = max(m0, eps), max(m1, eps)
                kl_pm = p0 * np.log2(p0_safe / m0) + (p1 * np.log2(p1_safe / m1) if p1 > 0 else 0)
                kl_qm = q0 * np.log2(q0_safe / m0)
                score = 0.5 * (kl_pm + kl_qm)
                
            elif method == 'geometric_jsd':
                log_m0 = 0.5 * np.log(p0_safe) + 0.5 * np.log(q0_safe)
                log_m1 = 0.5 * np.log(p1_safe) + 0.5 * np.log(q1_safe)
                m0, m1 = np.exp(log_m0), np.exp(log_m1)
                m_sum = m0 + m1
                m0, m1 = m0 / m_sum, m1 / m_sum
                m0, m1 = max(m0, eps), max(m1, eps)
                
                kl_pm = p0 * np.log2(p0_safe / m0) + (p1 * np.log2(p1_safe / m1) if p1 > eps else 0)
                kl_qm = q0 * np.log2(q0_safe / m0)
                score = 0.5 * (kl_pm + kl_qm)
                
            elif method == 'bhattacharyya':
                bc = np.sqrt(p0_safe * q0_safe) + np.sqrt(p1_safe * q1_safe)
                score = max(0.0, 1.0 - bc)
                
            elif method == 'hellinger':
                h = np.sqrt((np.sqrt(p0_safe) - np.sqrt(q0_safe))**2 + 
                           (np.sqrt(p1_safe) - np.sqrt(q1_safe))**2)
                score = h / np.sqrt(2.0)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if np.isfinite(score):
                score_sum += max(0.0, score)
        
        return (gene_name, float(score_sum))
    
    # =========================================================================
    # Convenience methods for bidirectional analysis
    # =========================================================================
    
    def compare_bidirectional(self, group1_samples: List[str], group2_samples: List[str],
                               method: str = 'bidirectional_gjsd') -> Dict[str, pd.DataFrame]:
        """
        Perform bidirectional comparison returning features specific to each group.
        
        Parameters
        ----------
        group1_samples : List[str]
            First group samples
        group2_samples : List[str]
            Second group samples
        method : str
            Method to use (default: bidirectional_gjsd)
            
        Returns
        -------
        Dict with keys:
            'all': Full results DataFrame
            'group1_specific': Features specific to group1 (positive direction)
            'group2_specific': Features specific to group2 (negative direction)
        """
        # Run comparison with group1 as target
        results = self.compare(group1_samples, group2_samples, method=method)
        
        # Split by direction
        group1_specific = results[results['direction'] > 0].copy()
        group2_specific = results[results['direction'] < 0].copy()
        
        # Sort each by their respective specificity
        group1_specific = group1_specific.sort_values('signed_specificity', ascending=False)
        group2_specific = group2_specific.sort_values('signed_specificity', ascending=True)
        
        return {
            'all': results,
            'group1_specific': group1_specific,
            'group2_specific': group2_specific
        }


# =============================================================================
# Utility functions for downstream analysis
# =============================================================================

def filter_bidirectional_results(results_df: pd.DataFrame, 
                                  min_magnitude: float = 0.1,
                                  min_abs_direction: float = 0.5) -> Dict[str, pd.DataFrame]:
    """
    Filter bidirectional results by magnitude and direction thresholds.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from BulkGJSD with asymmetric method
    min_magnitude : float
        Minimum gjsd_score to include
    min_abs_direction : float
        Minimum absolute direction score for confident assignment
        
    Returns
    -------
    Dict with 'target_specific', 'other_specific', 'ambiguous' DataFrames
    """
    # Apply magnitude filter
    filtered = results_df[results_df['gjsd_score'] >= min_magnitude].copy()
    
    # Split by direction confidence
    target_specific = filtered[filtered['direction'] >= min_abs_direction]
    other_specific = filtered[filtered['direction'] <= -min_abs_direction]
    ambiguous = filtered[filtered['direction'].abs() < min_abs_direction]
    
    return {
        'target_specific': target_specific.sort_values('direction', ascending=False),
        'other_specific': other_specific.sort_values('direction', ascending=True),
        'ambiguous': ambiguous.sort_values('gjsd_score', ascending=False)
    }


def rank_by_population_specificity(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add population-specific ranks to results.
    
    Useful for comparing rankings: genes that rank highly for one population
    but lowly for the other are the most specific.
    """
    df = results_df.copy()
    
    # Rank by each specificity score
    df['rank_target'] = df['specificity_target'].rank(ascending=False)
    df['rank_other'] = df['specificity_other'].rank(ascending=False)
    
    # Differential rank: large positive = target-specific, large negative = other-specific  
    df['rank_diff'] = df['rank_other'] - df['rank_target']
    
    return df
