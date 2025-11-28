# %% Simple gJSD class for bulk RNA-seq
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


# %% Fixed BulkGJSD class
class BulkGJSD:
    def __init__(self, counts_df: pd.DataFrame, n_processes: int = 4):
        self.counts = counts_df
        self.n_processes = n_processes
    
    def compare(self, target_samples: List[str], other_samples: List[str], 
                method: str = 'geometric_jsd') -> pd.DataFrame:
        
        X_target = self.counts[target_samples].values.T
        X_other = self.counts[other_samples].values.T
        
        n_target = len(target_samples)
        n_other = len(other_samples)
        
        mu_other = X_other.mean(axis=0)
        var_other = X_other.var(axis=0, ddof=1) if n_other > 1 else mu_other
        
        gene_data_list = []
        for i, gene in enumerate(self.counts.index):
            target_expr = X_target[:, i]
            if method in ('gaussian_gjsd', 'extended_gjsd'):
                gene_data_list.append((gene, target_expr, n_target, mu_other[i], var_other[i]))
            else:
                gene_data_list.append((gene, target_expr, n_target, mu_other[i]))
        
        compute_func = partial(self._compute_gjsd_single, method=method)
        
        with Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(compute_func, gene_data_list, chunksize=100),
                total=len(gene_data_list),
                desc=f"gJSD ({method})"
            ))
        
        df = pd.DataFrame(results, columns=['gene', 'gjsd_score'])
        df['mean_target'] = self.counts[target_samples].mean(axis=1).values
        df['mean_other'] = self.counts[other_samples].mean(axis=1).values
        df['log2FC'] = np.log2((df['mean_target'] + 1) / (df['mean_other'] + 1))
        df = df.set_index('gene')
        
        # Lower score = more specific for all methods
        df = df.sort_values('gjsd_score', ascending=True)
        
        return df
    
    @staticmethod
    def _compute_gjsd_single(gene_data: Tuple, method: str = 'geometric_jsd') -> Tuple[str, float]:
        
        eps = 1e-12
        
        if len(gene_data) == 4:
            gene_name, target_expr, n_target, mu_other = gene_data
            var_other_provided = None
        else:
            gene_name, target_expr, n_target, mu_other, var_other_provided = gene_data
        
        target_expr = np.asarray(target_expr, dtype=float)
        if len(target_expr) != n_target:
            n_target = len(target_expr)
        
        score_sum = 0.0
        
        # =====================================================================
        # GAUSSIAN CLOSED-FORM METHODS (Nielsen 2025)
        # =====================================================================
        if method in ('gaussian_gjsd', 'extended_gjsd'):
            
            mu_target = float(np.mean(target_expr))
            
            if n_target >= 2:
                var_target = float(np.var(target_expr, ddof=1))
            else:
                var_target = max(mu_target, eps)
            
            var_target = max(var_target, eps)
            
            mu_other_val = float(mu_other)
            
            if var_other_provided is not None:
                var_other = max(float(var_other_provided), eps)
            else:
                if mu_target > eps:
                    cv_target = np.sqrt(var_target) / mu_target
                    var_other = (cv_target * max(mu_other_val, eps)) ** 2
                    var_other = max(var_other, eps)
                else:
                    var_other = var_target
            
            v1, v2 = var_target, var_other
            mu1, mu2 = mu_target, mu_other_val
            
            # Both means near zero = not expressed anywhere
            if abs(mu1) < eps and abs(mu2) < eps:
                return gene_name, 0.0
            
            # Jeffreys Divergence
            variance_ratio_term = 0.5 * (v1/v2 + v2/v1 - 2.0)
            mean_diff_term = 0.5 * (mu1 - mu2)**2 * (1.0/v1 + 1.0/v2)
            jeffreys = variance_ratio_term + mean_diff_term
            
            # Bhattacharyya Distance
            var_sum = v1 + v2
            bhattacharyya = (
                (mu1 - mu2)**2 / (4.0 * var_sum) +
                0.5 * np.log(var_sum / (2.0 * np.sqrt(v1 * v2)))
            )
            
            gjsd = 0.5 * jeffreys - bhattacharyya
            gjsd = max(0.0, gjsd)
            
            if not np.isfinite(gjsd):
                gjsd = 0.0
            
            # Scale by n_target, return 1/gjsd so lower = more specific
            if gjsd > eps:
                score_sum = (1.0 / gjsd) * n_target
            else:
                score_sum = float('inf')
            
            return gene_name, float(score_sum)
        
        # =====================================================================
        # DISCRETE PER-SAMPLE METHODS
        # =====================================================================
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
        
        return gene_name, float(score_sum)