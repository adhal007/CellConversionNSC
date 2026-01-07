import pandas as pd
import numpy as np
import pybedtools
from tqdm import tqdm

class MasterRegulatorAnalysis:
    """
    3-Tier Master Regulator Identification with Softmax Probability Scoring
    """
    
    def __init__(self, deseq_results, chip_edges, enhancer_db, dar_peaks, 
                syngo_genes, syngo_annotations, log2fc_threshold=1.0, padj_threshold=0.05):
        """
        Initialize with all required data
        """
        # UPPERCASE ALL GENE SYMBOLS AT THE START
        self.deseq_results = deseq_results.copy()
        self.deseq_results['symbol'] = self.deseq_results['symbol'].str.upper()
        
        self.chip_edges = chip_edges.copy()
        self.chip_edges['TF'] = self.chip_edges['TF'].str.upper()
        self.chip_edges['Target'] = self.chip_edges['Target'].str.upper()
        
        self.enhancer_db = enhancer_db.copy()
        self.enhancer_db['symbol'] = self.enhancer_db['symbol'].str.upper()
        
        # FILTER: Only keep enhancers for genes in our RNA-seq data
        valid_genes = set(self.deseq_results['symbol'].unique())
        enhancer_db_filtered = self.enhancer_db[self.enhancer_db['symbol'].isin(valid_genes)]
        
        print(f"Filtered enhancers: {len(self.enhancer_db)} → {len(enhancer_db_filtered)} (genes in RNA-seq)")
        self.enhancer_db = enhancer_db_filtered
        
        self.dar_peaks = dar_peaks.copy()
        self.syngo_genes = syngo_genes
        self.syngo_annotations = syngo_annotations
        
        self.log2fc_threshold = log2fc_threshold
        self.padj_threshold = padj_threshold
        
        # Extract SynGO gene list (already uppercase)
        self.syngo_gene_set = self._extract_syngo_genes()
        
        # Store intermediate results
        self.tier1_tfs = None
        self.tier2_results = None
        self.tier3_results = None
        self.final_results = None
        self.accessible_genes_e14 = None
        self.accessible_genes_e18 = None
        
        print(f"Initialized MasterRegulatorAnalysis")
        print(f"  DE genes: {len(self.deseq_results)}")
        print(f"  ChIP edges: {len(self.chip_edges)}")
        print(f"  Enhancers: {len(self.enhancer_db)}")
        print(f"  DA peaks: {len(self.dar_peaks)}")
        print(f"  SynGO functional genes: {len(self.syngo_gene_set)}")
    
    def _extract_syngo_genes(self):
        """Extract gene symbols from SynGO"""
        if 'hgnc_symbol' in self.syngo_genes.columns:
            genes = set(self.syngo_genes['hgnc_symbol'].str.upper().dropna().unique())
        elif 'gene_symbol' in self.syngo_genes.columns:
            genes = set(self.syngo_genes['gene_symbol'].str.upper().dropna().unique())
        elif 'symbol' in self.syngo_genes.columns:
            genes = set(self.syngo_genes['symbol'].str.upper().dropna().unique())
        else:
            genes = set(self.syngo_genes.iloc[:, 0].str.upper().dropna().unique())
        
        return genes
    
    def tier1_filter_tfs(self, direction='E14'):
        """
        Tier 1: Filter DE TFs
        """
        print(f"\n{'='*80}")
        print(f"TIER 1: Filtering {direction}-enriched TFs")
        print(f"{'='*80}")
        
        if direction == 'E14':
            tfs = self.deseq_results[
                (self.deseq_results['is_TF']) & 
                (self.deseq_results['log2FoldChange'] > self.log2fc_threshold) &
                (self.deseq_results['padj'] < self.padj_threshold)
            ].copy()
        else:  # E18
            tfs = self.deseq_results[
                (self.deseq_results['is_TF']) & 
                (self.deseq_results['log2FoldChange'] < -self.log2fc_threshold) &
                (self.deseq_results['padj'] < self.padj_threshold)
            ].copy()
        
        # Store
        self.tier1_tfs = tfs.copy()
        
        print(f"Found {len(tfs)} {direction}-enriched TFs")
        
        # Show top 5
        tfs_sorted = tfs.copy()
        tfs_sorted['abs_log2fc'] = tfs_sorted['log2FoldChange'].abs()
        top5 = tfs_sorted.nlargest(5, 'abs_log2fc')['symbol'].tolist()
        print(f"Top 5 by expression: {top5}")
        
        return tfs
    
    def tier2_target_consistency(self, tf_list, direction='E14'):
        """
        Tier 2: Calculate target consistency for each TF
        """
        print(f"\n{'='*80}")
        print(f"TIER 2: Calculating target consistency")
        print(f"{'='*80}")
        
        # Uppercase gene symbols
        self.chip_edges['TF'] = self.chip_edges['TF'].str.upper()
        self.chip_edges['Target'] = self.chip_edges['Target'].str.upper()
        self.deseq_results['symbol_upper'] = self.deseq_results['symbol'].str.upper()
        
        results = []
        
        for idx, tf_row in tqdm(tf_list.iterrows(), total=len(tf_list), desc="Processing TFs"):
            tf_name = tf_row['symbol'].upper()
            tf_log2fc = tf_row['log2FoldChange']
            
            # Get targets from ChIP
            targets = self.chip_edges[self.chip_edges['TF'] == tf_name]['Target'].unique()
            
            if len(targets) == 0:
                continue
            
            # Filter to functional targets (SynGO)
            functional_targets = [t for t in targets if t in self.syngo_gene_set]
            
            if len(functional_targets) == 0:
                continue
            
            # Check which functional targets are DE in same direction
            target_de = self.deseq_results[
                self.deseq_results['symbol_upper'].isin(functional_targets)
            ]
            
            if direction == 'E14':
                consistent_targets = target_de[
                    (target_de['log2FoldChange'] > 0) &
                    (target_de['padj'] < self.padj_threshold)
                ]
            else:  # E18
                consistent_targets = target_de[
                    (target_de['log2FoldChange'] < 0) &
                    (target_de['padj'] < self.padj_threshold)
                ]
            
            # Calculate consistency score
            consistency_score = len(consistent_targets) / len(functional_targets) if len(functional_targets) > 0 else 0
            
            results.append({
                'TF': tf_row['symbol'],
                'log2FC': tf_log2fc,
                'padj': tf_row['padj'],
                'n_total_targets': len(targets),
                'n_functional_targets': len(functional_targets),
                'n_de_targets': len(consistent_targets),  # ← Fixed: use n_de_targets
                'n_consistent_targets': len(consistent_targets),  # Keep for backward compatibility
                'target_consistency': consistency_score,
                'functional_targets': list(functional_targets),
                'consistent_targets': list(consistent_targets['symbol'].str.upper().values)
            })
                    
        results_df = pd.DataFrame(results)
        
        # Store
        self.tier2_results = results_df.copy()
        
        print(f"\nProcessed {len(results_df)} TFs with functional targets")
        print(f"Mean target consistency: {results_df['target_consistency'].mean():.2f}")
        
        return results_df
    
    def tier3_chromatin_accessibility(self, tf_scores_df, direction='E14'):
        """
        Tier 3: Calculate chromatin accessibility score
        """
        print(f"\n{'='*80}")
        print(f"TIER 3: Calculating chromatin accessibility")
        print(f"{'='*80}")
        
        # Separate DA peaks by direction
        if direction == 'E14':
            da_peaks_direction = self.dar_peaks[self.dar_peaks['Fold'] > 0].copy()
        else:
            da_peaks_direction = self.dar_peaks[self.dar_peaks['Fold'] < 0].copy()
        
        print(f"Using {len(da_peaks_direction)} {direction}-enriched DA peaks")
        
        # Create BED objects for overlap
        da_peaks_bed = pybedtools.BedTool.from_dataframe(
            da_peaks_direction[['seqnames', 'start', 'end']].rename(columns={'seqnames': 'chrom'})
        )
        
        enhancer_bed = pybedtools.BedTool.from_dataframe(
            self.enhancer_db[['chr', 'start', 'end', 'symbol']].rename(columns={'chr': 'chrom', 'symbol': 'name'})
        )
        
        # Find enhancers overlapping DA peaks
        overlaps = enhancer_bed.intersect(da_peaks_bed, wa=True, u=True)
        
        # Get genes with accessible enhancers
        accessible_genes = set()
        for interval in overlaps:
            accessible_genes.add(interval.name.upper())
        
        # Store accessible genes
        if direction == 'E14':
            self.accessible_genes_e14 = accessible_genes
        else:
            self.accessible_genes_e18 = accessible_genes
        
        print(f"Found {len(accessible_genes)} genes with {direction}-accessible enhancers")
        
        # Calculate accessibility score for each TF
        accessibility_scores = []

        for idx, row in tf_scores_df.iterrows():
            functional_targets = row['functional_targets']
            consistent_targets = row['consistent_targets']  # These are DE targets
            
            # NEW: Calculate functional AND DE
            functional_targets_upper = set([t.upper() for t in functional_targets])
            consistent_targets_upper = set(consistent_targets)  # Already uppercase from fix
            functional_and_de = functional_targets_upper & consistent_targets_upper
            n_functional_and_de = len(functional_and_de)
            
            # Check how many have accessible enhancers
            n_accessible = sum(1 for t in functional_and_de if t in accessible_genes)
            
            accessibility_score = n_accessible / len(functional_targets) if len(functional_targets) > 0 else 0
            
            accessibility_scores.append({
                'TF': row['TF'],
                'n_functional_and_de': n_functional_and_de,  # ← ADD THIS
                'n_accessible_targets': n_accessible,
                'n_accessible': n_accessible,  # ← ADD THIS (alias)
                'chromatin_accessibility': accessibility_score,
                'accessible_targets': [t for t in functional_and_de if t in accessible_genes]
            })
        
        accessibility_df = pd.DataFrame(accessibility_scores)
        
        # Merge with tier2 results
        final_df = tf_scores_df.merge(accessibility_df, on='TF', how='left')
        final_df['chromatin_accessibility'].fillna(0, inplace=True)
        
        # Store
        self.tier3_results = final_df.copy()
        
        print(f"Mean chromatin accessibility: {final_df['chromatin_accessibility'].mean():.2f}")
        
        return final_df
    
    def calculate_mr_probability(self, tf_scores_df, temperature=1.0, weights=None):
        """
        Calculate Master Regulator probability using softmax
        
        Parameters:
        -----------
        tf_scores_df : DataFrame with all three tier scores
        temperature : float, controls sharpness
        weights : dict with keys 'expression', 'consistency', 'chromatin' (default: equal weights)
        
        Returns:
        --------
        DataFrame with MR_probability
        """
        print(f"\n{'='*80}")
        print(f"CALCULATING MASTER REGULATOR PROBABILITIES (T={temperature})")
        print(f"{'='*80}")
        
        # Default equal weights
        if weights is None:
            weights = {'expression': 1.0, 'consistency': 1.0, 'chromatin': 1.0}
        
        # Normalize each feature to 0-1 scale INDEPENDENTLY
        tf_scores_df['expression_norm'] = (
            np.abs(tf_scores_df['log2FC']) / 
            np.abs(tf_scores_df['log2FC']).max()
        )
        
        # target_consistency is already 0-1
        # chromatin_accessibility is already 0-1
        
        # SCALE features to have comparable ranges
        # Problem: chromatin_accessibility is often 0.01-0.05, gets drowned out
        # Solution: Scale each to have similar std or range
        
        expression_scaled = tf_scores_df['expression_norm'] * weights['expression']
        consistency_scaled = tf_scores_df['target_consistency'] * weights['consistency']
        chromatin_scaled = tf_scores_df['chromatin_accessibility'] * weights['chromatin']
        
        # Combined raw score
        tf_scores_df['raw_score'] = (
            expression_scaled + 
            consistency_scaled + 
            chromatin_scaled
        )
        
        print(f"\nFeature statistics:")
        print(f"  Expression (scaled): mean={expression_scaled.mean():.3f}, std={expression_scaled.std():.3f}")
        print(f"  Consistency (scaled): mean={consistency_scaled.mean():.3f}, std={consistency_scaled.std():.3f}")
        print(f"  Chromatin (scaled): mean={chromatin_scaled.mean():.3f}, std={chromatin_scaled.std():.3f}")
        
        # Softmax transformation
        exp_scores = np.exp(tf_scores_df['raw_score'] / temperature)
        tf_scores_df['MR_probability'] = exp_scores / exp_scores.sum()
        
        # Sort by probability
        tf_scores_df = tf_scores_df.sort_values('MR_probability', ascending=False)
        
        # Add rank
        tf_scores_df['rank'] = range(1, len(tf_scores_df) + 1)
        
        # Store
        self.final_results = tf_scores_df.copy()
        
        print(f"\nTop 10 Master Regulators:")
        print(tf_scores_df[['TF', 'log2FC', 'target_consistency', 'chromatin_accessibility', 
                            'n_accessible_targets', 'MR_probability', 'rank']].head(10))
        
        return tf_scores_df
    
    def calculate_mr_probability_cascade(self, tf_scores_df, temperature=1.0):
        """
        Calculate Master Regulator probability as cascade of conditional probabilities
        
        P(MR) = P(TF_DE) × P(targets_DE | TF_DE) × P(functional | DE) × P(accessible | functional)
        """
        print(f"\n{'='*80}")
        print(f"CALCULATING MASTER REGULATOR PROBABILITIES (Cascade Model)")
        print(f"{'='*80}")
        
        results = []
        
        for idx, row in tf_scores_df.iterrows():
            tf_name = row['TF']
            
            # Layer 1: TF Expression Probability
            # Use sigmoid to convert log2FC to probability
            log2fc_abs = np.abs(row['log2FC'])
            p_expression = 1 / (1 + np.exp(-log2fc_abs + 1))  # sigmoid shifted
            
            # Layer 2: P(targets DE | TF DE)
            n_total_targets = row['n_total_targets']
            n_de_targets = row['n_consistent_targets']  # DE in same direction
            p_targets_de = n_de_targets / n_total_targets if n_total_targets > 0 else 0
            
            # Layer 3: P(functional | targets DE)
            # What proportion of ALL ChIP targets are functional?
            n_functional_targets = row['n_functional_targets']
            p_functional = n_functional_targets / n_total_targets if n_total_targets > 0 else 0
            
            # Layer 4: P(accessible | functional & DE)
            # Of the functional targets that ARE DE, how many have DA enhancers?
            functional_targets = set(row['functional_targets'])
            consistent_targets = set(row['consistent_targets'])
            accessible_targets = set(row['accessible_targets'])
            
            # Intersection: functional AND DE
            functional_and_de = functional_targets & consistent_targets
            n_functional_and_de = len(functional_and_de)
            
            # Of those, how many have accessible enhancers?
            functional_de_accessible = functional_and_de & accessible_targets
            n_accessible_given_functional_de = len(functional_de_accessible)
            
            p_accessible = (n_accessible_given_functional_de / n_functional_and_de 
                        if n_functional_and_de > 0 else 0)
            
            # PRODUCT OF CONDITIONAL PROBABILITIES
            p_mr_raw = p_expression * p_targets_de * p_functional * p_accessible
            
            results.append({
                'TF': tf_name,
                'log2FC': row['log2FC'],
                'padj': row['padj'],
                'n_total_targets': n_total_targets,
                'n_de_targets': n_de_targets,
                'n_functional_targets': n_functional_targets,
                'n_functional_and_de': n_functional_and_de,
                'n_accessible_given_functional_de': n_accessible_given_functional_de,
                'p_expression': p_expression,
                'p_targets_de': p_targets_de,
                'p_functional': p_functional,
                'p_accessible': p_accessible,
                'p_mr_raw': p_mr_raw,
                'functional_targets': row['functional_targets'],
                'consistent_targets': row['consistent_targets'],
                'accessible_targets': row['accessible_targets']
            })
        
        results_df = pd.DataFrame(results)
        
        # Normalize raw probabilities using softmax for ranking
        exp_scores = np.exp(results_df['p_mr_raw'] / temperature)
        results_df['MR_probability'] = exp_scores / exp_scores.sum()
        
        # Sort by probability
        results_df = results_df.sort_values('MR_probability', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Store
        self.final_results = results_df.copy()
        
        print(f"\nProbability Layer Statistics:")
        print(f"  P(expression): mean={results_df['p_expression'].mean():.3f}")
        print(f"  P(targets DE): mean={results_df['p_targets_de'].mean():.3f}")
        print(f"  P(functional): mean={results_df['p_functional'].mean():.3f}")
        print(f"  P(accessible): mean={results_df['p_accessible'].mean():.3f}")
        print(f"  P(MR raw): mean={results_df['p_mr_raw'].mean():.3f}")
        
        print(f"\nTop 10 Master Regulators:")
        display_cols = ['TF', 'log2FC', 'n_de_targets', 'n_functional_and_de', 
                    'n_accessible_given_functional_de', 'p_mr_raw', 'MR_probability', 'rank']
        print(results_df[display_cols].head(10))
        
        return results_df

    def calculate_mr_probability_with_saturation(self, tf_scores_df, k_func=10, k_enh=15, k_da=15, temperature=1.0):
        """
        Calculate Master Regulator probability using 4-component exponential saturation
        
        P = E_TF × E_func × E_enh × E_DA
        
        Where:
        - E_TF: TF differential expression (sigmoid)
        - E_func: Fraction of ChIP targets that are functional (SynGO)
        - E_enh: Fraction of functional targets that are DE
        - E_DA: Fraction of functional+DE targets with accessible enhancers
        """
        print(f"\n{'='*80}")
        print(f"CALCULATING MR PROBABILITIES (4-Component Saturation Model)")
        print(f"{'='*80}")
        print(f"Parameters: k_func={k_func}, k_enh={k_enh}, k_da={k_da}")
        
        results = []
        
        for idx, row in tf_scores_df.iterrows():
            tf_name = row['TF']
            
            # Extract counts
            n_total = row['n_total_targets']
            n_functional = row['n_functional_targets']
            n_functional_and_de = row['n_functional_and_de']
            n_accessible = row['n_accessible']
            
            # === COMPONENT 1: E_TF - TF Expression ===
            log2fc = row['log2FC']
            E_TF = 1 / (1 + np.exp(-(np.abs(log2fc) - 1) / 0.5))
            
            # === COMPONENT 2: E_func - Functional Enrichment ===
            # What fraction of ChIP targets are functional (in SynGO)?
            f_func = n_functional / n_total if n_total > 0 else 0
            E_func = 1 - np.exp(-k_func * f_func)
            
            # === COMPONENT 3: E_enh - DE Enrichment among Functional ===
            # Of functional targets, what fraction are DE?
            f_de_func = n_functional_and_de / n_functional if n_functional > 0 else 0
            E_enh = 1 - np.exp(-k_enh * f_de_func)
            
            # === COMPONENT 4: E_DA - Chromatin Accessibility ===
            # Of functional+DE targets, what fraction have accessible enhancers?
            f_da = n_accessible / n_functional_and_de if n_functional_and_de > 0 else 0
            E_DA = 1 - np.exp(-k_da * f_da)
            
            # === MULTIPLICATIVE MODEL: All 4 gates must pass ===
            P_raw = E_TF * E_func * E_enh * E_DA
            
            results.append({
                'TF': tf_name,
                'log2FC': log2fc,
                'padj': row['padj'],
                'n_total_targets': n_total,
                'n_functional_targets': n_functional,
                'n_functional_and_de': n_functional_and_de,
                'n_accessible': n_accessible,
                'f_func': f_func,
                'f_de_func': f_de_func,
                'f_da': f_da,
                'E_TF': E_TF,
                'E_func': E_func,
                'E_enh': E_enh,
                'E_DA': E_DA,
                'P_raw': P_raw,
                'functional_targets': row['functional_targets'],
                'consistent_targets': row['consistent_targets'],
                'accessible_targets': row['accessible_targets']
            })
        
        results_df = pd.DataFrame(results)
        
        # Normalize by max
        max_p = results_df['P_raw'].max()
        results_df['MR_probability'] = results_df['P_raw'] / max_p if max_p > 0 else 0
        
        # Sort and rank
        results_df = results_df.sort_values('MR_probability', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        self.final_results = results_df.copy()
        
        # Summary
        print(f"\n4-Component Statistics:")
        print(f"  E_TF (expression):      mean={results_df['E_TF'].mean():.3f}, median={results_df['E_TF'].median():.3f}")
        print(f"  E_func (functional):    mean={results_df['E_func'].mean():.3f}, median={results_df['E_func'].median():.3f}")
        print(f"  E_enh (DE enrichment):  mean={results_df['E_enh'].mean():.3f}, median={results_df['E_enh'].median():.3f}")
        print(f"  E_DA (chromatin):       mean={results_df['E_DA'].mean():.3f}, median={results_df['E_DA'].median():.3f}")
        print(f"  P_raw (product):        mean={results_df['P_raw'].mean():.4f}, median={results_df['P_raw'].median():.4f}")
        
        print(f"\nTop 20 Master Regulators:")
        display_cols = ['TF', 'log2FC', 'n_functional_and_de', 'n_accessible', 
                    'E_TF', 'E_func', 'E_enh', 'E_DA', 'MR_probability', 'rank']
        print(results_df[display_cols].head(20))
        
        # Plot
        self._plot_component_contributions(results_df.head(20))
        
        return results_df


    def _plot_component_contributions(self, top_tfs):
        """Visualize 3 components + chromatin evidence"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 3 scoring components
        components = ['E_TF', 'E_func', 'E_enh']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        # Plot 1: Stacked bar
        x = range(len(top_tfs))
        bottom = np.zeros(len(top_tfs))
        
        for comp, color in zip(components, colors):
            values = top_tfs[comp].values
            axes[0].bar(x, values, bottom=bottom, label=comp, color=color, alpha=0.8)
            bottom += values
        
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(top_tfs['TF'].values, rotation=45, ha='right')
        axes[0].set_ylabel('Component Value')
        axes[0].set_title('3-Component Score')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Component heatmap
        comp_matrix = top_tfs[components].values.T
        im = axes[1].imshow(comp_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_xticks(range(len(top_tfs)))
        axes[1].set_xticklabels(top_tfs['TF'].values, rotation=45, ha='right')
        axes[1].set_yticks(range(len(components)))
        axes[1].set_yticklabels(components)
        axes[1].set_title('Component Heatmap')
        plt.colorbar(im, ax=axes[1])
        
        # Plot 3: Chromatin evidence
        chromatin_colors = ['#27ae60' if x else '#e74c3c' for x in top_tfs['has_chromatin_evidence']]
        axes[2].bar(x, top_tfs['chromatin_strength'], color=chromatin_colors, alpha=0.7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(top_tfs['TF'].values, rotation=45, ha='right')
        axes[2].set_ylabel('Chromatin Strength')
        axes[2].set_title('Chromatin Evidence (green=yes, red=no)')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def calculate_mr_probability_3component(self, tf_scores_df, k_func=10, k_enh=15, temperature=1.0):
        """
        Calculate Master Regulator probability using 3-component model (no chromatin)
        
        P = E_TF × E_func × E_enh
        
        Chromatin is added as annotation, not scoring component
        """
        print(f"\n{'='*80}")
        print(f"CALCULATING MR PROBABILITIES (3-Component Model)")
        print(f"{'='*80}")
        print(f"Parameters: k_func={k_func}, k_enh={k_enh}")
        
        results = []
        
        for idx, row in tf_scores_df.iterrows():
            tf_name = row['TF']
            
            # Extract counts
            n_total = row['n_total_targets']
            n_functional = row['n_functional_targets']
            n_functional_and_de = row['n_functional_and_de']
            n_accessible = row['n_accessible']
            
            # === COMPONENT 1: E_TF - TF Expression ===
            log2fc = row['log2FC']
            E_TF = 1 / (1 + np.exp(-(np.abs(log2fc) - 1) / 0.5))
            
            # === COMPONENT 2: E_func - Functional Enrichment ===
            f_func = n_functional / n_total if n_total > 0 else 0
            E_func = 1 - np.exp(-k_func * f_func)
            
            # === COMPONENT 3: E_enh - DE Enrichment among Functional ===
            f_de_func = n_functional_and_de / n_functional if n_functional > 0 else 0
            E_enh = 1 - np.exp(-k_enh * f_de_func)
            
            # === 3-COMPONENT SCORE ===
            P_raw = E_TF * E_func * E_enh
            
            # === CHROMATIN AS ANNOTATION (not in score) ===
            has_chromatin = n_accessible > 0
            chromatin_strength = n_accessible / n_functional_and_de if n_functional_and_de > 0 else 0
            
            results.append({
                'TF': tf_name,
                'log2FC': log2fc,
                'padj': row['padj'],
                'n_total_targets': n_total,
                'n_functional_targets': n_functional,
                'n_functional_and_de': n_functional_and_de,
                'n_accessible': n_accessible,
                'f_func': f_func,
                'f_de_func': f_de_func,
                'E_TF': E_TF,
                'E_func': E_func,
                'E_enh': E_enh,
                'P_raw': P_raw,
                'has_chromatin_evidence': has_chromatin,
                'chromatin_strength': chromatin_strength,
                'functional_targets': row['functional_targets'],
                'consistent_targets': row['consistent_targets'],
                'accessible_targets': row['accessible_targets']
            })
        
        results_df = pd.DataFrame(results)
        
        # Normalize by max
        max_p = results_df['P_raw'].max()
        results_df['MR_probability'] = results_df['P_raw'] / max_p if max_p > 0 else 0
        
        # Sort and rank
        results_df = results_df.sort_values('MR_probability', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Add category
        results_df['category'] = results_df.apply(
            lambda x: 'A_chromatin_evidence' if x['has_chromatin_evidence'] and x['chromatin_strength'] > 0.05
            else 'B_pioneer_candidate' if x['n_functional_and_de'] > 20 and not x['has_chromatin_evidence']
            else 'C_low_evidence',
            axis=1
        )
        
        self.final_results = results_df.copy()
        
        # Summary
        print(f"\n3-Component Statistics:")
        print(f"  E_TF (expression):      mean={results_df['E_TF'].mean():.3f}")
        print(f"  E_func (functional):    mean={results_df['E_func'].mean():.3f}")
        print(f"  E_enh (DE enrichment):  mean={results_df['E_enh'].mean():.3f}")
        print(f"  P_raw (product):        mean={results_df['P_raw'].mean():.4f}")
        
        print(f"\nChromatin Evidence:")
        print(f"  Category A (with chromatin): {(results_df['category']=='A_chromatin_evidence').sum()}")
        print(f"  Category B (pioneer candidate): {(results_df['category']=='B_pioneer_candidate').sum()}")
        print(f"  Category C (low evidence): {(results_df['category']=='C_low_evidence').sum()}")
        
        print(f"\nTop 20 Master Regulators:")
        display_cols = ['TF', 'log2FC', 'n_functional_and_de', 'n_accessible', 
                    'E_TF', 'E_func', 'E_enh', 'MR_probability', 'category', 'rank']
        print(results_df[display_cols].head(20))
        
        return results_df


    def run_full_analysis(self, direction='E14', k_func=10, k_enh=10, temperature=1.0):
        """
        Run 3-component MR analysis (chromatin as annotation)
        """
        print(f"\n{'#'*80}")
        print(f"RUNNING 3-COMPONENT MR ANALYSIS: {direction}")
        print(f"{'#'*80}")
        
        # Tier 1
        tfs = self.tier1_filter_tfs(direction=direction)
        
        # Tier 2
        tf_scores = self.tier2_target_consistency(tfs, direction=direction)
        
        # Tier 3
        tf_scores = self.tier3_chromatin_accessibility(tf_scores, direction=direction)
        
        # 3-component probability
        final_scores = self.calculate_mr_probability_3component(
            tf_scores, 
            k_func=k_func, 
            k_enh=k_enh
        )
        
        return final_scores