"""
Simple GRN pruner: Filter Cell Oracle PKN by TF list and DE genes.
Uses gene SYMBOLS throughout.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


class GRNPruner:
    """Prune Cell Oracle PKN to stage-specific TF networks."""
    
    def __init__(self, pkn_file, counts_file, metadata_file, deseq_file, tf_list,
                 lfc_threshold=1.0, corr_threshold=0.3, qval_threshold=0.05, padj_threshold=0.05):
        """
        Args:
            pkn_file: Cell Oracle parquet (peak_id, gene_short_name, TF columns)
            counts_file, metadata_file, deseq_file: Your data files
            tf_list: List of TF SYMBOLS to keep
            lfc_threshold: E14: LFC > threshold, E18: LFC < -threshold
            corr_threshold: Minimum correlation
            qval_threshold: Q-value threshold
        """
        self.tf_list = set(tf_list)
        self.lfc_threshold = lfc_threshold
        self.corr_threshold = corr_threshold
        self.qval_threshold = qval_threshold
        self.padj_threshold = padj_threshold
        print("Loading data...")
        self.pkn = pd.read_parquet(pkn_file)
        
        counts = pd.read_csv(counts_file, sep=';', header=0).iloc[:, 1:]
        counts = counts.set_index(counts.columns[0])
        
        metadata = pd.read_csv(metadata_file, sep=';')
        deseq = pd.read_csv(deseq_file, index_col=0)
        
        # Map Ensembl IDs to symbols in counts
        ensembl_to_symbol = dict(zip(deseq['gene_id'], deseq['symbol']))
        counts.index = counts.index.map(lambda x: ensembl_to_symbol.get(x, x))
        counts = counts[~counts.index.duplicated(keep='first')]
        self.counts = counts
        
        self.metadata = metadata
        self.deseq = deseq
        
        print(f"PKN: {self.pkn.shape[0]} peaks, TF list: {len(self.tf_list)} TFs")
    
    def _pkn_to_edges(self):
        """Convert PKN to TF-target edge list (gene SYMBOLS)."""
        gene_col = [c for c in self.pkn.columns if 'gene' in c.lower()][0]
        tf_cols = [c for c in self.pkn.columns if c not in ['peak_id', gene_col]]
        
        edges = []
        for _, row in self.pkn.iterrows():
            target = row[gene_col]
            for tf in tf_cols:
                if row[tf] > 0:
                    edges.append({'TF': tf, 'target': target})
        
        return pd.DataFrame(edges).drop_duplicates()
    
    def _filter_tf_tf(self, edges):
        """Keep only TF-TF edges from tf_list."""
        return edges[edges['TF'].isin(self.tf_list) & edges['target'].isin(self.tf_list)].copy()
    
    def _get_samples(self, stage):
        """Get CD133 samples."""
        mask = (self.metadata['Stage'] == stage) & (self.metadata['Marker'] == 'Progenitors')
        return self.metadata[mask]['SampleID'].tolist()
    
    def _normalize(self, df):
        """Log2(CPM+1)."""
        cpm = df.div(df.sum(axis=0), axis=1) * 1e6
        return np.log2(cpm + 1)
    
    def _compute_correlations(self, edges, stage):
        """Compute correlations using gene SYMBOLS."""
        samples = self._get_samples(stage)
        expr = self._normalize(self.counts[samples])
        
        results = []
        for _, row in edges.iterrows():
            tf, target = row['TF'], row['target']
            
            if tf not in expr.index or target not in expr.index or tf == target:
                continue
            
            corr, pval = spearmanr(expr.loc[tf], expr.loc[target])
            if not np.isnan(corr):
                results.append({'TF': tf, 'target': target, 'correlation': corr, 'pval': pval})
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            _, df['qval'], _, _ = multipletests(df['pval'], method='fdr_bh')
        return df
    
    def _filter_by_de(self, edges, stage):
        """Filter edges where TF or target is DE (using SYMBOLS)."""
        if stage == 'E14':
            de_symbols = set(self.deseq[(self.deseq['log2FoldChange'] > self.lfc_threshold) & (self.deseq['padj'] < self.padj_threshold)]['symbol'])
        else:
            de_symbols = set(self.deseq[(self.deseq['log2FoldChange'] < -self.lfc_threshold) & (self.deseq['padj'] < self.padj_threshold)]['symbol'])
        
        return edges[edges['TF'].isin(de_symbols) | edges['target'].isin(de_symbols)].copy()
    
    def build_networks(self):
        """Build E14 and E18 TF-TF networks. Returns (e14_grn, e18_grn)."""
        print("\n=== Converting PKN ===")
        edges = self._pkn_to_edges()
        print(f"Total edges: {len(edges)}")
        
        print("\n=== Filtering to TF-TF ===")
        edges = self._filter_tf_tf(edges)
        print(f"TF-TF edges: {len(edges)}")
        
        print("\n=== E14 Network ===")
        e14_edges = self._filter_by_de(edges.copy(), 'E14')
        print(f"After DE filter: {len(e14_edges)}")
        e14_corr = self._compute_correlations(e14_edges, 'E14')
        print(f"Correlations computed: {len(e14_corr)}")
        e14_grn = e14_corr[(e14_corr['correlation'] >= self.corr_threshold) & 
                           (e14_corr['qval'] < self.qval_threshold)].copy()
        print(f"Final E14: {len(e14_grn)} edges, {e14_grn['TF'].nunique()} TFs")
        
        print("\n=== E18 Network ===")
        e18_edges = self._filter_by_de(edges.copy(), 'E18')
        print(f"After DE filter: {len(e18_edges)}")
        e18_corr = self._compute_correlations(e18_edges, 'E18')
        print(f"Correlations computed: {len(e18_corr)}")
        e18_grn = e18_corr[(e18_corr['correlation'] >= self.corr_threshold) & 
                           (e18_corr['qval'] < self.qval_threshold)].copy()
        print(f"Final E18: {len(e18_grn)} edges, {e18_grn['TF'].nunique()} TFs")
        
        return e14_grn, e18_grn


# === USAGE IN NOTEBOOK ===
# pruner = GRNPruner(
#     pkn_file='your_cellorcale.parquet',
#     counts_file='/mnt/user-data/uploads/E14__E18_LGE_cortex_seq_Counts.csv',
#     metadata_file='/mnt/user-data/uploads/Samples_Pooling_RNA-Seq.csv',
#     deseq_file='/mnt/user-data/uploads/deseq_temporal_cd133.csv',
#     tf_list=['Sox2', 'Pax6', 'Tbr2'],  # Your TF symbols
#     lfc_threshold=1.0,
#     corr_threshold=0.3
# )
# e14, e18 = pruner.build_networks()