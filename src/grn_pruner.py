"""
Simple GRN pruner: Filter Cell Oracle PKN by TF list and DE genes.
Uses gene SYMBOLS throughout.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests



class GRNPruner:
    """Prune base GRN to stage-specific TF networks."""
    
    def __init__(self, base_grn, counts_file, metadata_file, deseq_file, tf_list,
                 lfc_threshold=1.0, corr_threshold=0.3, qval_threshold=0.05, padj_threshold=0.05,
                 grn_format='auto'):
        """
        Args:
            base_grn: Base GRN in one of these formats:
                - CellOracle parquet file path (peak_id, gene_short_name, TF columns)
                - ChIP-seq edge list: DataFrame or file path with [TF, Target] columns
                - ATAC+ChIP: DataFrame or file path with [peak_id, gene_short_name] + TF overlap
            counts_file, metadata_file, deseq_file: RNA-seq data files
            tf_list: List of TF SYMBOLS to keep
            lfc_threshold: E14: LFC > threshold, E18: LFC < -threshold
            corr_threshold: Minimum correlation
            qval_threshold: Q-value threshold
            grn_format: 'celloracle', 'chipseq', 'atac_chip', or 'auto' (detect automatically)
        """
        self.tf_list = set(tf_list)
        self.lfc_threshold = lfc_threshold
        self.corr_threshold = corr_threshold
        self.qval_threshold = qval_threshold
        self.padj_threshold = padj_threshold
        
        print("Loading base GRN...")
        self.base_grn, self.grn_format = self._load_grn(base_grn, grn_format)
        print(f"GRN format: {self.grn_format}")
        
        print("Loading RNA-seq data...")
        counts = pd.read_csv(counts_file, sep=';', header=0).iloc[:, 1:]
        counts = counts.set_index(counts.columns[0])
        
        metadata = pd.read_csv(metadata_file, sep=';')
        deseq = pd.read_csv(deseq_file, index_col=0)
        
        # Map Ensembl IDs to symbols
        ensembl_to_symbol = dict(zip(deseq['gene_id'], deseq['symbol']))
        counts.index = counts.index.map(lambda x: ensembl_to_symbol.get(x, x))
        counts = counts[~counts.index.duplicated(keep='first')]
        self.counts = counts
        
        self.metadata = metadata
        self.deseq = deseq
        
        print(f"Base GRN loaded, TF list: {len(self.tf_list)} TFs")
    
    def _load_grn(self, base_grn, grn_format):
        """Load GRN from various formats."""
        # Load file if path provided
        if isinstance(base_grn, str):
            if base_grn.endswith('.parquet'):
                df = pd.read_parquet(base_grn)
            elif base_grn.endswith('.csv'):
                df = pd.read_csv(base_grn)
            elif base_grn.endswith('.tsv'):
                df = pd.read_csv(base_grn, sep='\t')
            else:
                raise ValueError(f"Unknown file format: {base_grn}")
        else:
            df = base_grn.copy()
        
        # Auto-detect format if needed
        if grn_format == 'auto':
            if 'TF' in df.columns and 'Target' in df.columns:
                grn_format = 'chipseq'
            elif 'TF' in df.columns and 'target' in df.columns:
                grn_format = 'chipseq'
            elif 'peak_id' in df.columns and any('gene' in c.lower() for c in df.columns):
                # Has peak_id and gene column
                tf_cols = [c for c in df.columns if c not in ['peak_id'] and 'gene' not in c.lower()]
                if len(tf_cols) > 10:
                    grn_format = 'celloracle'
                else:
                    grn_format = 'atac_chip'
            else:
                raise ValueError(f"Cannot auto-detect GRN format. Columns: {df.columns.tolist()}")
        
        return df, grn_format
    
    def _grn_to_edges(self):
        """Convert base GRN to standard edge list (TF, target)."""
        if self.grn_format == 'chipseq':
            # Already in edge list format
            edges = self.base_grn.copy()
            
            # Standardize column names
            if 'Target' in edges.columns:
                edges = edges.rename(columns={'Target': 'target'})
            
            # Keep only TF and target columns
            edges = edges[['TF', 'target']].drop_duplicates()
            
            # FIX CASE: Convert to proper mouse gene case (First letter caps)
            edges['TF'] = edges['TF'].str.capitalize()
            edges['target'] = edges['target'].str.capitalize()
            
            print(f"ChIP-seq edges: {len(edges)}")
            
        elif self.grn_format == 'celloracle':
            # CellOracle format: peak_id, gene_short_name, TF columns (binary)
            gene_col = [c for c in self.base_grn.columns if 'gene' in c.lower()][0]
            tf_cols = [c for c in self.base_grn.columns if c not in ['peak_id', gene_col]]
            
            edges = []
            for _, row in self.base_grn.iterrows():
                target = row[gene_col]
                for tf in tf_cols:
                    if row[tf] > 0:
                        edges.append({'TF': tf, 'target': target})
            
            edges = pd.DataFrame(edges).drop_duplicates()
            print(f"CellOracle edges: {len(edges)}")
            
        elif self.grn_format == 'atac_chip':
            # ATAC peaks with ChIP TF binding
            gene_col = [c for c in self.base_grn.columns if 'gene' in c.lower()][0]
            tf_col = 'TF'
            
            edges = self.base_grn[[tf_col, gene_col]].copy()
            edges = edges.rename(columns={gene_col: 'target', tf_col: 'TF'})
            
            # FIX CASE
            edges['TF'] = edges['TF'].str.capitalize()
            edges['target'] = edges['target'].str.capitalize()
            
            edges = edges.drop_duplicates()
            print(f"ATAC+ChIP edges: {len(edges)}")
        
        else:
            raise ValueError(f"Unknown GRN format: {self.grn_format}")
        
        return edges
    
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
        else:
            # Return empty dataframe with correct columns
            df = pd.DataFrame(columns=['TF', 'target', 'correlation', 'pval', 'qval'])
        
        return df


    
    def _filter_by_de(self, edges, stage):
        """Filter edges where BOTH TF AND target are DE."""
        if stage == 'E14':
            de_symbols = set(self.deseq[
                (self.deseq['log2FoldChange'] > self.lfc_threshold) & 
                (self.deseq['padj'] < self.padj_threshold)
            ]['symbol'])
        else:
            de_symbols = set(self.deseq[
                (self.deseq['log2FoldChange'] < -self.lfc_threshold) & 
                (self.deseq['padj'] < self.padj_threshold)
            ]['symbol'])
        
        return edges[edges['TF'].isin(de_symbols) & edges['target'].isin(de_symbols)].copy()
    
    def build_networks(self):
        """Build E14 and E18 TF-TF networks. Returns (e14_grn, e18_grn)."""
        print("\n=== Converting to edge list ===")
        edges = self._grn_to_edges()
        print(f"Total edges: {len(edges)}")
        
        print("\n=== Filtering to TF-TF ===")
        edges = self._filter_tf_tf(edges)
        print(f"TF-TF edges: {len(edges)}")
        
        print("\n=== E14 Network ===")
        e14_edges = self._filter_by_de(edges.copy(), 'E14')
        print(f"After DE filter: {len(e14_edges)}")
        
        if len(e14_edges) == 0:
            print("WARNING: No E14 edges after filtering!")
            e14_grn = pd.DataFrame(columns=['TF', 'target', 'correlation', 'pval', 'qval'])
        else:
            e14_corr = self._compute_correlations(e14_edges, 'E14')
            print(f"Correlations computed: {len(e14_corr)}")
            
            if len(e14_corr) == 0:
                print("WARNING: No E14 correlations computed!")
                e14_grn = pd.DataFrame(columns=['TF', 'target', 'correlation', 'pval', 'qval'])
            else:
                e14_grn = e14_corr[(e14_corr['correlation'] >= self.corr_threshold) & 
                                (e14_corr['qval'] < self.qval_threshold)].copy()
                print(f"Final E14: {len(e14_grn)} edges, {e14_grn['TF'].nunique()} TFs")
        
        print("\n=== E18 Network ===")
        e18_edges = self._filter_by_de(edges.copy(), 'E18')
        print(f"After DE filter: {len(e18_edges)}")
        
        if len(e18_edges) == 0:
            print("WARNING: No E18 edges after filtering!")
            e18_grn = pd.DataFrame(columns=['TF', 'target', 'correlation', 'pval', 'qval'])
        else:
            e18_corr = self._compute_correlations(e18_edges, 'E18')
            print(f"Correlations computed: {len(e18_corr)}")
            
            if len(e18_corr) == 0:
                print("WARNING: No E18 correlations computed!")
                e18_grn = pd.DataFrame(columns=['TF', 'target', 'correlation', 'pval', 'qval'])
            else:
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