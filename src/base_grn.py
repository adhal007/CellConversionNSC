import pandas as pd
from celloracle import motif_analysis as ma
import genomepy
import os 
import pyranges as pr
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
class GRNCo:
    """
    Wrapper for CellOracle TF motif scanning on ATAC peaks.
    """

    def __init__(self, bed_path, ref_genome="mm39", genomes_dir=None):
        self.bed_path = bed_path
        self.ref_genome = ref_genome
        self.genomes_dir = genomes_dir

        self.bed = None
        self.tss_annotated = None
        self.tfi = None

    # -------------------------
    # Step 1: Load BED
    # -------------------------
    def load_bed(self):
        self.bed = ma.read_bed(self.bed_path)
        return self.bed

    # -------------------------
    # Step 2: Annotate peaks with TSS
    # -------------------------
    def annotate_tss(self):
        peaks = ma.process_bed_file.df_to_list_peakstr(self.bed)
        tss = ma.get_tss_info(
            peak_str_list=peaks,
            ref_genome=self.ref_genome
        )

        peak_ids = ma.process_bed_file.df_to_list_peakstr(tss)
        self.tss_annotated = pd.DataFrame({
            "peak_id": peak_ids,
            "gene_short_name": tss.gene_short_name.values
        }).reset_index(drop=True)

        return self.tss_annotated

    # -------------------------
    # Step 3: Ensure genome installed
    # -------------------------
    def ensure_genome(self):
        installed = ma.is_genome_installed(
            ref_genome=self.ref_genome,
            genomes_dir=self.genomes_dir
        )

        if not installed:
            genomepy.install_genome(
                name=self.ref_genome,
                provider="UCSC"
            )

        return installed

    # -------------------------
    # Step 4: Scan motifs
    # -------------------------
    def scan_motifs(self, fpr=0.02, motifs=None, verbose=True):
        self.tfi = ma.TFinfo(
            peak_data_frame=self.tss_annotated,
            ref_genome=self.ref_genome,
            genomes_dir=self.genomes_dir
        )

        self.tfi.scan(
            fpr=fpr,
            motifs=motifs,
            verbose=verbose
        )

        return self.tfi

    # -------------------------
    # Step 5: Filter motifs
    # -------------------------
    def filter_motifs(self, score_threshold=10):
        self.tfi.reset_filtering()
        self.tfi.filter_motifs_by_score(threshold=score_threshold)
        self.tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)

    # -------------------------
    # Step 6: Save outputs
    # -------------------------
    def save_tfinfo(self, h5_path):
        self.tfi.to_hdf5(file_path=h5_path)

    def save_dataframe(self, parquet_path):
        df = self.tfi.to_dataframe()
        df.to_parquet(parquet_path)
        return df

class GRNBuilder:
    """
    Build Gene Regulatory Networks from multi-omics data.
    
    Combines RNA-seq, ATAC-seq, ChIP-seq, and enhancer predictions
    to construct condition-specific GRNs.
    """
    
    def __init__(self, 
                 deseq_results,
                 consensus_peaks,
                 dar_results,
                 promoters,
                 enhancers,
                 chip_peaks,
                 output_dir):
        """
        Initialize GRN builder with all required data.
        
        Args:
            deseq_results: DataFrame with DE analysis (columns: symbol, log2FoldChange, padj, is_TF)
            consensus_peaks: DataFrame with consensus ATAC peaks (columns: Chromosome, Start, End)
            dar_results: DataFrame with DA peaks (columns: seqnames, start, end, Fold, FDR, ...)
            promoters: DataFrame with promoter regions (columns: chr, start, end, gene)
            enhancers: DataFrame with enhancer regions (columns: chr, start, end, symbol)
            chip_peaks: DataFrame with ChIP-seq peaks (columns: chr, start, end, TF, score)
            output_dir: Path to save output files
        """
        self.deseq_results = deseq_results
        self.consensus_peaks = consensus_peaks
        self.dar_results = dar_results
        self.promoters = promoters
        self.enhancers = enhancers
        self.chip_peaks = chip_peaks
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate inputs
        self._validate_inputs()
        
        print("="*80)
        print("GRN Builder Initialized")
        print("="*80)
        print(f"DE results: {len(self.deseq_results)} genes")
        print(f"  DE genes: {((self.deseq_results['padj'] < 0.05) & (self.deseq_results['log2FoldChange'].abs() > 1)).sum()}")
        print(f"  DE TFs: {((self.deseq_results['padj'] < 0.05) & (self.deseq_results['log2FoldChange'].abs() > 1) & (self.deseq_results['is_TF'])).sum()}")
        print(f"Consensus peaks: {len(self.consensus_peaks)}")
        print(f"DA peaks: {len(self.dar_results)}")
        print(f"Promoters: {len(self.promoters)}")
        print(f"Enhancers: {len(self.enhancers)}")
        print(f"ChIP peaks: {len(self.chip_peaks)}")
        print(f"  Unique TFs: {self.chip_peaks['TF'].nunique()}")
        print(f"Output: {self.output_dir}")
    
    def _validate_inputs(self):
        """Validate that all inputs have required columns."""
        required_cols = {
            'deseq_results': ['symbol', 'log2FoldChange', 'padj', 'is_TF'],
            'consensus_peaks': ['Chromosome', 'Start', 'End'],
            'dar_results': ['seqnames', 'start', 'end', 'Fold'],
            'promoters': ['chr', 'start', 'end', 'gene'],
            'enhancers': ['chr', 'start', 'end', 'symbol'],
            'chip_peaks': ['chr', 'start', 'end', 'TF']
        }
        
        for df_name, cols in required_cols.items():
            df = getattr(self, df_name)
            missing = set(cols) - set(df.columns)
            if missing:
                raise ValueError(f"{df_name} missing columns: {missing}")
        
        print("\n✓ All inputs validated")

    # Add this method to the GRNBuilder class in base_grn.py

    def identify_accessible_regulatory_regions(self, de_only=True, padj_cutoff=0.05, lfc_cutoff=1):
        """
        Identify accessible regulatory regions (promoters + enhancers) for genes.
        
        Args:
            de_only: If True, only consider DE genes
            padj_cutoff: FDR cutoff for DE genes
            lfc_cutoff: Log2 fold-change cutoff for DE genes
        
        Returns:
            DataFrame with accessible regulatory regions and DA annotation
        """
        import pyranges as pr
        
        print("\n" + "="*80)
        print("IDENTIFYING ACCESSIBLE REGULATORY REGIONS")
        print("="*80)
        
        # Get target genes
        if de_only:
            target_genes = self.deseq_results[
                (self.deseq_results['padj'] < padj_cutoff) & 
                (self.deseq_results['log2FoldChange'].abs() > lfc_cutoff)
            ]['symbol'].tolist()
            print(f"Target: {len(target_genes)} DE genes")
        else:
            target_genes = self.deseq_results['symbol'].tolist()
            print(f"Target: All {len(target_genes)} genes")
        
        # Filter promoters and enhancers
        promoters_target = self.promoters[self.promoters['gene'].isin(target_genes)]
        enhancers_target = self.enhancers[self.enhancers['symbol'].isin(target_genes)]
        
        print(f"Promoters: {len(promoters_target)}")
        print(f"Enhancers: {len(enhancers_target)}")
        
        # Convert to PyRanges
        consensus_pr = pr.PyRanges(self.consensus_peaks)
        
        promoters_pr = pr.PyRanges(promoters_target.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}
        ))
        
        enhancers_pr = pr.PyRanges(enhancers_target.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}
        ))
        
        # Find accessible regions
        accessible_promoters = promoters_pr.join(consensus_pr).df
        accessible_enhancers = enhancers_pr.join(consensus_pr).df
        
        print(f"\nAccessible promoters: {len(accessible_promoters)}")
        print(f"Accessible enhancers: {len(accessible_enhancers)}")
        
        # Combine using consensus peak coordinates (Start_b, End_b)
        regulatory_regions = pd.concat([
            accessible_promoters[['Chromosome', 'Start_b', 'End_b', 'gene']].rename(
                columns={'Start_b': 'Start', 'End_b': 'End'}
            ),
            accessible_enhancers[['Chromosome', 'Start_b', 'End_b', 'symbol']].rename(
                columns={'Start_b': 'Start', 'End_b': 'End', 'symbol': 'gene'}
            )
        ], ignore_index=True)
        
        # Add peak_id
        regulatory_regions['peak_id'] = (
            regulatory_regions['Chromosome'].astype(str) + ':' + 
            regulatory_regions['Start'].astype(str) + '-' + 
            regulatory_regions['End'].astype(str)
        )
        
        print(f"\nTotal accessible regulatory regions: {len(regulatory_regions)}")
        print(f"Unique genes: {regulatory_regions['gene'].nunique()}")
        
        # Annotate with DA status
        print("\nAnnotating DA status...")
        regulatory_regions = self._annotate_da_status(regulatory_regions)
        
        # Store results
        self.regulatory_regions = regulatory_regions
        
        return regulatory_regions

    def _annotate_da_status(self, regulatory_regions):
        """Annotate regulatory regions with DA status."""

        
        # Separate E14 and E18 DA peaks
        dar_e14 = self.dar_results[self.dar_results['Fold'] > 0].copy()
        dar_e18 = self.dar_results[self.dar_results['Fold'] < 0].copy()
        
        print(f"  E14 DA peaks: {len(dar_e14)}")
        print(f"  E18 DA peaks: {len(dar_e18)}")
        
        # Convert to PyRanges
        consensus_pr = pr.PyRanges(self.consensus_peaks)
        dar_e14_pr = pr.PyRanges(dar_e14.rename(columns={'seqnames': 'Chromosome', 'start': 'Start', 'end': 'End'}))
        dar_e18_pr = pr.PyRanges(dar_e18.rename(columns={'seqnames': 'Chromosome', 'start': 'Start', 'end': 'End'}))
        
        # Find consensus peaks overlapping DA
        consensus_da_e14 = consensus_pr.join(dar_e14_pr).df
        consensus_da_e18 = consensus_pr.join(dar_e18_pr).df
        
        # Get DA peak IDs
        e14_da_peak_ids = set(
            consensus_da_e14['Chromosome'].astype(str) + ':' + 
            consensus_da_e14['Start'].astype(str) + '-' + 
            consensus_da_e14['End'].astype(str)
        )
        
        e18_da_peak_ids = set(
            consensus_da_e18['Chromosome'].astype(str) + ':' + 
            consensus_da_e18['Start'].astype(str) + '-' + 
            consensus_da_e18['End'].astype(str)
        )
        
        # Annotate
        regulatory_regions['is_E14_DA'] = regulatory_regions['peak_id'].isin(e14_da_peak_ids)
        regulatory_regions['is_E18_DA'] = regulatory_regions['peak_id'].isin(e18_da_peak_ids)
        
        print(f"  Regions with E14 DA: {regulatory_regions['is_E14_DA'].sum()}")
        print(f"  Regions with E18 DA: {regulatory_regions['is_E18_DA'].sum()}")
        
        return regulatory_regions
    
    # Add this method to GRNBuilder class in base_grn.py

    def build_chip_edges(self):
        """
        Build TF→Gene edges by overlapping ChIP peaks with accessible regulatory regions.
        
        Returns:
            DataFrame with TF→Gene edges
        """
        
        print("\n" + "="*80)
        print("BUILDING TF→GENE EDGES FROM ChIP-SEQ")
        print("="*80)
        
        if self.regulatory_regions is None:
            raise ValueError("Run identify_accessible_regulatory_regions() first!")
        
        print(f"ChIP peaks: {len(self.chip_peaks)}")
        print(f"ChIP TFs: {self.chip_peaks['TF'].nunique()}")
        print(f"Regulatory regions: {len(self.regulatory_regions)}")
        
        # Convert to PyRanges
        chip_pr = pr.PyRanges(self.chip_peaks.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}
        ))
        
        regulatory_pr = pr.PyRanges(self.regulatory_regions[[
            'Chromosome', 'Start', 'End', 'gene', 'is_E14_DA', 'is_E18_DA'
        ]])
        
        # Overlap ChIP with regulatory regions
        print("\nOverlapping ChIP peaks with regulatory regions...")
        chip_regulatory_overlap = regulatory_pr.join(chip_pr).df
        
        print(f"Overlapping regions: {len(chip_regulatory_overlap)}")
        
        # Build edges: TF → Gene
        edges = chip_regulatory_overlap[[
            'TF', 'gene', 'Chromosome', 'Start', 'End', 'is_E14_DA', 'is_E18_DA'
        ]].copy()
        
        # Remove duplicate TF-gene pairs
        edges = edges.drop_duplicates(subset=['TF', 'gene'])
        
        print(f"\nBase edges: {len(edges)}")
        print(f"Unique TFs: {edges['TF'].nunique()}")
        print(f"Unique target genes: {edges['gene'].nunique()}")
        print(f"Edges with E14 DA: {edges['is_E14_DA'].sum()}")
        print(f"Edges with E18 DA: {edges['is_E18_DA'].sum()}")
        
        # Store results
        self.base_edges = edges
        
        return edges
    
    # Add this method to GRNBuilder class in base_grn.py

    def filter_by_tf_expression(self, padj_cutoff=0.05, lfc_cutoff=1):
        """
        Filter edges to only include DE TFs.
        
        Args:
            padj_cutoff: FDR cutoff for DE TFs
            lfc_cutoff: Log2 fold-change cutoff for DE TFs
        
        Returns:
            DataFrame with filtered edges
        """
        print("\n" + "="*80)
        print("FILTERING BY TF EXPRESSION")
        print("="*80)
        
        if self.base_edges is None:
            raise ValueError("Run build_chip_edges() first!")
        
        print(f"Base edges: {len(self.base_edges)}")
        
        # Get DE TFs
        de_tfs = self.deseq_results[
            (self.deseq_results['padj'] < padj_cutoff) & 
            (self.deseq_results['log2FoldChange'].abs() > lfc_cutoff) & 
            (self.deseq_results['is_TF'])
        ]
        
        print(f"DE TFs: {len(de_tfs)}")
        print(f"  E14 (LFC > {lfc_cutoff}): {(de_tfs['log2FoldChange'] > lfc_cutoff).sum()}")
        print(f"  E18 (LFC < -{lfc_cutoff}): {(de_tfs['log2FoldChange'] < -lfc_cutoff).sum()}")
        
        # Filter edges
        edges_filtered = self.base_edges[
            self.base_edges['TF'].str.upper().isin(de_tfs['symbol'].str.upper())
        ].copy()
        
        print(f"\nFiltered edges: {len(edges_filtered)}")
        print(f"Unique DE TFs: {edges_filtered['TF'].nunique()}")
        print(f"Unique targets: {edges_filtered['gene'].nunique()}")
        
        # Add TF direction (E14 vs E18)
        tf_direction = dict(zip(
            de_tfs['symbol'].str.upper(), 
            ['E14' if lfc > 0 else 'E18' for lfc in de_tfs['log2FoldChange']]
        ))
        
        edges_filtered['TF_direction'] = edges_filtered['TF'].str.upper().map(tf_direction)
        
        print(f"\nE14 TFs: {(edges_filtered['TF_direction'] == 'E14').sum()} edges")
        print(f"E18 TFs: {(edges_filtered['TF_direction'] == 'E18').sum()} edges")
        
        # Store results
        self.filtered_edges = edges_filtered
        
        return edges_filtered
    
    # Add this method to GRNBuilder class in base_grn.py

    def split_by_condition(self):
        """
        Split filtered edges into condition-specific GRNs (E14 vs E18).
        
        Returns:
            Tuple of (e14_grn, e18_grn) DataFrames
        """
        print("\n" + "="*80)
        print("SPLITTING INTO CONDITION-SPECIFIC GRNs")
        print("="*80)
        
        if self.filtered_edges is None:
            raise ValueError("Run filter_by_tf_expression() first!")
        
        # Separate by TF direction
        e14_grn = self.filtered_edges[self.filtered_edges['TF_direction'] == 'E14'].copy()
        e18_grn = self.filtered_edges[self.filtered_edges['TF_direction'] == 'E18'].copy()
        
        print(f"\nE14 GRN:")
        print(f"  Edges: {len(e14_grn)}")
        print(f"  TFs: {e14_grn['TF'].nunique()}")
        print(f"  Target genes: {e14_grn['gene'].nunique()}")
        print(f"  Edges with DA: {e14_grn['is_E14_DA'].sum()} ({100*e14_grn['is_E14_DA'].sum()/len(e14_grn):.1f}%)")
        
        print(f"\nE18 GRN:")
        print(f"  Edges: {len(e18_grn)}")
        print(f"  TFs: {e18_grn['TF'].nunique()}")
        print(f"  Target genes: {e18_grn['gene'].nunique()}")
        print(f"  Edges with DA: {e18_grn['is_E18_DA'].sum()} ({100*e18_grn['is_E18_DA'].sum()/len(e18_grn):.1f}%)")
        
        # Store results
        self.e14_grn = e14_grn
        self.e18_grn = e18_grn
        
        return e14_grn, e18_grn
    
    # Add this method to GRNBuilder class in base_grn.py

    def visualize_networks(self, save=True):
        """
        Visualize condition-specific GRNs and identify hubs.
        
            Args:
                save: Whether to save figures
        """

        
        print("\n" + "="*80)
        print("NETWORK VISUALIZATION AND HUB ANALYSIS")
        print("="*80)
        
        if self.e14_grn is None or self.e18_grn is None:
            raise ValueError("Run split_by_condition() first!")
        
        # Build NetworkX graphs
        G_e14 = nx.from_pandas_edgelist(self.e14_grn, source='TF', target='gene', 
                                        create_using=nx.DiGraph())
        G_e18 = nx.from_pandas_edgelist(self.e18_grn, source='TF', target='gene', 
                                        create_using=nx.DiGraph())
        
        # Hub analysis
        e14_out = dict(G_e14.out_degree())
        e18_out = dict(G_e18.out_degree())
        
        e14_hubs = sorted([(node, deg) for node, deg in e14_out.items() if deg > 0], 
                        key=lambda x: x[1], reverse=True)
        e18_hubs = sorted([(node, deg) for node, deg in e18_out.items() if deg > 0], 
                        key=lambda x: x[1], reverse=True)
        
        print("\nE14 Top 10 Hubs:")
        for i, (tf, n_targets) in enumerate(e14_hubs[:10], 1):
            da_edges = self.e14_grn[(self.e14_grn['TF'] == tf) & (self.e14_grn['is_E14_DA'])].shape[0]
            print(f"  {i:2d}. {tf:15s} → {n_targets:4d} targets ({da_edges} with DA)")
        
        print("\nE18 Top 10 Hubs:")
        for i, (tf, n_targets) in enumerate(e18_hubs[:10], 1):
            da_edges = self.e18_grn[(self.e18_grn['TF'] == tf) & (self.e18_grn['is_E18_DA'])].shape[0]
            print(f"  {i:2d}. {tf:15s} → {n_targets:4d} targets ({da_edges} with DA)")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='white')
        
        # E14 Network
        self._plot_network(G_e14, e14_hubs, axes[0], 'E14 Network (Neurogenic)', '#e74c3c')
        
        # E18 Network
        self._plot_network(G_e18, e18_hubs, axes[1], 'E18 Network (Gliogenic)', '#3498db')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='E14 TFs'),
            mpatches.Patch(facecolor='#3498db', edgecolor='black', label='E18 TFs'),
            mpatches.Patch(facecolor='#ecf0f1', edgecolor='black', label='Target genes'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
                fontsize=12, frameon=True, fancybox=True)
        
        plt.suptitle('Gene Regulatory Networks\nE14→E18 Cortical Progenitor Transition', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            output_path = os.path.join(self.output_dir, 'network_visualization.pdf')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\nSaved: {output_path}")
        
        plt.show()
        
        return e14_hubs, e18_hubs

    def _plot_network(self, G, hubs, ax, title, color):
        """Helper function to plot a single network."""
        
        out_degree = dict(G.out_degree())
        tfs = [node for node in G.nodes() if out_degree[node] > 0]
        targets = [node for node in G.nodes() if out_degree[node] == 0]
        
        # Hierarchical layout
        pos = {}
        
        # Place TFs in circle at top
        tf_angles = np.linspace(0, 2*np.pi, len(tfs), endpoint=False)
        for i, tf in enumerate(tfs):
            radius = 2 if len(tfs) <= 10 else 3
            pos[tf] = (np.cos(tf_angles[i]) * radius, np.sin(tf_angles[i]) * radius + 5)
        
        # Place targets in cloud below
        np.random.seed(42)
        for target in targets:
            pos[target] = (np.random.uniform(-8, 8), np.random.uniform(-3, 3))
        
        # Node sizes
        node_sizes = []
        for node in G.nodes():
            if node in tfs:
                node_sizes.append(max(out_degree[node] * 20, 500))
            else:
                node_sizes.append(10)
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            if node in tfs:
                node_colors.append(color)
            else:
                node_colors.append('#ecf0f1')
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                            alpha=0.8, ax=ax, edgecolors='black', linewidths=0.5)
        
        nx.draw_networkx_edges(G, pos, edge_color='#95a5a6', alpha=0.2, 
                            arrows=True, arrowsize=5, ax=ax, width=0.3,
                            connectionstyle='arc3,rad=0.1')
        
        # Label only TFs
        tf_labels = {tf: tf for tf in tfs}
        nx.draw_networkx_labels(G, pos, labels=tf_labels, font_size=10 if len(tfs) > 10 else 12, 
                            font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 8)
        
        # Add stats
        stats_text = f'{len(tfs)} TFs\n{len(targets)} targets\n{G.number_of_edges()} edges'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))