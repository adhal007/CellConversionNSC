import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt 

class NetworkAnalyzer:
    def __init__(self, grn_df: pd.DataFrame, tf_col: str, target_col: str, deseq_results: pd.DataFrame) -> None:
        """Initialize NetworkAnalyzer."""
        self.grn_df = grn_df
        self.tf_col = tf_col 
        self.target_col = target_col 
        self.deseq_results = deseq_results
        self.G = None 
        self.scores = None 
        
    def build_graph(self):
        """Build directed graph from GRN."""
        if self.G is None:              
            self.G = nx.DiGraph()
            self.G.add_edges_from(self.grn_df[[self.tf_col, self.target_col]].itertuples(index=False, name=None))
        return self.G
        
    def compute_network_stats(self):
        """Compute network centrality metrics."""
        # Build graph first if not already built
        if self.G is None:
            self.build_graph()
            
        print(f"\nNetwork: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Compute metrics
        out_deg = dict(self.G.out_degree())
        in_deg = dict(self.G.in_degree())
        reachability = {n: len(nx.descendants(self.G, n)) for n in self.G.nodes()}
        pagerank = nx.pagerank(self.G) if self.G.number_of_edges() > 0 else {n: 0 for n in self.G.nodes()}
        betweenness = nx.betweenness_centrality(self.G, normalized=True)
        katz = nx.katz_centrality(self.G, alpha=0.01, beta=1.0, max_iter=1000)
        
        self.scores = pd.DataFrame({
            "out_degree": out_deg,
            "in_degree": in_deg,
            "reachability": reachability,
            "pagerank": pagerank,
            "betweenness": betweenness,
            "katz": katz
        }).fillna(0)
        
        # Get TF set (case-insensitive matching)
        tf_symbols_upper = set(self.deseq_results[self.deseq_results['is_TF']]['symbol'].str.upper())
        
        # Filter to TFs - match nodes case-insensitively
        tf_scores = self.scores[self.scores.index.str.upper().isin(tf_symbols_upper)]
        
        # SCC Analysis
        print("\n" + "="*80)
        print("STRONGLY CONNECTED COMPONENTS (SCCs)")
        print("="*80)
        
        sccs = list(nx.strongly_connected_components(self.G))
        print(f"Total SCCs: {len(sccs)}")
        
        # Filter SCCs with size >= 2 (feedback loops)
        sccs_feedback = [scc for scc in sccs if len(scc) >= 2]
        print(f"SCCs with feedback loops (size >= 2): {len(sccs_feedback)}")
        
        if len(sccs_feedback) > 0:
            print("\nFeedback loop SCCs:")
            for i, scc in enumerate(sorted(sccs_feedback, key=len, reverse=True), 1):
                print(f"  SCC {i}: Size {len(scc)}: {sorted(list(scc))}")
        else:
            print("\nNo feedback loops found - network is a DAG")
        
        # Largest SCC
        largest_scc = max(sccs, key=len)
        print(f"\nLargest SCC: Size {len(largest_scc)}")
        if len(largest_scc) >= 2:
            print(f"  Members: {sorted(list(largest_scc))}")
            
        return {
            'network_scores': self.scores, 
            'tf_scores': tf_scores, 
            'largest_scc': largest_scc, 
            'all_sccs': sccs
        }
        
    def visualize_network(self, title: str, network_score_col: str = 'reachability', top_k: int = 50):
        """Visualize top regulators and their descendants."""
        if self.scores is None:
            print("Please run compute_network_stats() first!")
            return
            
        # Get top master regulators
        top_mrs = self.scores.sort_values(network_score_col, ascending=False).head(top_k).index
        
        nodes = set(top_mrs)
        for tf in top_mrs:
            nodes |= nx.descendants(self.G, tf)
        
        H = self.G.subgraph(nodes).copy()
        
        print(f"Subnetwork: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
        
        # IMPROVED VISUALIZATION
        plt.figure(figsize=(20, 15))
        
        # Better layout with more separation
        pos = nx.spring_layout(
            H, 
            k=2,              # Increase separation
            iterations=100,   # More iterations for better layout
            seed=42
        )
        
        # Node sizes based on degree
        out_deg = dict(self.G.out_degree())
        node_sizes = [1000 + out_deg.get(node, 0) * 50 for node in H.nodes()]
        
        # Draw edges with transparency
        nx.draw_networkx_edges(
            H, pos,
            edge_color='#333333',  # Dark gray
            alpha=0.2,             # Much lighter/transparent
            arrows=True,
            arrowsize=7.5,
            width=0.8,             # Thinner edges
            connectionstyle='arc3,rad=0.1'  # Curved edges to reduce overlap
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            H, pos,
            node_color='#3498db',  # Blue
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            H, pos,
            font_size=12,
            font_weight='bold',
            font_color='black'
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def check_tf_tf_edges(self):
        # Check E14 GRN
        print("Total edges:", len(self.grn_df))
        print("Unique TFs:", self.grn_df[self.tf_col].nunique())
        print("Unique targets:", self.grn_df[self.target_col].nunique())

        # Check how many targets are also TFs (TF-TF edges)
        tf_set = set(self.deseq_results[self.deseq_results['is_TF']]['symbol'].str.upper())
        tf_tf_edges = self.grn_df[self.grn_df[self.target_col].str.upper().isin(tf_set)]

        print(f"\nTF-TF edges: {len(tf_tf_edges)}")
        print(f"TF→non-TF edges: {len(self.grn_df) - len(tf_tf_edges)}")

        if len(tf_tf_edges) > 0:
            print("\nSample TF-TF edges:")
            print(tf_tf_edges.head(10))
        return tf_tf_edges 
    
    def get_tf_tf_network(self, tf_symbols, grn_e14):
        # tf_symbols = list(nsc.tf_symbols)
        tf_symbols = [i.upper() for i in tf_symbols]
        grn_e14['gene'] = [i.upper() for i in grn_e14['gene'].tolist()]
        tf_tf_grn_e14 = grn_e14[(grn_e14['gene'].isin(tf_symbols)) & (grn_e14['TF'] != grn_e14['gene'])]

        # grn_e18['gene'] = [i.upper() for i in grn_e18['gene'].tolist()]
        # tf_tf_grn_e18 = grn_e18[(grn_e18['gene'].isin(tf_symbols)) & (grn_e18['TF'] != grn_e18['gene'])]
        return tf_tf_grn_e14
    
    def annotate_patterning_factors(grn_df, patterning_factors, tf_col='TF', gene_col='gene'):
        """
        Add columns indicating if TF or target is a patterning factor.
        
        Args:
            grn_df: DataFrame with GRN edges (columns: TF, gene, ...)
            patterning_factors: List of TF names to mark
            tf_col: Column name for TFs (default: 'TF')
            gene_col: Column name for target genes (default: 'gene')
        
        Returns:
            GRN DataFrame with added columns: 'TF_is_patterning', 'target_is_patterning'
        """
        # Convert to uppercase for case-insensitive matching
        patterning_set = set([pf.upper() for pf in patterning_factors])
        
        print(f"\nAnnotating GRN with patterning factors...")
        print(f"Total edges: {len(grn_df)}")
        print(f"Patterning factors to mark: {len(patterning_factors)}")
        
        # Add boolean columns
        grn_df['TF_is_patterning'] = grn_df[tf_col].str.upper().isin(patterning_set)
        grn_df['target_is_patterning'] = grn_df[gene_col].str.upper().isin(patterning_set)
        
        # Statistics
        tf_patterning_count = grn_df['TF_is_patterning'].sum()
        target_patterning_count = grn_df['target_is_patterning'].sum()
        both_patterning_count = (grn_df['TF_is_patterning'] & grn_df['target_is_patterning']).sum()
        either_patterning_count = (grn_df['TF_is_patterning'] | grn_df['target_is_patterning']).sum()
        
        print(f"\nEdges where TF is patterning factor: {tf_patterning_count} ({tf_patterning_count/len(grn_df)*100:.1f}%)")
        print(f"Edges where target is patterning factor: {target_patterning_count} ({target_patterning_count/len(grn_df)*100:.1f}%)")
        print(f"Edges where both are patterning factors: {both_patterning_count}")
        print(f"Edges involving any patterning factor: {either_patterning_count} ({either_patterning_count/len(grn_df)*100:.1f}%)")
        
        # Show which patterning factors were found
        found_tfs = set(grn_df[grn_df['TF_is_patterning']][tf_col].str.upper())
        found_genes = set(grn_df[grn_df['target_is_patterning']][gene_col].str.upper())
        
        print(f"\nPatterning factors found as TFs: {len(found_tfs)}")
        if found_tfs:
            print(f"  {sorted(found_tfs)}")
        
        print(f"Patterning factors found as targets: {len(found_genes)}")
        if found_genes:
            print(f"  {sorted(found_genes)}")
        
        # Show any patterning factors NOT found in the GRN
        not_found = patterning_set - found_tfs - found_genes
        if not_found:
            print(f"\nPatterning factors NOT found in GRN: {sorted(not_found)}")
        
        return grn_df


# # # Example usage:
# # patterning_factors = ['PAX6', 'EMX2', 'NKX2-1', 'GSX2', 'DLX1', 'DLX2']

# # Annotate E14 GRN
# grn_builder.grn_e14 = annotate_patterning_factors(
#     grn_builder.grn_e14, 
#     patterning_factors
# )

# # Annotate E18 GRN
# grn_builder.grn_e18 = annotate_patterning_factors(
#     grn_builder.grn_e18,
#     patterning_factors
# )