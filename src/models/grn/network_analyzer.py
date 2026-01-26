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
            k=3,              # Increase separation
            iterations=100,   # More iterations for better layout
            seed=42
        )
        
        # Node sizes based on degree
        out_deg = dict(self.G.out_degree())
        node_sizes = [500 + out_deg.get(node, 0) * 50 for node in H.nodes()]
        
        # Draw edges with transparency
        nx.draw_networkx_edges(
            H, pos,
            edge_color='#333333',  # Dark gray
            alpha=0.2,             # Much lighter/transparent
            arrows=True,
            arrowsize=15,
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
            font_size=8,
            font_weight='bold',
            font_color='black'
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()