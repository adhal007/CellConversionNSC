
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set

def plot_candidate_tf_heatmap(
    e14_tfs: pd.DataFrame,
    e18_tfs: pd.DataFrame,
    vst_counts: pd.DataFrame,
    metadata: pd.DataFrame,
    ensembl_to_symbol: Dict[str, str],
    group1: str = 'E14',
    group2: str = 'E18',
    group_col: str = 'Stage',
    title: str = "Candidate TFs",
    figsize: Tuple[int, int] = (12, 10),
    save_path: str = None
):
    """
    Plot heatmap of candidate TFs expression across conditions.
    
    Args:
        e14_tfs: DataFrame of E14-high TFs (to upregulate)
        e18_tfs: DataFrame of E18-high TFs (to downregulate)
        vst_counts: VST-normalized counts (genes x samples)
        metadata: Sample metadata
        ensembl_to_symbol: Mapping dict
        group1, group2: Group names
        group_col: Column in metadata for grouping
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    
    # Get samples
    g1_samples = [s for s in metadata[metadata[group_col] == group1].index 
                  if s in vst_counts.columns]
    g2_samples = [s for s in metadata[metadata[group_col] == group2].index 
                  if s in vst_counts.columns]
    
    # Get gene IDs (index) for TFs
    g1_tf_ids = [idx for idx in e14_tfs.index if idx in vst_counts.index]
    g2_tf_ids = [idx for idx in e18_tfs.index if idx in vst_counts.index]
    
    # Combine: E14-high first, then E18-high
    all_tf_ids = g1_tf_ids + g2_tf_ids
    n_g1 = len(g1_tf_ids)
    
    if len(all_tf_ids) == 0:
        print("No TFs found in expression data")
        return
    
    print(f"{group1}-high TFs: {n_g1}")
    print(f"{group2}-high TFs: {len(g2_tf_ids)}")
    
    # Get expression matrix
    expr = vst_counts.loc[all_tf_ids, g1_samples + g2_samples]
    
    # Z-score normalize
    expr_z = expr.subtract(expr.mean(axis=1), axis=0).div(expr.std(axis=1), axis=0)
    
    # Get gene symbols for y-axis labels
    gene_labels = [ensembl_to_symbol.get(g, g) for g in all_tf_ids]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        expr_z.values,
        cmap='RdBu_r',
        center=0,
        vmin=-2.5,
        vmax=2.5,
        xticklabels=False,
        yticklabels=gene_labels,
        ax=ax,
        cbar_kws={'label': 'Z-score', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='white'
    )
    
    # Style y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, style='italic')
    
    # Add vertical line separating E14 and E18 samples
    ax.axvline(len(g1_samples), color='black', linewidth=2)
    
    # Add horizontal line separating E14-high and E18-high TFs
    if n_g1 > 0 and n_g1 < len(all_tf_ids):
        ax.axhline(n_g1, color='black', linewidth=2)
    
    # Add sample group labels
    ax.text(len(g1_samples) / 2, -0.5, group1, ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='#C0392B')
    ax.text(len(g1_samples) + len(g2_samples) / 2, -0.5, group2, ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#2980B9')
    
    # # Add TF group labels on right side
    # if n_g1 > 0:
    #     ax.text(len(g1_samples) + len(g2_samples) + 0.5, n_g1 / 2, 
    #             f'↑ {group1}\n(upregulate)', ha='left', va='center',
    #             fontsize=10, color='#C0392B', fontweight='bold')
    # if len(g2_tf_ids) > 0:
    #     ax.text(len(g1_samples) + len(g2_samples) + 0.5, n_g1 + len(g2_tf_ids) / 2,
    #             f'↓ {group2}\n(downregulate)', ha='left', va='center',
    #             fontsize=10, color='#2980B9', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return expr_z

def plot_grn(
    grn: pd.DataFrame,
    title: str = "GRN",
    top_n_edges: int = 50,
    node_size_by: str = 'degree',
    edge_width_by: str = 'interaction_score',
    highlight_tf_tf: bool = True,
    figsize: Tuple[int, int] = (14, 12),
    save_path: str = None
):
    """
    Plot a Gene Regulatory Network.
    
    Args:
        grn: GRN dataframe from build_grn() with columns: source, target, interaction_score, target_is_TF
        title: Plot title
        top_n_edges: Number of top edges to plot (by interaction_score)
        node_size_by: 'degree' or 'expression'
        edge_width_by: Column to use for edge width
        highlight_tf_tf: Highlight TF→TF edges differently
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
    
    if len(grn) == 0:
        print("Empty GRN, nothing to plot")
        return
    
    # Filter to top edges
    grn_plot = grn.nlargest(top_n_edges, 'interaction_score').copy()
    
    print(f"Plotting top {len(grn_plot)} edges")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges
    for _, row in grn_plot.iterrows():
        G.add_edge(
            row['source'], 
            row['target'],
            weight=row['interaction_score'],
            is_tf_tf=row['target_is_TF'],
            n_samples=row.get('n_samples', 1)
        )
    
    # Node attributes
    sources = set(grn_plot['source'])
    targets = set(grn_plot['target'])
    tf_targets = set(grn_plot[grn_plot['target_is_TF']]['target'])
    
    for node in G.nodes():
        if node in sources:
            G.nodes[node]['type'] = 'TF_source'
        elif node in tf_targets:
            G.nodes[node]['type'] = 'TF_target'
        else:
            G.nodes[node]['type'] = 'Gene'
    
    # Calculate node sizes based on degree
    if node_size_by == 'degree':
        out_degrees = dict(G.out_degree())
        in_degrees = dict(G.in_degree())
        node_sizes = []
        for node in G.nodes():
            size = out_degrees.get(node, 0) * 100 + in_degrees.get(node, 0) * 50 + 200
            node_sizes.append(min(size, 2000))  # Cap size
    else:
        node_sizes = [500] * len(G.nodes())
    
    # Node colors
    node_colors = []
    for node in G.nodes():
        ntype = G.nodes[node]['type']
        if ntype == 'TF_source':
            node_colors.append('#E74C3C')  # Red for source TFs
        elif ntype == 'TF_target':
            node_colors.append('#9B59B6')  # Purple for TF targets
        else:
            node_colors.append('#3498DB')  # Blue for gene targets
    
    # Edge widths and colors
    edge_widths = []
    edge_colors = []
    edge_styles = []
    
    for u, v, data in G.edges(data=True):
        width = data['weight'] * 3 + 0.5
        edge_widths.append(width)
        
        if data.get('is_tf_tf', False) and highlight_tf_tf:
            edge_colors.append('#E74C3C')  # Red for TF→TF
            edge_styles.append('solid')
        else:
            edge_colors.append('#7F8C8D')  # Gray for TF→Gene
            edge_styles.append('solid')
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors='white',
        linewidths=2,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='TF (source)'),
        mpatches.Patch(color='#9B59B6', label='TF (target)'),
        mpatches.Patch(color='#3498DB', label='Gene (target)'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='TF → TF'),
        plt.Line2D([0], [0], color='#7F8C8D', linewidth=2, label='TF → Gene')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print network stats
    print(f"\nNetwork stats:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  TF→TF edges: {sum(1 for u,v,d in G.edges(data=True) if d.get('is_tf_tf'))}")
    
    return G


def plot_grn_comparison(
    grn1: pd.DataFrame,
    grn2: pd.DataFrame,
    name1: str = 'E14',
    name2: str = 'E18',
    top_n_edges: int = 30,
    figsize: Tuple[int, int] = (20, 10),
    save_path: str = None
):
    """
    Plot two GRNs side by side for comparison.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for ax, grn, name in zip(axes, [grn1, grn2], [name1, name2]):
        if len(grn) == 0:
            ax.text(0.5, 0.5, f"No edges in {name} GRN", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue
        
        # Filter to top edges
        grn_plot = grn.nlargest(top_n_edges, 'interaction_score').copy()
        
        # Create graph
        G = nx.DiGraph()
        
        for _, row in grn_plot.iterrows():
            G.add_edge(
                row['source'], 
                row['target'],
                weight=row['interaction_score'],
                is_tf_tf=row['target_is_TF']
            )
        
        # Node types
        sources = set(grn_plot['source'])
        tf_targets = set(grn_plot[grn_plot['target_is_TF']]['target'])
        
        for node in G.nodes():
            if node in sources:
                G.nodes[node]['type'] = 'TF_source'
            elif node in tf_targets:
                G.nodes[node]['type'] = 'TF_target'
            else:
                G.nodes[node]['type'] = 'Gene'
        
        # Node sizes
        out_degrees = dict(G.out_degree())
        in_degrees = dict(G.in_degree())
        node_sizes = [
            min(out_degrees.get(n, 0) * 100 + in_degrees.get(n, 0) * 50 + 200, 1500)
            for n in G.nodes()
        ]
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            ntype = G.nodes[node]['type']
            if ntype == 'TF_source':
                node_colors.append('#E74C3C')
            elif ntype == 'TF_target':
                node_colors.append('#9B59B6')
            else:
                node_colors.append('#3498DB')
        
        # Edge properties
        edge_widths = [d['weight'] * 3 + 0.5 for u, v, d in G.edges(data=True)]
        edge_colors = ['#E74C3C' if d.get('is_tf_tf') else '#7F8C8D' 
                       for u, v, d in G.edges(data=True)]
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                               alpha=0.6, arrows=True, arrowsize=12,
                               connectionstyle='arc3,rad=0.1', ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                               alpha=0.9, edgecolors='white', linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
        
        ax.set_title(f"{name} GRN\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
                     fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Shared legend
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='TF (source)'),
        mpatches.Patch(color='#9B59B6', label='TF (target)'),
        mpatches.Patch(color='#3498DB', label='Gene'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=2, label='TF → TF'),
        plt.Line2D([0], [0], color='#7F8C8D', linewidth=2, label='TF → Gene')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_grn_heatmap(
    grn: pd.DataFrame,
    title: str = "TF-Target Interaction Heatmap",
    top_n_tfs: int = 15,
    top_n_targets: int = 30,
    figsize: Tuple[int, int] = (14, 10),
    save_path: str = None
):
    """
    Plot GRN as a heatmap of TF-target interactions.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if len(grn) == 0:
        print("Empty GRN")
        return
    
    # Get top TFs by out-degree
    tf_counts = grn.groupby('source').size().sort_values(ascending=False)
    top_tfs = tf_counts.head(top_n_tfs).index.tolist()
    
    # Get top targets by in-degree
    target_counts = grn.groupby('target').size().sort_values(ascending=False)
    top_targets = target_counts.head(top_n_targets).index.tolist()
    
    # Create pivot table
    grn_filtered = grn[
        (grn['source'].isin(top_tfs)) & 
        (grn['target'].isin(top_targets))
    ]
    
    pivot = grn_filtered.pivot_table(
        index='source', 
        columns='target', 
        values='interaction_score',
        aggfunc='max'
    ).fillna(0)
    
    # Reorder
    pivot = pivot.loc[
        [tf for tf in top_tfs if tf in pivot.index],
        [t for t in top_targets if t in pivot.columns]
    ]
    
    # Mark TF targets
    tf_target_mask = [t in set(grn['source']) for t in pivot.columns]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Interaction Score'},
        linewidths=0.5,
        linecolor='white'
    )
    
    ax.set_xlabel('Target', fontsize=12)
    ax.set_ylabel('TF', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Highlight TF targets in column labels
    xlabels = ax.get_xticklabels()
    for i, label in enumerate(xlabels):
        if pivot.columns[i] in set(grn[grn['target_is_TF']]['target']):
            label.set_color('#E74C3C')
            label.set_fontweight('bold')
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

def plot_tf_filtering_summary(
    candidates: dict,
    group1: str = 'E14',
    group2: str = 'E18',
    group1_label: str = 'Neurogenic',
    group2_label: str = 'Gliogenic',
    group1_color: str = '#2C5282',
    group2_color: str = '#C53030',
    figsize_venn: tuple = (14, 6),
    figsize_upset: tuple = (14, 7),
    save_path: str = None,
    dpi: int = 300
) -> dict:
    """
    Create publication-ready Venn diagrams and UpSet plot for TF filtering.
    
    Args:
        candidates: Output from get_candidates_with_binding()
        group1, group2: Group names (e.g., 'E14', 'E18')
        group1_label, group2_label: Display labels (e.g., 'Neurogenic', 'Gliogenic')
        group1_color, group2_color: Colors for each group
        figsize_venn: Figure size for Venn diagrams
        figsize_upset: Figure size for UpSet plot
        save_path: Directory to save figures (None = don't save)
        dpi: Resolution for saved figures
        
    Returns:
        Dict with sets and summary statistics
    """
    from matplotlib_venn import venn2
    from upsetplot import from_contents, UpSet
    import matplotlib.pyplot as plt
    
    # Publication settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
    })
    
    # Extract sets
    g1_all = set(candidates[f'{group1}_TFs']['symbol'])
    g1_with_binding = set(candidates[f'{group1}_TFs_with_binding']['symbol'])
    g2_all = set(candidates[f'{group2}_TFs']['symbol'])
    g2_with_binding = set(candidates[f'{group2}_TFs_with_binding']['symbol'])
    tfs_in_overlap = candidates['tfs_in_overlap']
    
    # =========================================================================
    # VENN DIAGRAMS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=figsize_venn)
    
    # E14-high TFs
    v1 = venn2(
        [g1_all, tfs_in_overlap], 
        set_labels=('DESeq2 sig', 'ChIP+ATAC'),
        ax=axes[0]
    )
    # Customize colors and fonts
    if v1.get_patch_by_id('10'):
        v1.get_patch_by_id('10').set_color('#fee2e2')
        v1.get_patch_by_id('10').set_edgecolor('black')
        v1.get_patch_by_id('10').set_linewidth(1.5)
    if v1.get_patch_by_id('01'):
        v1.get_patch_by_id('01').set_color('#dbeafe')
        v1.get_patch_by_id('01').set_edgecolor('black')
        v1.get_patch_by_id('01').set_linewidth(1.5)
    if v1.get_patch_by_id('11'):
        v1.get_patch_by_id('11').set_color('#c4b5fd')
        v1.get_patch_by_id('11').set_edgecolor('black')
        v1.get_patch_by_id('11').set_linewidth(1.5)
    
    # Increase label font sizes
    for text in v1.set_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight('bold')
    for text in v1.subset_labels:
        if text:
            text.set_fontsize(16)
            text.set_fontweight('bold')
    
    axes[0].set_title(
        f'{group1}-high TFs ({group1_label})\nTotal: {len(g1_all)}, With binding: {len(g1_with_binding)}', 
        fontsize=16, fontweight='bold', color=group1_color, pad=20
    )
    
    # E18-high TFs
    v2 = venn2(
        [g2_all, tfs_in_overlap], 
        set_labels=('DESeq2 sig', 'ChIP+ATAC'),
        ax=axes[1]
    )
    if v2.get_patch_by_id('10'):
        v2.get_patch_by_id('10').set_color('#fee2e2')
        v2.get_patch_by_id('10').set_edgecolor('black')
        v2.get_patch_by_id('10').set_linewidth(1.5)
    if v2.get_patch_by_id('01'):
        v2.get_patch_by_id('01').set_color('#dbeafe')
        v2.get_patch_by_id('01').set_edgecolor('black')
        v2.get_patch_by_id('01').set_linewidth(1.5)
    if v2.get_patch_by_id('11'):
        v2.get_patch_by_id('11').set_color('#c4b5fd')
        v2.get_patch_by_id('11').set_edgecolor('black')
        v2.get_patch_by_id('11').set_linewidth(1.5)
    
    for text in v2.set_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight('bold')
    for text in v2.subset_labels:
        if text:
            text.set_fontsize(16)
            text.set_fontweight('bold')
    
    axes[1].set_title(
        f'{group2}-high TFs ({group2_label})\nTotal: {len(g2_all)}, With binding: {len(g2_with_binding)}', 
        fontsize=16, fontweight='bold', color=group2_color, pad=20
    )
    
    plt.suptitle(
        'TF Filtering: DESeq2 Significance + ChIP-ATAC Binding Evidence', 
        fontsize=18, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(f'{save_path}/venn_tfs_deseq_binding.pdf', dpi=dpi, bbox_inches='tight')
        fig.savefig(f'{save_path}/venn_tfs_deseq_binding.png', dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # UPSET PLOT
    # =========================================================================
    tf_sets = {
        f'{group1} DESeq2': g1_all,
        f'{group1} + Binding': g1_with_binding,
        f'{group2} DESeq2': g2_all,
        f'{group2} + Binding': g2_with_binding,
    }
    
    upset_data = from_contents(tf_sets)
    
    fig = plt.figure(figsize=figsize_upset)
    upset = UpSet(
        upset_data, 
        subset_size='count', 
        show_counts=True, 
        sort_by='cardinality',
        element_size=46,
        intersection_plot_elements=6
    )
    upset.plot(fig=fig)
    
    # Increase font sizes in UpSet plot
    for ax in fig.axes:
        ax.tick_params(labelsize=12)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
    
    plt.suptitle(
        'TF Candidates: DESeq2 vs ChIP+ATAC Binding Evidence', 
        fontsize=18, fontweight='bold', y=1.02
    )
    
    if save_path:
        fig.savefig(f'{save_path}/upset_tfs_deseq_binding.pdf', dpi=dpi, bbox_inches='tight')
        fig.savefig(f'{save_path}/upset_tfs_deseq_binding.png', dpi=dpi, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    summary = {
        'sets': {
            f'{group1}_all': g1_all,
            f'{group1}_with_binding': g1_with_binding,
            f'{group1}_no_binding': g1_all - g1_with_binding,
            f'{group2}_all': g2_all,
            f'{group2}_with_binding': g2_with_binding,
            f'{group2}_no_binding': g2_all - g2_with_binding,
            'tfs_in_chip_atlas': tfs_in_overlap
        },
        'counts': {
            f'{group1}_deseq_sig': len(g1_all),
            f'{group1}_with_binding': len(g1_with_binding),
            f'{group1}_no_binding': len(g1_all - g1_with_binding),
            f'{group2}_deseq_sig': len(g2_all),
            f'{group2}_with_binding': len(g2_with_binding),
            f'{group2}_no_binding': len(g2_all - g2_with_binding),
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: TF Filtering")
    print("="*70)
    print(f"\n{'Category':<35} {group1+'-high':<15} {group2+'-high':<15}")
    print("-"*70)
    print(f"{'DESeq2 significant':<35} {len(g1_all):<15} {len(g2_all):<15}")
    print(f"{'With ChIP+ATAC binding':<35} {len(g1_with_binding):<15} {len(g2_with_binding):<15}")
    print(f"{'Without binding (for JASPAR)':<35} {len(g1_all - g1_with_binding):<15} {len(g2_all - g2_with_binding):<15}")
    
    print(f"\n{group1}-high TFs to UPREGULATE:")
    print(f"  With binding:    {sorted(g1_with_binding)}")
    print(f"  Without binding: {sorted(g1_all - g1_with_binding)}")
    
    print(f"\n{group2}-high TFs to DOWNREGULATE:")
    print(f"  With binding:    {sorted(g2_with_binding)}")
    print(f"  Without binding: {sorted(g2_all - g2_with_binding)}")
    
    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)
    
    return summary