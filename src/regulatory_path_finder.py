# ==============================================================================
# RegulatoryPathFinder - Trace regulatory paths from master TFs to effector genes
# ==============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict
import warnings

# Biopython for sequence handling
from Bio import SeqIO
from Bio.Seq import Seq

# JASPAR for motif scanning
try:
    from pyjaspar import jaspardb
    JASPAR_AVAILABLE = True
except ImportError:
    JASPAR_AVAILABLE = False
    warnings.warn("pyjaspar not installed. Motif validation will be disabled.")

# For motif scanning
try:
    from Bio import motifs
    BIOPYTHON_MOTIFS = True
except ImportError:
    BIOPYTHON_MOTIFS = False


class RegulatoryPathFinder:
    """
    Find and validate regulatory paths from master TFs to terminal effector genes.
    
    This class can work standalone or with an NSCAnalysis object.
    
    Workflow:
        1. Identify terminal genes (DEGs that are not TFs)
        2. Build regulatory graph from overlap_df
        3. Trace upstream paths using DFS
        4. Filter paths by DEG requirements
        5. Validate edges with ATAC accessibility
        6. Validate edges with JASPAR motifs
        7. Score and rank paths
        8. Output table and network
    """
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(
        self,
        deseq_results: pd.DataFrame,
        overlap_df: pd.DataFrame,
        chip_annotated: pd.DataFrame,
        atac_annotated: pd.DataFrame,
        tf_symbols: Set[str],
        ensembl_to_symbol: Dict[str, str],
        genome_fasta_path: str = None,
        gtf_path: str = None,
        jaspar_species: str = 'Mus musculus'
    ):
        """
        Initialize RegulatoryPathFinder.
        
        Args:
            deseq_results: DESeq2 results DataFrame (index = Ensembl IDs)
            overlap_df: TF-gene binding evidence (ChIP ∩ ATAC)
            chip_annotated: ChIP-seq peaks annotated to genes
            atac_annotated: ATAC-seq peaks annotated to genes
            tf_symbols: Set of TF gene symbols
            ensembl_to_symbol: Ensembl ID → gene symbol mapping
            genome_fasta_path: Path to reference genome FASTA (for motif scanning)
            gtf_path: Path to GTF (for TSS coordinates)
            jaspar_species: Species for JASPAR lookup
        """
        # Store inputs
        self.deseq_results = deseq_results
        self.overlap_df = overlap_df
        self.chip_annotated = chip_annotated
        self.atac_annotated = atac_annotated
        self.tf_symbols = tf_symbols
        self.ensembl_to_symbol = ensembl_to_symbol
        self.symbol_to_ensembl = {v: k for k, v in ensembl_to_symbol.items()}
        self.genome_fasta_path = genome_fasta_path
        self.gtf_path = gtf_path
        self.jaspar_species = jaspar_species
        
        # Build gene symbol column in deseq_results if not present
        if 'symbol' not in self.deseq_results.columns:
            self.deseq_results = self.deseq_results.copy()
            self.deseq_results['symbol'] = self.deseq_results.index.map(self.ensembl_to_symbol)
        
        # Containers (built lazily)
        self.forward_graph = None      # TF → [target genes]
        self.reverse_graph = None      # gene → [upstream TFs]
        self.terminal_genes = None     # Dict of E14/E18 terminal genes
        self.deg_symbols = None        # Set of all DEG symbols
        self.genome = None             # Loaded genome sequences
        self.tss_coords = None         # gene → (chr, tss, strand)
        self.jaspar_db = None          # JASPAR database connection
        self.motif_cache = {}          # TF → motif PWM cache
        
        # Results
        self.all_paths = None          # All discovered paths
        self.validated_paths = None    # Paths with validation scores
        
    @classmethod
    def from_nsc_analysis(
        cls,
        nsc,
        genome_fasta_path: str = None,
        jaspar_species: str = 'Mus musculus'
    ):
        """
        Create RegulatoryPathFinder from an NSCAnalysis object.
        
        Args:
            nsc: NSCAnalysis object (must have run_deseq() already called)
            genome_fasta_path: Path to reference genome FASTA
            jaspar_species: Species for JASPAR lookup
        """
        if nsc.deseq_results is None:
            raise ValueError("Run nsc.run_deseq() before creating RegulatoryPathFinder")
        
        return cls(
            deseq_results=nsc.deseq_results,
            overlap_df=nsc.overlap_df,
            chip_annotated=nsc.chip_annotated,
            atac_annotated=nsc.atac_annotated,
            tf_symbols=nsc.tf_symbols,
            ensembl_to_symbol=nsc.ensembl_to_symbol,
            genome_fasta_path=genome_fasta_path,
            gtf_path=str(nsc.paths['gtf']),
            jaspar_species=jaspar_species
        )
    
    # =========================================================================
    # STEP 1: IDENTIFY TERMINAL GENES
    # =========================================================================
    
    def identify_terminal_genes(
        self,
        group1: str = 'E14',
        group2: str = 'E18',
        padj_thresh: float = 0.05,
        lfc_thresh: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify DEGs that are NOT TFs (terminal effector genes).
        
        Terminal genes = genes that don't regulate other genes in our data
        (i.e., they don't appear as TF in overlap_df)
        
        Args:
            group1, group2: Condition names
            padj_thresh: Adjusted p-value threshold
            lfc_thresh: Log2 fold change threshold
            
        Returns:
            Dict with '{group1}_terminals' and '{group2}_terminals' DataFrames
        """
        print(f"\n{'='*60}")
        print("Step 1: Identifying terminal effector genes")
        print(f"{'='*60}")
        
        # TFs that have binding evidence (appear as regulators in overlap_df)
        tfs_with_targets = set(self.overlap_df['TF'].unique())
        print(f"TFs with known targets in overlap_df: {len(tfs_with_targets)}")
        
        # Get significant DEGs
        sig = self.deseq_results[
            (self.deseq_results['padj'] < padj_thresh) &
            (self.deseq_results['log2FoldChange'].abs() > lfc_thresh)
        ].copy()
        
        print(f"Significant DEGs (padj<{padj_thresh}, |log2FC|>{lfc_thresh}): {len(sig)}")
        
        # Store DEG symbols for later filtering
        self.deg_symbols = set(sig['symbol'].dropna())
        
        # Terminal genes: NOT in overlap_df as a regulator (TF column)
        sig['is_terminal'] = ~sig['symbol'].isin(tfs_with_targets)
        
        # Also flag if it's a TF at all (from TF list)
        sig['is_tf'] = sig['symbol'].isin(self.tf_symbols)
        
        terminals = sig[sig['is_terminal']].copy()
        
        # Split by direction
        # Negative log2FC = higher in group1 (reference)
        # Positive log2FC = higher in group2
        g1_terminals = terminals[terminals['log2FoldChange'] < 0].sort_values('log2FoldChange')
        g2_terminals = terminals[terminals['log2FoldChange'] > 0].sort_values('log2FoldChange', ascending=False)
        
        print(f"\n{group1}-high terminal genes: {len(g1_terminals)}")
        if len(g1_terminals) > 0:
            print(f"  Examples: {list(g1_terminals['symbol'].head(5))}")
        
        print(f"\n{group2}-high terminal genes: {len(g2_terminals)}")
        if len(g2_terminals) > 0:
            print(f"  Examples: {list(g2_terminals['symbol'].head(5))}")
        
        self.terminal_genes = {
            f'{group1}_terminals': g1_terminals,
            f'{group2}_terminals': g2_terminals
        }
        
        return self.terminal_genes
    
    # =========================================================================
    # STEP 2: BUILD REGULATORY GRAPH
    # =========================================================================
    
    def build_regulatory_graph(self) -> None:
        """
        Build forward and reverse regulatory graphs from overlap_df.
        
        forward_graph: TF → [list of genes it regulates]
        reverse_graph: gene → [list of TFs that regulate it]
        """
        print(f"\n{'='*60}")
        print("Step 2: Building regulatory graph")
        print(f"{'='*60}")
        
        # Get unique TF-gene pairs (ignore sample/condition for graph structure)
        tf_gene_pairs = self.overlap_df[['TF', 'gene']].drop_duplicates()
        
        print(f"Unique TF-gene edges: {len(tf_gene_pairs)}")
        
        # Forward graph: TF → targets
        self.forward_graph = defaultdict(list)
        for _, row in tf_gene_pairs.iterrows():
            self.forward_graph[row['TF']].append(row['gene'])
        
        # Reverse graph: gene → regulators (upstream TFs)
        self.reverse_graph = defaultdict(list)
        for _, row in tf_gene_pairs.iterrows():
            self.reverse_graph[row['gene']].append(row['TF'])
        
        # Convert to regular dicts
        self.forward_graph = dict(self.forward_graph)
        self.reverse_graph = dict(self.reverse_graph)
        
        print(f"Forward graph: {len(self.forward_graph)} TFs with targets")
        print(f"Reverse graph: {len(self.reverse_graph)} genes with known regulators")
        
        # Stats
        if self.forward_graph:
            n_targets = [len(v) for v in self.forward_graph.values()]
            n_regulators = [len(v) for v in self.reverse_graph.values()]
            
            print(f"\nTargets per TF: mean={np.mean(n_targets):.1f}, max={max(n_targets)}")
            print(f"Regulators per gene: mean={np.mean(n_regulators):.1f}, max={max(n_regulators)}")
    
    # =========================================================================
    # STEP 3: FIND UPSTREAM PATHS (DFS)
    # =========================================================================
    
    def find_upstream_paths(
        self,
        target_gene: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find all regulatory paths upstream of target_gene using DFS.
        
        Returns:
            List of paths, each path is [master_TF, ..., intermediate_TF, target_gene]
        """
        if self.reverse_graph is None:
            self.build_regulatory_graph()
        
        all_paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            """Recursive DFS to find all upstream paths."""
            upstream_tfs = self.reverse_graph.get(current, [])
            
            if not upstream_tfs or depth >= max_depth:
                if len(path) > 1:
                    all_paths.append(path)  # <-- FIXED: Don't reverse!
                return
            
            for tf in upstream_tfs:
                if tf not in path:
                    dfs(tf, [tf] + path, depth + 1)
        
        dfs(target_gene, [target_gene], 1)
        
        return all_paths
    
    def find_all_paths(
        self,
        group1: str = 'E14',
        group2: str = 'E18',
        max_depth: int = 3,
        max_genes_per_group: int = None
    ) -> Dict[str, List[Dict]]:
        """
        Find upstream paths for all terminal genes in both conditions.
        
        Args:
            group1, group2: Condition names
            max_depth: Maximum path depth
            max_genes_per_group: Limit number of genes to trace (for speed)
            
        Returns:
            Dict with '{group1}_paths' and '{group2}_paths'
        """
        print(f"\n{'='*60}")
        print("Step 3: Finding upstream regulatory paths")
        print(f"{'='*60}")
        
        if self.terminal_genes is None:
            self.identify_terminal_genes(group1, group2)
        
        if self.reverse_graph is None:
            self.build_regulatory_graph()
        
        results = {}
        
        for condition in [group1, group2]:
            key = f'{condition}_terminals'
            terminals_df = self.terminal_genes[key]
            
            genes_to_trace = terminals_df['symbol'].dropna().tolist()
            if max_genes_per_group:
                genes_to_trace = genes_to_trace[:max_genes_per_group]
            
            print(f"\n{condition}: Tracing {len(genes_to_trace)} terminal genes...")
            
            condition_paths = []
            genes_with_paths = 0
            
            for gene in genes_to_trace:
                paths = self.find_upstream_paths(gene, max_depth=max_depth)
                if paths:
                    genes_with_paths += 1
                    for path in paths:
                        condition_paths.append({
                            'target': gene,
                            'path': path,
                            'depth': len(path),
                            'condition': condition
                        })
            
            results[f'{condition}_paths'] = condition_paths
            
            print(f"  Genes with upstream paths: {genes_with_paths}/{len(genes_to_trace)}")
            print(f"  Total paths found: {len(condition_paths)}")
            
            # Path depth distribution
            if condition_paths:
                depths = [p['depth'] for p in condition_paths]
                print(f"  Path depths: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")
        
        self.all_paths = results
        return results
    
    # =========================================================================
    # STEP 4: FILTER PATHS BY DEG REQUIREMENTS
    # =========================================================================
    
    def filter_paths_by_deg(
        self,
        mode: str = 'balanced'
    ) -> Dict[str, List[Dict]]:
        """
        Filter paths based on DEG requirements.
        
        Modes:
            'strict': All nodes in path must be DEGs
            'balanced': Master (first) + target (last) must be DEGs
            'permissive': Target + at least one upstream node must be DEG
            
        Args:
            mode: Filtering stringency
            
        Returns:
            Filtered paths dictionary
        """
        print(f"\n{'='*60}")
        print(f"Step 4: Filtering paths (mode='{mode}')")
        print(f"{'='*60}")
        
        if self.all_paths is None:
            raise ValueError("Run find_all_paths() first")
        
        if self.deg_symbols is None:
            raise ValueError("Run identify_terminal_genes() first")
        
        filtered = {}
        
        for key, paths in self.all_paths.items():
            filtered_paths = []
            
            for path_info in paths:
                path = path_info['path']
                
                if self._check_deg_requirement(path, mode):
                    path_info_copy = path_info.copy()
                    path_info_copy['deg_filter'] = mode
                    filtered_paths.append(path_info_copy)
            
            filtered[key] = filtered_paths
            
            condition = key.split('_')[0]
            print(f"{condition}: {len(filtered_paths)}/{len(paths)} paths passed ({mode})")
        
        return filtered
    
    def _check_deg_requirement(self, path: List[str], mode: str) -> bool:
        """Check if path meets DEG requirement based on mode."""
        if mode == 'strict':
            # All nodes must be DEGs
            return all(node in self.deg_symbols for node in path)
        
        elif mode == 'balanced':
            # Master (first) and target (last) must be DEGs
            master = path[0]
            target = path[-1]
            return (master in self.deg_symbols) and (target in self.deg_symbols)
        
        elif mode == 'permissive':
            # Target + at least one upstream must be DEG
            target = path[-1]
            upstream = path[:-1]
            return (target in self.deg_symbols) and any(node in self.deg_symbols for node in upstream)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # =========================================================================
    # STEP 5: ATAC-BASED EDGE VALIDATION
    # =========================================================================
    
    def validate_edge_accessibility(
        self,
        tf: str,
        target: str,
        condition: str
    ) -> Dict:
        """
        Check if TF→target edge has condition-specific accessibility.
        
        Args:
            tf: Upstream TF
            target: Downstream gene
            condition: 'E14' or 'E18'
            
        Returns:
            Dict with accessibility metrics
        """
        # Get binding evidence for this edge
        edge_data = self.overlap_df[
            (self.overlap_df['TF'] == tf) &
            (self.overlap_df['gene'] == target)
        ]
        
        if len(edge_data) == 0:
            return {
                'has_binding': False,
                'condition_specific': False,
                'atac_signal_ratio': None
            }
        
        # Calculate mean ATAC signal by condition
        signal_by_condition = edge_data.groupby('condition')['atac_signal'].mean()
        
        e14_signal = signal_by_condition.get('E14', 0)
        e18_signal = signal_by_condition.get('E18', 0)
        
        # Condition specificity
        if condition == 'E14':
            is_specific = e14_signal > e18_signal
            ratio = e14_signal / (e18_signal + 0.1)
        else:
            is_specific = e18_signal > e14_signal
            ratio = e18_signal / (e14_signal + 0.1)
        
        return {
            'has_binding': True,
            'condition_specific': is_specific,
            'atac_signal_E14': e14_signal,
            'atac_signal_E18': e18_signal,
            'atac_signal_ratio': ratio
        }
    
    def validate_path_accessibility(
        self,
        path: List[str],
        condition: str
    ) -> Dict:
        """
        Validate all edges in a path for condition-specific accessibility.
        
        Args:
            path: List of genes [master, ..., target]
            condition: 'E14' or 'E18'
            
        Returns:
            Dict with edge validation results
        """
        edge_validations = []
        
        for i in range(len(path) - 1):
            tf = path[i]
            target = path[i + 1]
            
            validation = self.validate_edge_accessibility(tf, target, condition)
            validation['edge'] = f"{tf}→{target}"
            edge_validations.append(validation)
        
        # Summary metrics
        n_edges = len(edge_validations)
        n_with_binding = sum(1 for v in edge_validations if v['has_binding'])
        n_condition_specific = sum(1 for v in edge_validations if v.get('condition_specific', False))
        
        return {
            'edges': edge_validations,
            'n_edges': n_edges,
            'n_with_binding': n_with_binding,
            'n_condition_specific': n_condition_specific,
            'frac_with_binding': n_with_binding / n_edges if n_edges > 0 else 0,
            'frac_condition_specific': n_condition_specific / n_edges if n_edges > 0 else 0
        }
    
    # =========================================================================
    # STEP 6: JASPAR MOTIF VALIDATION
    # =========================================================================
    
    def _init_jaspar(self):
        """Initialize JASPAR database connection."""
        if not JASPAR_AVAILABLE:
            warnings.warn("pyjaspar not available. Motif validation disabled.")
            return False
        
        if self.jaspar_db is None:
            print("Initializing JASPAR database...")
            self.jaspar_db = jaspardb()
        
        return True
    
    def _load_genome(self):
        """Load reference genome FASTA."""
        if self.genome is not None:
            return True
        
        if self.genome_fasta_path is None:
            warnings.warn("No genome FASTA provided. Motif scanning disabled.")
            return False
        
        print(f"Loading genome from {self.genome_fasta_path}...")
        self.genome = SeqIO.to_dict(SeqIO.parse(self.genome_fasta_path, "fasta"))
        print(f"  Loaded {len(self.genome)} chromosomes")
        return True
    
    def _load_tss_coordinates(self):
        """Parse GTF to get TSS coordinates for each gene."""
        if self.tss_coords is not None:
            return True
        
        if self.gtf_path is None:
            warnings.warn("No GTF provided. Cannot get TSS coordinates.")
            return False
        
        print(f"Parsing TSS coordinates from GTF...")
        self.tss_coords = {}
        
        with open(self.gtf_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 9 or fields[2] != 'gene':
                    continue
                
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                
                # Parse gene_name
                gene_name = None
                for attr in fields[8].split(';'):
                    attr = attr.strip()
                    if attr.startswith('gene_name'):
                        gene_name = attr.split('"')[1]
                        break
                
                if gene_name:
                    # TSS is start for + strand, end for - strand
                    tss = start if strand == '+' else end
                    self.tss_coords[gene_name] = (chrom, tss, strand)
        
        print(f"  Loaded TSS for {len(self.tss_coords)} genes")
        return True
    
    def get_promoter_sequence(
        self,
        gene: str,
        upstream: int = 2000,
        downstream: int = 500
    ) -> Optional[str]:
        """
        Extract promoter sequence for a gene.
        
        Args:
            gene: Gene symbol
            upstream: bp upstream of TSS
            downstream: bp downstream of TSS
            
        Returns:
            Promoter sequence string or None
        """
        if not self._load_genome():
            return None
        
        if not self._load_tss_coordinates():
            return None
        
        if gene not in self.tss_coords:
            return None
        
        chrom, tss, strand = self.tss_coords[gene]
        
        # Handle chromosome naming (chr1 vs 1)
        if chrom not in self.genome:
            if chrom.startswith('chr'):
                chrom_alt = chrom[3:]
            else:
                chrom_alt = 'chr' + chrom
            
            if chrom_alt in self.genome:
                chrom = chrom_alt
            else:
                return None
        
        # Get coordinates
        if strand == '+':
            start = max(0, tss - upstream)
            end = tss + downstream
        else:
            start = max(0, tss - downstream)
            end = tss + upstream
        
        # Extract sequence
        seq = str(self.genome[chrom].seq[start:end])
        
        # Reverse complement if on minus strand
        if strand == '-':
            seq = str(Seq(seq).reverse_complement())
        
        return seq.upper()
    
    def get_tf_motif(self, tf: str):
        """
        Get JASPAR motif for a TF.
        
        Args:
            tf: TF gene symbol
            
        Returns:
            JASPAR motif object or None
        """
        if tf in self.motif_cache:
            return self.motif_cache[tf]
        
        if not self._init_jaspar():
            return None
        
        try:
            # Search by name
            results = self.jaspar_db.fetch_motifs(
                collection='CORE',
                species=self.jaspar_species,
                tf_name=tf
            )
            
            if results:
                # Take highest version / most recent
                motif = results[0]
                self.motif_cache[tf] = motif
                return motif
            
        except Exception as e:
            pass
        
        self.motif_cache[tf] = None
        return None
    
    def scan_motif_in_sequence(
        self,
        sequence: str,
        motif,
        threshold_pct: float = 0.80
    ) -> List[Dict]:
        """
        Scan a sequence for motif matches.
        
        Args:
            sequence: DNA sequence string
            motif: JASPAR motif object
            threshold_pct: Minimum score as fraction of max possible score
            
        Returns:
            List of matches with position and score
        """
        if motif is None or sequence is None:
            return []
        
        matches = []
        
        try:
            # Get PWM from JASPAR motif
            pwm = motif.counts.normalize(pseudocounts=0.5)
            pssm = pwm.log_odds()
            
            # Calculate max possible score
            max_score = pssm.max
            min_score = pssm.min
            threshold = min_score + (max_score - min_score) * threshold_pct
            
            # Scan sequence
            for position, score in pssm.search(Seq(sequence), threshold=threshold):
                matches.append({
                    'position': position,
                    'score': score,
                    'relative_score': (score - min_score) / (max_score - min_score),
                    'strand': '+' if position >= 0 else '-'
                })
            
        except Exception as e:
            pass
        
        return matches
    
    def validate_edge_motif(
        self,
        tf: str,
        target: str,
        threshold_pct: float = 0.80
    ) -> Dict:
        """
        Validate TF→target edge by checking if TF's motif is in target's promoter.
        
        Args:
            tf: Upstream TF
            target: Downstream gene
            threshold_pct: Motif score threshold
            
        Returns:
            Dict with motif validation results
        """
        result = {
            'tf': tf,
            'target': target,
            'motif_found': False,
            'n_motif_hits': 0,
            'best_score': None,
            'jaspar_id': None,
            'error': None
        }
        
        # Get TF motif
        motif = self.get_tf_motif(tf)
        if motif is None:
            result['error'] = 'No JASPAR motif found'
            return result
        
        result['jaspar_id'] = motif.matrix_id
        
        # Get target promoter sequence
        sequence = self.get_promoter_sequence(target)
        if sequence is None:
            result['error'] = 'Could not get promoter sequence'
            return result
        
        # Scan for motif
        matches = self.scan_motif_in_sequence(sequence, motif, threshold_pct)
        
        result['n_motif_hits'] = len(matches)
        result['motif_found'] = len(matches) > 0
        
        if matches:
            result['best_score'] = max(m['relative_score'] for m in matches)
            result['matches'] = matches
        
        return result
    
    def validate_path_motifs(
        self,
        path: List[str],
        threshold_pct: float = 0.80
    ) -> Dict:
        """
        Validate all edges in a path using JASPAR motifs.
        
        Args:
            path: List of genes [master, ..., target]
            threshold_pct: Motif score threshold
            
        Returns:
            Dict with motif validation results for all edges
        """
        edge_validations = []
        
        for i in range(len(path) - 1):
            tf = path[i]
            target = path[i + 1]
            
            validation = self.validate_edge_motif(tf, target, threshold_pct)
            validation['edge'] = f"{tf}→{target}"
            edge_validations.append(validation)
        
        # Summary
        n_edges = len(edge_validations)
        n_with_motif = sum(1 for v in edge_validations if v.get('jaspar_id'))
        n_motif_found = sum(1 for v in edge_validations if v.get('motif_found', False))
        
        return {
            'edges': edge_validations,
            'n_edges': n_edges,
            'n_with_jaspar_motif': n_with_motif,
            'n_motif_validated': n_motif_found,
            'frac_with_motif': n_with_motif / n_edges if n_edges > 0 else 0,
            'frac_validated': n_motif_found / n_edges if n_edges > 0 else 0
        }
    
    # =========================================================================
    # STEP 7: SCORE AND RANK PATHS
    # =========================================================================
    
    def score_path(
        self,
        path: List[str],
        condition: str,
        include_motif: bool = True
    ) -> Dict:
        """
        Calculate comprehensive score for a regulatory path.
        
        Args:
            path: List of genes [master, ..., target]
            condition: 'E14' or 'E18'
            include_motif: Whether to include JASPAR validation
            
        Returns:
            Dict with all scores and validations
        """
        path_str = ' → '.join(path)
        
        result = {
            'path': path,
            'path_str': path_str,
            'condition': condition,
            'depth': len(path),
            'master': path[0],
            'target': path[-1]
        }
        
        # DEG status for each node
        result['deg_status'] = [gene in self.deg_symbols for gene in path]
        result['n_degs'] = sum(result['deg_status'])
        result['all_degs'] = all(result['deg_status'])
        
        # Get expression info for master and target
        for label, gene in [('master', path[0]), ('target', path[-1])]:
            if gene in self.symbol_to_ensembl:
                ens_id = self.symbol_to_ensembl[gene]
                if ens_id in self.deseq_results.index:
                    row = self.deseq_results.loc[ens_id]
                    result[f'{label}_log2fc'] = row['log2FoldChange']
                    result[f'{label}_padj'] = row['padj']
        
        # ATAC validation
        atac_validation = self.validate_path_accessibility(path, condition)
        result['atac_validation'] = atac_validation
        result['atac_score'] = atac_validation['frac_condition_specific']
        
        # Motif validation (optional - slow)
        if include_motif and self.genome_fasta_path:
            motif_validation = self.validate_path_motifs(path)
            result['motif_validation'] = motif_validation
            result['motif_score'] = motif_validation['frac_validated']
        else:
            result['motif_score'] = None
        
        # Combined score
        scores = [result['atac_score']]
        if result['motif_score'] is not None:
            scores.append(result['motif_score'])
        
        result['combined_score'] = np.mean(scores)
        
        return result
    
    def score_all_paths(
        self,
        filtered_paths: Dict[str, List[Dict]] = None,
        include_motif: bool = False,
        max_paths_per_condition: int = None
    ) -> pd.DataFrame:
        """
        Score all paths and return as DataFrame.
        
        Args:
            filtered_paths: Output from filter_paths_by_deg()
            include_motif: Include JASPAR validation (slower)
            max_paths_per_condition: Limit paths to score
            
        Returns:
            DataFrame with scored paths
        """
        print(f"\n{'='*60}")
        print("Step 5-7: Scoring and validating paths")
        print(f"{'='*60}")
        
        if filtered_paths is None:
            filtered_paths = self.filter_paths_by_deg(mode='balanced')
        
        all_scored = []
        
        for key, paths in filtered_paths.items():
            condition = key.split('_')[0]
            
            if max_paths_per_condition:
                paths = paths[:max_paths_per_condition]
            
            print(f"\n{condition}: Scoring {len(paths)} paths...")
            
            for i, path_info in enumerate(paths):
                if (i + 1) % 100 == 0:
                    print(f"  Scored {i+1}/{len(paths)} paths...")
                
                scored = self.score_path(
                    path_info['path'],
                    condition,
                    include_motif=include_motif
                )
                all_scored.append(scored)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_scored)
        
        # Sort by score
        if len(df) > 0:
            df = df.sort_values('combined_score', ascending=False)
        
        self.validated_paths = df
        
        print(f"\nTotal scored paths: {len(df)}")
        
        return df
    
    # =========================================================================
    # STEP 8: OUTPUT - TABLE
    # =========================================================================
    
    def get_summary_table(
        self,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get summary table of top paths.
        
        Args:
            top_n: Number of top paths to return
            
        Returns:
            Summary DataFrame
        """
        if self.validated_paths is None:
            raise ValueError("Run score_all_paths() first")
        
        df = self.validated_paths.head(top_n).copy()
        
        # Select key columns
        cols = [
            'path_str', 'condition', 'depth',
            'master', 'target',
            'master_log2fc', 'target_log2fc',
            'n_degs', 'all_degs',
            'atac_score', 'motif_score', 'combined_score'
        ]
        
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols]
    
    def export_paths(self, output_path: str):
        """Export all validated paths to TSV."""
        if self.validated_paths is None:
            raise ValueError("Run score_all_paths() first")
        
        # Select serializable columns only
        export_cols = [
            'path_str', 'condition', 'depth', 'master', 'target',
            'n_degs', 'all_degs', 'atac_score', 'motif_score', 'combined_score'
        ]
        
        # Add expression columns if present
        for col in ['master_log2fc', 'target_log2fc', 'master_padj', 'target_padj']:
            if col in self.validated_paths.columns:
                export_cols.append(col)
        
        available_cols = [c for c in export_cols if c in self.validated_paths.columns]
        
        self.validated_paths[available_cols].to_csv(output_path, sep='\t', index=False)
        print(f"Exported {len(self.validated_paths)} paths to {output_path}")
    
    # =========================================================================
    # STEP 9: OUTPUT - NETWORK
    # =========================================================================
    
    def build_network_edges(
        self,
        min_score: float = 0.0
    ) -> pd.DataFrame:
        """
        Build edge list for network visualization.
        
        Args:
            min_score: Minimum combined score to include
            
        Returns:
            DataFrame with source, target, and edge attributes
        """
        if self.validated_paths is None:
            raise ValueError("Run score_all_paths() first")
        
        edges = []
        
        for _, row in self.validated_paths.iterrows():
            if row['combined_score'] < min_score:
                continue
            
            path = row['path']
            condition = row['condition']
            
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                edge = {
                    'source': source,
                    'target': target,
                    'condition': condition,
                    'path_score': row['combined_score']
                }
                
                # Add ATAC info if available
                atac_val = row.get('atac_validation')
                if atac_val and isinstance(atac_val, dict) and 'edges' in atac_val:
                    for e in atac_val['edges']:
                        if e['edge'] == f"{source}→{target}":
                            edge['has_binding'] = e.get('has_binding', False)
                            edge['condition_specific'] = e.get('condition_specific', False)
                            break
                
                edges.append(edge)
        
        # Deduplicate edges
        df = pd.DataFrame(edges)
        if len(df) > 0:
            df = df.drop_duplicates(subset=['source', 'target', 'condition'])
        
        return df
    
    def build_network_nodes(self) -> pd.DataFrame:
        """
        Build node list for network visualization.
        
        Returns:
            DataFrame with node attributes
        """
        if self.validated_paths is None:
            raise ValueError("Run score_all_paths() first")
        
        # Collect all nodes
        all_nodes = set()
        for _, row in self.validated_paths.iterrows():
            all_nodes.update(row['path'])
        
        nodes = []
        for gene in all_nodes:
            node = {
                'gene': gene,
                'is_tf': gene in self.tf_symbols,
                'is_deg': gene in self.deg_symbols if self.deg_symbols else False
            }
            
            # Add expression info
            if gene in self.symbol_to_ensembl:
                ens_id = self.symbol_to_ensembl[gene]
                if ens_id in self.deseq_results.index:
                    row = self.deseq_results.loc[ens_id]
                    node['log2fc'] = row['log2FoldChange']
                    node['padj'] = row['padj']
                    
                    if row['log2FoldChange'] < -1:
                        node['direction'] = 'E14_high'
                    elif row['log2FoldChange'] > 1:
                        node['direction'] = 'E18_high'
                    else:
                        node['direction'] = 'neutral'
            
            # Check if master or terminal
            node['is_master'] = any(
                gene == r['master'] 
                for _, r in self.validated_paths.iterrows()
            )
            node['is_terminal'] = any(
                gene == r['target'] 
                for _, r in self.validated_paths.iterrows()
            )
            
            nodes.append(node)
        
        return pd.DataFrame(nodes)
    
    def export_network(
        self,
        output_prefix: str,
        min_score: float = 0.0
    ):
        """
        Export network files for visualization (Cytoscape, etc.)
        
        Args:
            output_prefix: Prefix for output files
            min_score: Minimum score threshold
        """
        edges_df = self.build_network_edges(min_score=min_score)
        nodes_df = self.build_network_nodes()
        
        edges_path = f"{output_prefix}_edges.tsv"
        nodes_path = f"{output_prefix}_nodes.tsv"
        
        edges_df.to_csv(edges_path, sep='\t', index=False)
        nodes_df.to_csv(nodes_path, sep='\t', index=False)
        
        print(f"Exported network:")
        print(f"  Edges: {len(edges_df)} → {edges_path}")
        print(f"  Nodes: {len(nodes_df)} → {nodes_path}")
    
    # =========================================================================
    # CONVENIENCE: RUN FULL PIPELINE
    # =========================================================================
    
    def run(
        self,
        group1: str = 'E14',
        group2: str = 'E18',
        padj_thresh: float = 0.05,
        lfc_thresh: float = 1.0,
        max_depth: int = 3,
        deg_filter_mode: str = 'balanced',
        include_motif: bool = False,
        max_paths: int = 1000,
        output_prefix: str = None
    ) -> pd.DataFrame:
        """
        Run the full regulatory path finding pipeline.
        
        Args:
            group1, group2: Condition names
            padj_thresh: DEG significance threshold
            lfc_thresh: Log2FC threshold
            max_depth: Maximum path depth
            deg_filter_mode: 'strict', 'balanced', or 'permissive'
            include_motif: Include JASPAR validation
            max_paths: Maximum paths to score
            output_prefix: If provided, export results
            
        Returns:
            DataFrame with scored paths
        """
        print("\n" + "="*70)
        print("REGULATORY PATH FINDER - FULL PIPELINE")
        print("="*70)
        
        # Step 1: Identify terminal genes
        self.identify_terminal_genes(
            group1=group1,
            group2=group2,
            padj_thresh=padj_thresh,
            lfc_thresh=lfc_thresh
        )
        
        # Step 2: Build graph
        self.build_regulatory_graph()
        
        # Step 3: Find paths
        self.find_all_paths(
            group1=group1,
            group2=group2,
            max_depth=max_depth
        )
        
        # Step 4: Filter by DEG
        filtered = self.filter_paths_by_deg(mode=deg_filter_mode)
        
        # Steps 5-7: Score paths
        results = self.score_all_paths(
            filtered_paths=filtered,
            include_motif=include_motif,
            max_paths_per_condition=max_paths
        )
        
        # Export if requested
        if output_prefix:
            self.export_paths(f"{output_prefix}_paths.tsv")
            self.export_network(output_prefix)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
        return results