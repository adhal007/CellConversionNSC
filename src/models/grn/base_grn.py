import pandas as pd
from celloracle import motif_analysis as ma
import genomepy
import os 
import pyranges as pr
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pybedtools
from tqdm import tqdm

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


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


class GRNBuilder:
    """
    Build condition-specific GRNs using:
    - TF-target prior (chip_all_mouse)
    - RNA-seq DE (binary indicators)
    - DA peaks (binary indicators)
    - ChIP peaks (ReMap binding evidence)
    - Regulatory regions (enhancers + promoters)
    """
    
    def __init__(self, deseq_results, consensus_peaks, dar_results, 
                merged_regulatory, remap_peaks, tf_target_df, output_dir,
                group1='E14', group2='E18'):  # ADD THIS
        """
        Initialize GRN builder.
        
        Args:
            deseq_results: DESeq2 results with columns [symbol, log2FoldChange, padj, is_TF]
            consensus_peaks: ATAC consensus peaks
            dar_results: DA peaks with columns [seqnames, start, end, Fold]
            merged_regulatory: Merged regulatory regions [chr, start, end, gene, region_type]
            remap_peaks: ReMap peaks [chr, start, end, TF]
            tf_target_df: TF-target prior network [TF, Target]
            output_dir: Output directory
            group1: str, default='E14' - Reference group name  # ADD
            group2: str, default='E18' - Comparison group name  # ADD
        """
        self.deseq_results = deseq_results
        self.consensus_peaks = consensus_peaks
        self.dar_results = dar_results
        self.merged_regulatory = merged_regulatory
        self.remap_peaks = remap_peaks
        self.tf_target_df = tf_target_df
        self.output_dir = output_dir
        self.group1 = group1  # ADD
        self.group2 = group2  # ADD
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("GRN Builder Initialized")
        print("="*80)
        print(f"DE results: {len(deseq_results)} genes")
        print(f"Regulatory regions: {len(merged_regulatory)}")
        print(f"ReMap peaks: {len(remap_peaks)}")
        print(f"TF-target prior: {len(tf_target_df)} edges")
    
    def compute_de_indicators(self, tau_lfc=1.0, tau_padj=0.05):
        """
        Step 1: Compute binary DE indicators E_x(c).
        
        Returns:
            dict, dict: Gene -> binary DE indicator for each group
        """
        print("\n" + "="*80)
        print("STEP 1: BINARY DE INDICATORS")
        print("="*80)
        
        # group1: positive logFC
        self.deseq_results[f'E_{self.group1}'] = (
            (self.deseq_results['log2FoldChange'] > tau_lfc) & 
            (self.deseq_results['padj'] < tau_padj)
        ).astype(int)
        
        # group2: negative logFC
        self.deseq_results[f'E_{self.group2}'] = (
            (self.deseq_results['log2FoldChange'] < -tau_lfc) & 
            (self.deseq_results['padj'] < tau_padj)
        ).astype(int)
        
        print(f"{self.group1} DE genes: {self.deseq_results[f'E_{self.group1}'].sum()}")
        print(f"{self.group2} DE genes: {self.deseq_results[f'E_{self.group2}'].sum()}")
        
        # Create lookup dictionaries with dynamic attribute names
        setattr(self, f'E_{self.group1}_dict', self.deseq_results.set_index('symbol')[f'E_{self.group1}'].to_dict())
        setattr(self, f'E_{self.group2}_dict', self.deseq_results.set_index('symbol')[f'E_{self.group2}'].to_dict())
        
        # Store DE gene lists
        e14_cond = (self.deseq_results['padj'] < tau_padj) & (self.deseq_results['log2FoldChange'] > tau_lfc)
        e18_cond = (self.deseq_results['padj'] < tau_padj) & (self.deseq_results['log2FoldChange'] < -tau_lfc)
        
        self.de_e14_tfs = self.deseq_results[e14_cond & self.deseq_results['is_TF']]['symbol'].to_numpy()
        self.de_e18_tfs = self.deseq_results[e18_cond & self.deseq_results['is_TF']]['symbol'].to_numpy()
        self.de_e14_genes = self.deseq_results[e14_cond]['symbol'].to_numpy()
        self.de_e18_genes = self.deseq_results[e18_cond]['symbol'].to_numpy()
        
        # Return using getattr to get the stored attributes
        return getattr(self, f'E_{self.group1}_dict'), getattr(self, f'E_{self.group2}_dict')
    
    def compute_da_indicators(self):
        """
        Step 2: Compute binary DA indicators D_r(c) for each regulatory element.
        
        Returns:
            merged_regulatory with D_{group1}, D_{group2} columns
        """
        print("\n" + "="*80)
        print("STEP 2: BINARY DA INDICATORS PER ELEMENT")
        print("="*80)
        
        print(f"Merged regulatory elements: {len(self.merged_regulatory)}")
        print(f"DA peaks: {len(self.dar_results)}")
        
        # Overlap merged_regulatory with dar_results
        reg_bed = pybedtools.BedTool.from_dataframe(
            self.merged_regulatory[['chr', 'start', 'end', 'gene', 'region_type']]
        )
        
        dar_bed = pybedtools.BedTool.from_dataframe(
            self.dar_results[['seqnames', 'start', 'end', 'Fold']].rename(columns={'seqnames': 'chr'})
        )
        
        reg_da_overlap = reg_bed.intersect(dar_bed, wa=True, wb=True)
        
        print(f"Computing DA overlap...")
        
        # Parse overlaps
        element_da_status = {}
        for interval in tqdm(reg_da_overlap):
            element_key = (interval[0], int(interval[1]), int(interval[2]), interval[3])
            da_fold = float(interval[8])
            
            # Keep max absolute fold change
            if element_key not in element_da_status:
                element_da_status[element_key] = da_fold
            else:
                if abs(da_fold) > abs(element_da_status[element_key]):
                    element_da_status[element_key] = da_fold
        
        print(f"Elements with DA overlap: {len(element_da_status)}")
        
        # Add to merged_regulatory
        self.merged_regulatory['element_key_tuple'] = list(zip(
            self.merged_regulatory['chr'],
            self.merged_regulatory['start'],
            self.merged_regulatory['end'],
            self.merged_regulatory['gene']
        ))
        
        # Create UNIQUE string element_key with gene included
        self.merged_regulatory['element_key'] = (
            self.merged_regulatory['chr'].astype(str) + ':' +
            self.merged_regulatory['start'].astype(str) + '-' +
            self.merged_regulatory['end'].astype(str) + '_' +
            self.merged_regulatory['gene'].astype(str)
        )
        
        self.merged_regulatory['da_fold'] = self.merged_regulatory['element_key_tuple'].map(
            lambda k: element_da_status.get(k, 0)
        )
        
        # Binary indicators - CHANGED
        self.merged_regulatory[f'D_{self.group1}'] = (self.merged_regulatory['da_fold'] > 0).astype(int)
        self.merged_regulatory[f'D_{self.group2}'] = (self.merged_regulatory['da_fold'] < 0).astype(int)
        
        print(f"\nResults:")
        print(f"  Elements with DA: {(self.merged_regulatory['da_fold'] != 0).sum()}")
        print(f"  Elements with {self.group1} DA: {self.merged_regulatory[f'D_{self.group1}'].sum()}")  # CHANGED
        print(f"  Elements with {self.group2} DA: {self.merged_regulatory[f'D_{self.group2}'].sum()}")  # CHANGED
        
        return self.merged_regulatory
    
    def compute_chip_binding(self):
        """
        Step 3: Compute TF-element binding B_{t,r} from ReMap.
        
        Returns:
            chip_overlap_df with columns [TF, element_key]
        """
        print("\n" + "="*80)
        print("STEP 3: TF-ELEMENT BINDING B_{t,r} (ReMap)")
        print("="*80)
        
        # element_key already exists from compute_da_indicators()
        remap_bed = pybedtools.BedTool.from_dataframe(
            self.remap_peaks[['chr', 'start', 'end', 'TF']]
        )
        
        regulatory_bed = pybedtools.BedTool.from_dataframe(
            self.merged_regulatory[['chr', 'start', 'end', 'element_key']]
        )
        
        # Intersect: regulatory elements first
        chip_overlap = regulatory_bed.intersect(remap_bed, wa=True, wb=True)
        
        # Convert to DataFrame
        self.chip_overlap_df = chip_overlap.to_dataframe(
            names=['chr_el', 'start_el', 'end_el', 'element_key',
                'chr_tf', 'start_tf', 'end_tf', 'TF']
        )
        
        # Canonicalize TF names
        self.chip_overlap_df['TF'] = self.chip_overlap_df['TF'].str.upper()
        
        print(f"TF-element overlaps: {len(self.chip_overlap_df):,}")
        print(f"Unique TFs with binding: {self.chip_overlap_df['TF'].nunique()}")
        print(f"Unique elements bound: {self.chip_overlap_df['element_key'].nunique()}")
        
        return self.chip_overlap_df

    def create_atac_signal_matrix(self, all_files):
        """
        Create signal matrix: regulatory elements x samples.
        
        For each regulatory element, find overlapping peaks and take max signal.
        If no overlap, signal = 0.
        
        Returns:
            signal_matrix: DataFrame with samples as rows, elements as columns
        """
        signal_matrix = []
        
        for PEAK_FILE in all_files:
            peaks = pd.read_csv(PEAK_FILE, sep='\t', header=None,
                            names=['chr', 'start', 'end', 'name', 'score', 
                                    'strand', 'signalValue', 'pValue', 'qValue', 'peak'])
            sample_id = PEAK_FILE.split('/')[-1].split('_')[0]
            reg_df = self.merged_regulatory
            
            # Overlap: reg_bed with peaks_bed (regulatory regions first)
            reg_bed = pybedtools.BedTool.from_dataframe(reg_df[['chr', 'start', 'end', 'element_key']])
            peaks_bed = pybedtools.BedTool.from_dataframe(peaks[['chr', 'start', 'end', 'signalValue']])
            
            overlap = reg_bed.intersect(peaks_bed, wa=True, wb=True)
            
            # Collect signals for this sample (take max if multiple peaks overlap one element)
            sample_signals = {}
            for interval in overlap:
                element_key = interval[3]
                signal = float(interval[7])  # signalValue from peak
                
                if element_key not in sample_signals:
                    sample_signals[element_key] = signal
                else:
                    sample_signals[element_key] = max(sample_signals[element_key], signal)
            
            # Create complete dataframe with all elements
            sample_signal_df = pd.DataFrame({
                'element_key': reg_df['element_key'].tolist(),
                'signal_value': [sample_signals.get(k, 0.0) for k in reg_df['element_key']],
                'sample_id': sample_id
            })
            
            # Pivot to wide format
            wide_df = sample_signal_df.pivot(
                index='sample_id', 
                columns='element_key', 
                values='signal_value'
            )
            signal_matrix.append(wide_df)
        
        signal_matrix_df = pd.concat(signal_matrix)
        return signal_matrix_df
    
    
    def compute_mad_normalized_accessibility(self, atac_meta, peak_dir, 
                                            group1_samples, group2_samples,
                                            plot=True):
        """
        Compute MAD-normalized accessibility scores for regulatory elements.
        
        Steps:
        1. Create signal matrix from ATAC peak files
        2. MAD normalize per element using global distribution
        3. Compute mean MAD score per condition
        4. Merge with merged_regulatory
        
        Args:
            atac_meta: DataFrame with ATAC metadata
            peak_dir: Directory containing narrowPeak files
            group1_samples: List of sample IDs for group1 (e.g., CD133 E14 Cortex)
            group2_samples: List of sample IDs for group2 (e.g., CD133 E18 Cortex)
            plot: Whether to plot MAD score distributions
            
        Returns:
            merged_regulatory with {group1}_mean_mad_score and {group2}_mean_mad_score columns
        """
        import scipy.stats as stats
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        
        print("\n" + "="*80)
        print("COMPUTING MAD-NORMALIZED ACCESSIBILITY")
        print("="*80)
        
        # Get all peak files for global distribution
        all_samples = group1_samples + group2_samples
        all_peak_files = [os.path.join(peak_dir, f"{s}_peaks.narrowPeak") for s in all_samples]
        all_peak_files = [f for f in all_peak_files if os.path.exists(f)]
        
        print(f"\n{self.group1} samples: {len(group1_samples)}")
        print(f"{self.group2} samples: {len(group2_samples)}")
        print(f"Total peak files: {len(all_peak_files)}")
        
        # Create signal matrix
        print("\nCreating ATAC signal matrix...")
        signal_matrix = self.create_atac_signal_matrix(all_peak_files)
        
        # Remove all-zero columns
        signal_matrix = signal_matrix.loc[:, (signal_matrix != 0).any()]
        print(f"Signal matrix shape: {signal_matrix.shape}")
        
        # Step 1: Compute per-element (per column) median and MAD
        # Use ALL samples to get global distribution per element
        print("\nComputing per-element median and MAD...")
        median_per_element = signal_matrix.median(axis=0)
        mad_per_element = signal_matrix.apply(
            lambda col: stats.median_abs_deviation(col, nan_policy='omit'), 
            axis=0
        )
        
        # Step 2: MAD normalize per element
        # (value - median) / (1.4826 * MAD)
        # 1.4826 makes MAD comparable to std for normal distribution
        print("MAD normalizing...")
        signal_matrix_mad_norm = (signal_matrix - median_per_element) / (1.4826 * mad_per_element)
        
        # Handle cases where MAD = 0 (constant signal)
        signal_matrix_mad_norm = signal_matrix_mad_norm.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Step 3: Compute mean MAD-normalized score per condition
        print(f"\nComputing mean MAD scores per condition...")
        group1_z = signal_matrix_mad_norm[
            signal_matrix_mad_norm.index.isin(group1_samples)
        ].mean(axis=0).reset_index()
        
        group2_z = signal_matrix_mad_norm[
            signal_matrix_mad_norm.index.isin(group2_samples)
        ].mean(axis=0).reset_index()
        
        # Set column names dynamically
        group1_z.columns = ['element_key', f'{self.group1}_mean_mad_score']
        group2_z.columns = ['element_key', f'{self.group2}_mean_mad_score']
        
        # Merge with merged_regulatory
        print("\nMerging with regulatory regions...")
        self.merged_regulatory = pd.merge(
            self.merged_regulatory, 
            group1_z, 
            on='element_key', 
            how='left'
        )
        self.merged_regulatory = pd.merge(
            self.merged_regulatory, 
            group2_z, 
            on='element_key', 
            how='left'
        )
        
        # Fill NaN with 0 (elements not in ATAC peaks)
        self.merged_regulatory[f'{self.group1}_mean_mad_score'] = \
            self.merged_regulatory[f'{self.group1}_mean_mad_score'].fillna(0)
        self.merged_regulatory[f'{self.group2}_mean_mad_score'] = \
            self.merged_regulatory[f'{self.group2}_mean_mad_score'].fillna(0)
        
        # Print stats
        print(f"\n{self.group1} MAD score range: "
            f"[{group1_z[f'{self.group1}_mean_mad_score'].min():.2f}, "
            f"{group1_z[f'{self.group1}_mean_mad_score'].max():.2f}]")
        print(f"{self.group2} MAD score range: "
            f"[{group2_z[f'{self.group2}_mean_mad_score'].min():.2f}, "
            f"{group2_z[f'{self.group2}_mean_mad_score'].max():.2f}]")
        
        # Calculate Tukey thresholds
        group1_col = f'{self.group1}_mean_mad_score'
        group2_col = f'{self.group2}_mean_mad_score'
        
        iqr1 = np.percentile(self.merged_regulatory[group1_col], 75) - \
            np.percentile(self.merged_regulatory[group1_col], 25)
        tukey_thresh1 = np.percentile(self.merged_regulatory[group1_col], 75) + 1.5 * iqr1
        
        iqr2 = np.percentile(self.merged_regulatory[group2_col], 75) - \
            np.percentile(self.merged_regulatory[group2_col], 25)
        tukey_thresh2 = np.percentile(self.merged_regulatory[group2_col], 75) + 1.5 * iqr2
        
        print(f"\nTukey outlier thresholds (Q3 + 1.5*IQR):")
        print(f"  {self.group1}: {tukey_thresh1:.2f}")
        print(f"  {self.group2}: {tukey_thresh2:.2f}")
        
        # Store thresholds
        self.mad_tukey_threshold_group1 = tukey_thresh1
        self.mad_tukey_threshold_group2 = tukey_thresh2
        
        # Plot distribution
        if plot:
            plt.figure(figsize=(10, 6))
            plt.hist(
                group1_z[f'{self.group1}_mean_mad_score'], 
                bins=50, alpha=0.5, label=self.group1, density=True
            )
            plt.hist(
                group2_z[f'{self.group2}_mean_mad_score'], 
                bins=50, alpha=0.5, label=self.group2, density=True
            )
            plt.xlabel('MAD Normalized Score')
            plt.ylabel('Density')
            plt.legend()
            plt.title('Per-Element MAD Normalized Accessibility')
            plt.axvline(tukey_thresh1, color='blue', linestyle='--', alpha=0.5, 
                    label=f'{self.group1} Tukey threshold')
            plt.axvline(tukey_thresh2, color='orange', linestyle='--', alpha=0.5,
                    label=f'{self.group2} Tukey threshold')
            plt.legend()
            plt.show()
        
        print("\n" + "="*80)
        print("MAD NORMALIZATION COMPLETE")
        print("="*80)
        print(f"Added columns: {group1_col}, {group2_col}")
        
        return self.merged_regulatory


    def build_condition_grns(self, zscore_threshold_group1=2.0, zscore_threshold_group2=2.0):
        """
        Step 4: Build condition-specific GRNs with expanded regulatory element selection.
        
        Uses OR logic for regulatory elements:
        - DA status (D_{group1} == 1 or D_{group2} == 1), OR
        - High accessibility ({group1}_mean_mad_score > threshold or {group2}_mean_mad_score > threshold)
        
        Args:
            zscore_threshold_group1: Minimum MAD score for group1 accessibility (default: 2.0)
            zscore_threshold_group2: Minimum MAD score for group2 accessibility (default: 2.0)
        
        Returns:
            grn_group1, grn_group2: Condition-specific GRNs
        """
        print("\n" + "=" * 80)
        print("STEP 4: BUILD CONDITION-SPECIFIC GRNs (DA OR Z-SCORE)")
        print("=" * 80)
        print(f"{self.group1} MAD score threshold: {zscore_threshold_group1}")
        print(f"{self.group2} MAD score threshold: {zscore_threshold_group2}")
        
        # Check if z-scores are computed
        group1_zscore_col = f'{self.group1}_mean_mad_score'
        group2_zscore_col = f'{self.group2}_mean_mad_score'
        
        if group1_zscore_col not in self.merged_regulatory.columns:
            print("\nWARNING: MAD scores not found. Using DA status only.")
            use_zscores = False
        else:
            use_zscores = True
        
        # ------------------
        # Group1 GRN
        # ------------------
        print("\n" + "-" * 80)
        print(f"Building {self.group1} GRN")
        print("-" * 80)
        
        # Get DE TFs with ChIP binding
        group1_chip_df = self.chip_overlap_df[
            self.chip_overlap_df['TF'].isin(
                [tf.upper() for tf in self.de_e14_tfs]
            )
        ].reset_index(drop=True)
        
        print(f"{self.group1} DE TFs with ChIP: {group1_chip_df['TF'].nunique()}")
        
        # Get regulatory elements for DE genes
        group1_DA_de_df = self.merged_regulatory[
            self.merged_regulatory['gene'].isin(self.de_e14_genes)
        ].reset_index(drop=True)
        
        print(f"Regulatory elements for {self.group1} DE genes: {len(group1_DA_de_df)}")
        
        # Apply OR logic: DA status OR high MAD score
        group1_da_col = f'D_{self.group1}'
        
        if use_zscores:
            group1_active_elements = group1_DA_de_df[
                (group1_DA_de_df[group1_da_col] == 1) |  # DA in group1
                (group1_DA_de_df[group1_zscore_col] > zscore_threshold_group1)  # OR high accessibility
            ].reset_index(drop=True)
            
            # Statistics
            da_only = group1_DA_de_df[
                (group1_DA_de_df[group1_da_col] == 1) & 
                (group1_DA_de_df[group1_zscore_col] <= zscore_threshold_group1)
            ]
            zscore_only = group1_DA_de_df[
                (group1_DA_de_df[group1_da_col] == 0) & 
                (group1_DA_de_df[group1_zscore_col] > zscore_threshold_group1)
            ]
            both = group1_DA_de_df[
                (group1_DA_de_df[group1_da_col] == 1) & 
                (group1_DA_de_df[group1_zscore_col] > zscore_threshold_group1)
            ]
            
            print(f"\n{self.group1} Element Selection (OR logic):")
            print(f"  DA only: {len(da_only)}")
            print(f"  MAD score only: {len(zscore_only)}")
            print(f"  Both DA and MAD score: {len(both)}")
            print(f"  Total active elements: {len(group1_active_elements)}")
            
        else:
            # Fall back to DA only
            group1_active_elements = group1_DA_de_df[
                group1_DA_de_df[group1_da_col] == 1
            ].reset_index(drop=True)
            print(f"Active elements (DA only): {len(group1_active_elements)}")
        
        # Merge: TF ChIP + active element + DE gene
        group1_tf_target_de_da = pd.merge(
            group1_chip_df,
            group1_active_elements,
            on='element_key'
        )
        
        print(f"\n{self.group1} TF→gene pairs (ChIP + Active + DE): {len(group1_tf_target_de_da)}")
        
        # TF → gene edges
        grn_group1_all = group1_tf_target_de_da[['TF', 'gene']].drop_duplicates()
        
        # Remove autoregulation
        self.grn_e14 = grn_group1_all[
            grn_group1_all['TF'] != grn_group1_all['gene']
        ].reset_index(drop=True)
        
        # ------------------
        # Group2 GRN
        # ------------------
        print("\n" + "-" * 80)
        print(f"Building {self.group2} GRN")
        print("-" * 80)
        
        # Get DE TFs with ChIP binding
        group2_chip_df = self.chip_overlap_df[
            self.chip_overlap_df['TF'].isin(
                [tf.upper() for tf in self.de_e18_tfs]
            )
        ].reset_index(drop=True)
        
        print(f"{self.group2} DE TFs with ChIP: {group2_chip_df['TF'].nunique()}")
        
        # Get regulatory elements for DE genes
        group2_DA_de_df = self.merged_regulatory[
            self.merged_regulatory['gene'].isin(self.de_e18_genes)
        ].reset_index(drop=True)
        
        print(f"Regulatory elements for {self.group2} DE genes: {len(group2_DA_de_df)}")
        
        # Apply OR logic: DA status OR high MAD score
        group2_da_col = f'D_{self.group2}'
        
        if use_zscores:
            group2_active_elements = group2_DA_de_df[
                (group2_DA_de_df[group2_da_col] == 1) |  # DA in group2
                (group2_DA_de_df[group2_zscore_col] > zscore_threshold_group2)  # OR high accessibility
            ].reset_index(drop=True)
            
            # Statistics
            da_only = group2_DA_de_df[
                (group2_DA_de_df[group2_da_col] == 1) & 
                (group2_DA_de_df[group2_zscore_col] <= zscore_threshold_group2)
            ]
            zscore_only = group2_DA_de_df[
                (group2_DA_de_df[group2_da_col] == 0) & 
                (group2_DA_de_df[group2_zscore_col] > zscore_threshold_group2)
            ]
            both = group2_DA_de_df[
                (group2_DA_de_df[group2_da_col] == 1) & 
                (group2_DA_de_df[group2_zscore_col] > zscore_threshold_group2)
            ]
            
            print(f"\n{self.group2} Element Selection (OR logic):")
            print(f"  DA only: {len(da_only)}")
            print(f"  MAD score only: {len(zscore_only)}")
            print(f"  Both DA and MAD score: {len(both)}")
            print(f"  Total active elements: {len(group2_active_elements)}")
            
        else:
            # Fall back to DA only
            group2_active_elements = group2_DA_de_df[
                group2_DA_de_df[group2_da_col] == 1
            ].reset_index(drop=True)
            print(f"Active elements (DA only): {len(group2_active_elements)}")
        
        # Merge: TF ChIP + active element + DE gene
        group2_tf_target_de_da = pd.merge(
            group2_chip_df,
            group2_active_elements,
            on='element_key'
        )
        
        print(f"\n{self.group2} TF→gene pairs (ChIP + Active + DE): {len(group2_tf_target_de_da)}")
        
        # TF → gene edges
        grn_group2_all = group2_tf_target_de_da[['TF', 'gene']].drop_duplicates()
        
        # Remove autoregulation
        self.grn_e18 = grn_group2_all[
            grn_group2_all['TF'] != grn_group2_all['gene']
        ].reset_index(drop=True)
        
        # ------------------
        # Summary
        # ------------------
        print("\n" + "=" * 80)
        print("FINAL GRN SUMMARY")
        print("=" * 80)
        print(
            f"{self.group1} GRN: {len(self.grn_e14)} edges, "
            f"{self.grn_e14['TF'].nunique()} TFs, "
            f"{self.grn_e14['gene'].nunique()} targets"
        )
        print(
            f"{self.group2} GRN: {len(self.grn_e18)} edges, "
            f"{self.grn_e18['TF'].nunique()} TFs, "
            f"{self.grn_e18['gene'].nunique()} targets"
        )
        
        # Store for inspection
        self.group1_active_elements = group1_active_elements
        self.group2_active_elements = group2_active_elements
        return self.grn_e14, self.grn_e18
        
    def save_grns(self):
        """Save condition-specific GRNs."""
        e14_path = os.path.join(self.output_dir, 'grn_e14.csv')
        e18_path = os.path.join(self.output_dir, 'grn_e18.csv')
        
        self.grn_e14.to_csv(e14_path, index=False)
        self.grn_e18.to_csv(e18_path, index=False)
        
        print(f"\nSaved:")
        print(f"  {e14_path}")
        print(f"  {e18_path}")



    
    def compute_element_zscores_parallel(self, 
                                        all_e14_peak_files, all_e18_peak_files,  # Global distribution
                                        cd133_e14_peak_files, cd133_e18_peak_files,  # CD133 Ctx only
                                        e14_sample_ids, e18_sample_ids, 
                                        n_cores=8):
        """
        Compute Z-scores for regulatory elements across conditions (parallelized).
        
        Args:
            all_e14_peak_files: ALL E14 narrowPeak files (for global distribution)
            all_e18_peak_files: ALL E18 narrowPeak files (for global distribution)
            cd133_e14_peak_files: CD133 Ctx E14 narrowPeak files (for Z-score calculation)
            cd133_e18_peak_files: CD133 Ctx E18 narrowPeak files (for Z-score calculation)
            e14_sample_ids: CD133 Ctx E14 sample IDs
            e18_sample_ids: CD133 Ctx E18 sample IDs
            n_cores: Number of cores for parallel processing
        """
        from multiprocessing import Pool
        
        print("\n" + "="*80)
        print("COMPUTING ELEMENT Z-SCORES (PARALLELIZED)")
        print("="*80)
        
        # Use ALL samples for global distribution
        all_peak_files = all_e14_peak_files + all_e18_peak_files
        cd133_peak_files = cd133_e14_peak_files + cd133_e18_peak_files
        
        print(f"Global distribution files: {len(all_peak_files)}")
        print(f"  E14: {len(all_e14_peak_files)}, E18: {len(all_e18_peak_files)}")
        print(f"CD133 Ctx files: {len(cd133_peak_files)}")
        print(f"  E14: {len(cd133_e14_peak_files)}, E18: {len(cd133_e18_peak_files)}")
        print(f"Using {n_cores} cores")
        
        # Extract only necessary columns as plain DataFrame
        reg_df = self.merged_regulatory[['chr', 'start', 'end', 'element_key']].copy()
        
        # Step 1: Build signal distribution from ALL samples
        print("\nStep 1: Building signal distribution per element (ALL samples)...")
        args_list = [(peak_file, reg_df) for peak_file in all_peak_files]
        
        with Pool(n_cores) as pool:
            results = pool.map(_process_sample_for_distribution, args_list)
        
        # Aggregate signals
        element_signals = {}
        for sample_signals in results:
            for element_key, signals in sample_signals.items():
                if element_key not in element_signals:
                    element_signals[element_key] = []
                element_signals[element_key].extend(signals)
        
        print(f"Elements with signals: {len(element_signals)}")
        
        # Calculate stats
        element_stats = []
        for element_key, signals in element_signals.items():
            element_stats.append({
                'element_key': element_key,
                'mean_signal': np.mean(signals),
                'sd_signal': np.std(signals),
                'n_samples': len(signals)
            })
        
        element_stats_df = pd.DataFrame(element_stats)
        print(f"\nElement statistics computed")

        # Convert to dict for fast lookup
        element_stats_dict = {
            row['element_key']: {'mean': row['mean_signal'], 'sd': row['sd_signal']}
            for _, row in element_stats_df.iterrows()
        }

        # Step 2: Calculate Z-scores ONLY for CD133 Ctx samples
        print("\nStep 2: Calculating Z-scores for CD133 Ctx samples...")
        args_list = [(peak_file, element_stats_dict, reg_df) for peak_file in cd133_peak_files]
        
        with Pool(n_cores) as pool:
            all_sample_zscores = pool.map(_calculate_zscores_for_sample, args_list)
        
        # Aggregate by condition
        e14_zscores = {}
        e18_zscores = {}
        
        for i, sample_zscores in enumerate(all_sample_zscores):
            if i < len(cd133_e14_peak_files):
                # E14 CD133 Ctx sample
                for element_key, zscores in sample_zscores.items():
                    if element_key not in e14_zscores:
                        e14_zscores[element_key] = []
                    e14_zscores[element_key].extend(zscores)
            else:
                # E18 CD133 Ctx sample
                for element_key, zscores in sample_zscores.items():
                    if element_key not in e18_zscores:
                        e18_zscores[element_key] = []
                    e18_zscores[element_key].extend(zscores)
        
        # Sum Z-scores per condition
        e14_sum = {k: np.sum(v) for k, v in e14_zscores.items()}
        e18_sum = {k: np.sum(v) for k, v in e18_zscores.items()}
        
        # Add to merged_regulatory
        self.merged_regulatory['zscore_E14'] = self.merged_regulatory['element_key'].map(e14_sum).fillna(0)
        self.merged_regulatory['zscore_E18'] = self.merged_regulatory['element_key'].map(e18_sum).fillna(0)
        
        print(f"\nZ-scores computed:")
        print(f"  E14 elements with signal: {(self.merged_regulatory['zscore_E14'] != 0).sum()}")
        print(f"  E18 elements with signal: {(self.merged_regulatory['zscore_E18'] != 0).sum()}")
        
        return self.merged_regulatory
    
# Define worker function
def _process_sample(args):
    """Process one sample and return element:signal mapping."""
    peak_file, sample_id, reg_df = args
    
    # Load narrowPeak
    peaks = pd.read_csv(peak_file, sep='\t', header=None,
                    names=['chr', 'start', 'end', 'name', 'score', 
                            'strand', 'signalValue', 'pValue', 'qValue', 'peak'])
    
    # Overlap
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks[['chr', 'start', 'end', 'signalValue']])
    reg_bed = pybedtools.BedTool.from_dataframe(reg_df[['chr', 'start', 'end', 'element_key']])
    
    overlap = peaks_bed.intersect(reg_bed, wa=True, wb=True)
    
    # Collect signals for this sample
    sample_signals = {}
    for interval in overlap:
        signal = float(interval[3])
        element_key = interval[7]
        sample_signals[element_key] = signal
    
    return sample_id, sample_signals

# ==============================================================================
# HELPER FUNCTIONS (NO CHANGES)
# ==============================================================================

def _process_sample_for_distribution(args):
    """Process ONE sample."""
    import pandas as pd
    import pybedtools
    
    peak_file, reg_data = args
    
    peaks = pd.read_csv(peak_file, sep='\t', header=None,
                    names=['chr', 'start', 'end', 'name', 'score', 
                            'strand', 'signalValue', 'pValue', 'qValue', 'peak'])
    
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks[['chr', 'start', 'end', 'signalValue']])
    reg_bed = pybedtools.BedTool.from_dataframe(reg_data)
    
    overlap = peaks_bed.intersect(reg_bed, wa=True, wb=True)
    
    sample_signals = {}
    for interval in overlap:
        signal = float(interval[3])
        element_key = interval[7]
        if element_key not in sample_signals:
            sample_signals[element_key] = []
        sample_signals[element_key].append(signal)
    
    return sample_signals


def _calculate_zscores_for_sample(args):
    """Calculate Z-scores for ONE sample."""
    import pandas as pd
    import numpy as np
    import pybedtools
    
    peak_file, element_stats_dict, reg_data = args
    
    peaks = pd.read_csv(peak_file, sep='\t', header=None,
                    names=['chr', 'start', 'end', 'name', 'score', 
                            'strand', 'signalValue', 'pValue', 'qValue', 'peak'])
    
    peaks_bed = pybedtools.BedTool.from_dataframe(peaks[['chr', 'start', 'end', 'signalValue']])
    reg_bed = pybedtools.BedTool.from_dataframe(reg_data)
    
    overlap = peaks_bed.intersect(reg_bed, wa=True, wb=True)
    
    sample_zscores = {}
    
    for interval in overlap:
        signal = float(interval[3])
        element_key = interval[7]
        
        if element_key in element_stats_dict:
            mean_sig = element_stats_dict[element_key]['mean']
            sd_sig = element_stats_dict[element_key]['sd']
            
            if sd_sig > 0:
                zscore = (signal - mean_sig) / sd_sig
            else:
                zscore = 0
            
            if element_key not in sample_zscores:
                sample_zscores[element_key] = []
            sample_zscores[element_key].append(zscore)
    
    return sample_zscores

# import numpy as np
# import pandas as pd
# from tqdm import tqdm 
# def sigmoid(x):
#     """Sigmoid function."""
#     return 1 / (1 + np.exp(-x))

# # ==============================================================================
# # STEP 1: Define binary RNA-seq DE indicators E_x(c)
# # ==============================================================================

# print("\n" + "="*80)
# print("STEP 1: BINARY DE INDICATORS")
# print("="*80)

# tau_lfc = 1.0  # |logFC| threshold
# tau_padj = 0.05

# # E14 condition: positive logFC
# deseq_results['E_E14'] = (
#     (deseq_results['log2FoldChange'] > tau_lfc) & 
#     (deseq_results['padj'] < tau_padj)
# ).astype(int)

# # E18 condition: negative logFC  
# deseq_results['E_E18'] = (
#     (deseq_results['log2FoldChange'] < -tau_lfc) & 
#     (deseq_results['padj'] < tau_padj)
# ).astype(int)

# print(f"E14 DE genes: {deseq_results['E_E14'].sum()}")
# print(f"E18 DE genes: {deseq_results['E_E18'].sum()}")

# # Create lookup dictionaries
# E_E14_dict = deseq_results.set_index('symbol')['E_E14'].to_dict()
# E_E18_dict = deseq_results.set_index('symbol')['E_E18'].to_dict()


# # ==============================================================================
# # STEP 2: COMPUTE DA INDICATORS D_r(c) FOR EACH ELEMENT
# # ==============================================================================
# import pybedtools
# print("\n" + "="*80)
# print("STEP 2: BINARY DA INDICATORS PER ELEMENT")
# print("="*80)

# print(f"Merged regulatory elements: {len(merged_regulatory)}")
# print(f"DA peaks: {len(dar_all)}")

# # Overlap merged_regulatory with dar_all
# reg_bed = pybedtools.BedTool.from_dataframe(
#     merged_regulatory[['chr', 'start', 'end', 'gene', 'region_type']]
# )

# dar_bed = pybedtools.BedTool.from_dataframe(
#     dar_all[['seqnames', 'start', 'end', 'Fold']].rename(columns={'seqnames': 'chr'})
# )

# reg_da_overlap = reg_bed.intersect(dar_bed, wa=True, wb=True)


# print(f"Computing DA overlap...")

# # Parse overlaps to get DA status per element
# element_da_status = {}  # key: (chr, start, end, gene), value: Fold

# for interval in tqdm(reg_da_overlap):
#     element_key = (interval[0], int(interval[1]), int(interval[2]), interval[3])
#     da_fold = float(interval[8])
    
#     # Keep the DA peak with max absolute fold change
#     if element_key not in element_da_status:
#         element_da_status[element_key] = da_fold
#     else:
#         if abs(da_fold) > abs(element_da_status[element_key]):
#             element_da_status[element_key] = da_fold

# print(f"Elements with DA overlap: {len(element_da_status)}")

# # Add to merged_regulatory
# merged_regulatory['element_key'] = list(zip(
#     merged_regulatory['chr'],
#     merged_regulatory['start'],
#     merged_regulatory['end'],
#     merged_regulatory['gene']
# ))

# merged_regulatory['da_fold'] = merged_regulatory['element_key'].map(
#     lambda k: element_da_status.get(k, 0)
# )

# # Binary indicators: D_r(E14) = 1 if da_fold > 0, D_r(E18) = 1 if da_fold < 0
# merged_regulatory['D_E14'] = (merged_regulatory['da_fold'] > 0).astype(int)
# merged_regulatory['D_E18'] = (merged_regulatory['da_fold'] < 0).astype(int)

# print(f"\nResults:")
# print(f"  Elements with DA: {(merged_regulatory['da_fold'] != 0).sum()}")
# print(f"  Elements with E14 DA (Fold > 0): {merged_regulatory['D_E14'].sum()}")
# print(f"  Elements with E18 DA (Fold < 0): {merged_regulatory['D_E18'].sum()}")
# print(f"\nSample:")
# print(merged_regulatory[merged_regulatory['D_E14'] == 1][['chr', 'start', 'end', 'gene', 'region_type', 'da_fold', 'D_E14']].head())



# # Load Cistrome metadata
# cistrome_meta_path = '/mnt/lscratch/users/adhal/SingleCellUtils/data/pkn_data/mouse/CistromeDB_mm10_tranfac_version_3.0/mm10_tranfac_QC.txt'
# cistrome_meta = pd.read_csv(cistrome_meta_path, sep='\t')

# # Load ReMap (BED format: chr, start, end, name, score, strand, ...)
# remap_path = '/mnt/lscratch/users/adhal/CorticalNeuronFate/CellConversionNSC/data/chip_seq/remap2022_nr_macs2_mm39_v1_0.bed'
# remap_peaks = pd.read_csv(remap_path, sep='\t', header=None,
#                           names=['chr', 'start', 'end', 'name', 'score', 'strand',
#                                 'thick_start', 'thick_end', 'color'])

# # Extract TF name from 'name' column (format: TF:celltype)
# remap_peaks['TF'] = remap_peaks['name'].str.split(':').str[0]

# print(f"ReMap peaks loaded: {len(remap_peaks):,}")
# print(f"Unique TFs in ReMap: {remap_peaks['TF'].nunique()}")

# print("\nSample TFs:")
# print(remap_peaks['TF'].value_counts().head(10))

# remap_bed = pybedtools.BedTool.from_dataframe(
#     remap_peaks[['chr', 'start', 'end', 'TF']]
# )

# # Create a unique element key
# merged_regulatory['element_key'] = (
#     merged_regulatory['chr'].astype(str) + ':' +
#     merged_regulatory['start'].astype(str) + '-' +
#     merged_regulatory['end'].astype(str)
# )

# regulatory_bed = pybedtools.BedTool.from_dataframe(
#     merged_regulatory[['chr', 'start', 'end', 'element_key']]
# )

# print("\n" + "="*80)
# print("STEP 3: TF–ELEMENT BINDING B_{t,r} (ReMap only)")
# print("="*80)

# # Intersect ReMap peaks with regulatory elements
# chip_overlap = remap_bed.intersect(
#     regulatory_bed,
#     wa=True,  # TF peak columns
#     wb=True   # element columns
# )

# # Convert to DataFrame with canonical column names
# chip_overlap_df = chip_overlap.to_dataframe(
#     names=[
#         'chr_tf', 'start_tf', 'end_tf', 'TF',
#         'chr_el', 'start_el', 'end_el', 'element_key'
#     ]
# )

# # Canonicalize TF names
# chip_overlap_df['TF'] = chip_overlap_df['TF'].str.upper()

# # B_{t,r} = 1 if ANY overlap exists
# B_tr = (
#     chip_overlap_df
#     .drop_duplicates(subset=['TF', 'element_key'])
#     .assign(B_tr=1)
# )

# print(f"TF–element pairs with ReMap binding: {len(B_tr):,}")
# print(f"Unique TFs with binding: {B_tr['TF'].nunique():,}")
# print(f"Unique regulatory elements bound: {B_tr['element_key'].nunique():,}")

# TF_target_df = pd.read_csv('/mnt/lscratch/users/adhal/CorticalNeuronFate/CellConversionNSC/data/chip_seq/mouse_all_chip.csv')


# ## Can we overlap DE TFs and DE genes here and seen which ones we geT?
# e14_cond = (deseq_results['padj'] < 0.05) & (deseq_results['log2FoldChange'] > 1.00)
# e18_cond = (deseq_results['padj'] < 0.05)& (deseq_results['log2FoldChange'] < -1.00)
# de_e14_tfs = deseq_results[e14_cond & deseq_results['is_TF']]['symbol'].to_numpy()
# de_e18_tfs = deseq_results[e18_cond & deseq_results['is_TF']]['symbol'].to_numpy()

# de_e14_genes =deseq_results[e14_cond & ~deseq_results['is_TF']]['symbol'].to_numpy()
# de_e18_genes =deseq_results[e18_cond & ~deseq_results['is_TF']]['symbol'].to_numpy()

# ## TFs with binding evidence 
# e14_chip_df = chip_overlap_df[chip_overlap_df['TF'].isin([i.upper() for i in de_e14_tfs])].reset_index(drop=True)
# e18_chip_df = chip_overlap_df[chip_overlap_df['TF'].isin([i.upper() for i in de_e18_tfs])].reset_index(drop=True)

# ## DA regions with DE genes
# e14_DA_de_df = merged_regulatory[merged_regulatory['gene'].isin(de_e14_genes)].reset_index(drop=True)
# e18_DA_de_df = merged_regulatory[merged_regulatory['gene'].isin(de_e18_genes)].reset_index(drop=True)

# e14_tf_target_de_da = pd.merge(e14_chip_df, e14_DA_de_df[e14_DA_de_df['D_E14'] == 1].reset_index(drop=True), left_on='element_key', right_on='element_key')
# e18_tf_target_de_da = pd.merge(e18_chip_df, e18_DA_de_df[e18_DA_de_df['D_E18'] == 1].reset_index(drop=True), left_on='element_key', right_on='element_key')

# grn_e18 = e18_tf_target_de_da[['TF', 'gene']]
# grn_e14 = e14_tf_target_de_da[['TF', 'gene']]


# TF_target_df_sub_e14 = TF_target_df[(TF_target_df['TF'].isin(grn_e14['TF'])) & (TF_target_df['Target'].isin(grn_e14['TF']))].reset_index(drop=True)
# grn_e14_for_merge = grn_e14.copy()
# grn_e14_for_merge['gene'] = grn_e14_for_merge['gene'].str.upper()
# grn_e14_for_merge.columns = ['TF', 'Target']
# new_full_net_e14 = pd.concat([TF_target_df_sub_e14[['TF', 'Target']], grn_e14_for_merge.reset_index(drop=True)], axis=0).drop_duplicates()
# new_full_net_e14 = new_full_net_e14[new_full_net_e14['TF'] != new_full_net_e14['Target']].reset_index(drop=True)
# new_full_net_e14.columns = ['TF', 'gene']

# TF_target_df_sub_e18 = TF_target_df[(TF_target_df['TF'].isin(grn_e18['TF'])) & (TF_target_df['Target'].isin(grn_e18['TF']))].reset_index(drop=True)
# grn_e18_for_merge = grn_e18.copy()
# grn_e18_for_merge['gene'] = grn_e18_for_merge['gene'].str.upper()
# grn_e18_for_merge.columns = ['TF', 'Target']
# new_full_net_e18 = pd.concat([TF_target_df_sub_e18[['TF', 'Target']], grn_e18_for_merge.reset_index(drop=True)], axis=0).drop_duplicates()
# new_full_net_e18 = new_full_net_e18[new_full_net_e18['TF'] != new_full_net_e18['Target']].reset_index(drop=True)
# new_full_net_e18.columns = ['TF', 'gene']