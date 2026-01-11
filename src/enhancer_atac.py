"""
Simple classes for ATAC-seq consensus peaks and enhancer integration.
"""

import pandas as pd
import pyranges as pr
from pyliftover import LiftOver
from pathlib import Path
import pybedtools

class ConsensusPeakBuilder:
    """Create consensus peaks from ATAC-seq narrowPeak files."""
    
    def __init__(self, metadata_file, peak_dir, factor='CD133', conditions=['E14', 'E18'], region=None):
        """
        Args:
            metadata_file: Path to samples CSV
            peak_dir: Directory containing narrowPeak files
            factor: Factor to filter (e.g., 'CD133')
            conditions: Conditions to include (e.g., ['E14', 'E18'])
            region: Optional region filter (e.g., 'Ctx', 'LGE')
        """
        self.metadata = pd.read_csv(metadata_file)
        self.peak_dir = Path(peak_dir)
        
        # Filter samples
        mask = (self.metadata['Factor'] == factor) & (self.metadata['Condition'].isin(conditions))
        if region is not None:
            mask = mask & (self.metadata['Tissue'].str.contains(region, case=False, na=False))
        
        self.samples = self.metadata[mask].copy()
        
        print(f"Found {len(self.samples)} samples for {factor} in {conditions}" + 
              (f" ({region})" if region else ""))
        

    def build_consensus(self):
        """Load all peaks and merge to create consensus."""
        all_peaks = []
        
        for _, row in self.samples.iterrows():
            peak_file = self.peak_dir / Path(row['Peaks']).name
            
            if not peak_file.exists():
                print(f"Warning: {peak_file} not found, skipping")
                continue
            
            df = pd.read_csv(peak_file, sep='\t', header=None, usecols=[0,1,2])
            df.columns = ['Chromosome', 'Start', 'End']
            df['Start'] = df['Start'] + 1
            all_peaks.append(df)
        
        combined = pd.concat(all_peaks, ignore_index=True)
        
        # Keep standard chromosomes - same as R keepStandardChromosomes
        # Remove random, chrUn, haplotype scaffolds
        combined = combined[~combined['Chromosome'].str.contains('_')]
        
        pr_peaks = pr.PyRanges(combined)
        consensus = pr_peaks.merge()
        
        print(f"Consensus peaks: {len(consensus)}")
        return consensus.df
    def build_consensus_by_condition(self):
        """Build separate consensus peaks for each condition (FAST VERSION)."""
        condition_consensus = {}
        
        for condition in self.samples['Condition'].unique():
            print(f"\nBuilding consensus for {condition}...")
            samples_cond = self.samples[self.samples['Condition'] == condition]
            
            all_peaks = []
            for _, row in samples_cond.iterrows():
                peak_file = self.peak_dir / Path(row['Peaks']).name
                if not peak_file.exists():
                    continue
                
                df = pd.read_csv(peak_file, sep='\t', header=None, usecols=[0,1,2])
                df.columns = ['Chromosome', 'Start', 'End']
                all_peaks.append(df)
            
            if not all_peaks:
                print(f"No peaks found for {condition}")
                continue
            
            combined = pd.concat(all_peaks, ignore_index=True)
            combined = combined[~combined['Chromosome'].str.contains('_')]
            
            # SIMPLE MERGE - no signal tracking
            pr_peaks = pr.PyRanges(combined)
            consensus_merged = pr_peaks.merge()
            consensus_df = consensus_merged.df
            
            condition_consensus[condition] = consensus_df
            print(f"{condition} consensus peaks: {len(consensus_df)}")
        
        return condition_consensus
    
    def _merge_with_signal(self, combined_df):
        """Helper: merge peaks and keep max signal."""
        pr_peaks = pr.PyRanges(combined_df)
        consensus_merged = pr_peaks.merge()
        
        consensus_with_signal = []
        for _, merged_peak in consensus_merged.df.iterrows():
            overlapping = combined_df[
                (combined_df['Chromosome'] == merged_peak['Chromosome']) &
                (combined_df['Start'] < merged_peak['End']) &
                (combined_df['End'] > merged_peak['Start'])
            ]
            
            max_signal = overlapping['Signal'].max() if len(overlapping) > 0 else 0
            
            consensus_with_signal.append({
                'Chromosome': merged_peak['Chromosome'],
                'Start': merged_peak['Start'],
                'End': merged_peak['End'],
                'Signal': max_signal
            })
        
        return pd.DataFrame(consensus_with_signal)
    
    def save_bed(self, consensus_df, output_file):
        """Save consensus peaks as BED file."""
        consensus_df[['Chromosome', 'Start', 'End']].to_csv(
            output_file, sep='\t', header=False, index=False
        )
        print(f"Saved to {output_file}")


class EnhancerIntegrator:
    """LiftOver enhancers and overlap with consensus peaks."""
    
    def __init__(self, enhancer_files, from_assembly='mm9'):
        """
        Args:
            enhancer_files: List of enhancer atlas files
            from_assembly: Source assembly ('mm9' or 'mm10')
        """
        self.enhancer_files = enhancer_files
        self.from_assembly = from_assembly
        self.enhancers_mm39 = None
        
        print(f"Initializing liftOver from {from_assembly} to mm39...")
        self.lo = LiftOver(from_assembly, 'mm39')
    
    def load_and_liftover(self):
        """Load enhancers and liftover to mm39."""
        all_enh = []
        
        # Load all files
        for f in self.enhancer_files:
            print(f"Loading {Path(f).name}")
            df = pd.read_csv(f, sep='\t', header=None, names=['id', 'score'])
            
            # Parse: chr:start-end_ENSMUSG$Symbol$chr$tss
            df[['coords', 'gene_info']] = df['id'].str.split('_', n=1, expand=True)
            df[['chr', 'range']] = df['coords'].str.split(':', expand=True)
            df[['start', 'end']] = df['range'].str.split('-', expand=True)
            df[['ensembl', 'symbol', 'gene_chr', 'tss']] = df['gene_info'].str.split('$', expand=True)
            
            df['start'] = pd.to_numeric(df['start'])
            df['end'] = pd.to_numeric(df['end'])
            
            all_enh.append(df[['chr', 'start', 'end', 'symbol', 'score']])
        
        combined = pd.concat(all_enh, ignore_index=True)
        
        # Remove duplicates, keep highest score
        combined['enh_gene_id'] = (combined['chr'] + ':' + 
                                   combined['start'].astype(str) + '-' + 
                                   combined['end'].astype(str) + '_' + 
                                   combined['symbol'])
        combined = combined.sort_values('score', ascending=False).drop_duplicates('enh_gene_id', keep='first')
        
        print(f"Loaded {len(combined)} unique enhancer-gene pairs in {self.from_assembly}")
        
        # LiftOver to mm39
        print(f"Lifting over to mm39...")
        lifted = []
        failed = 0
        
        for _, row in combined.iterrows():
            result = self.lo.convert_coordinate(row['chr'], int(row['start']))
            
            if result:
                new_chr, new_start, _, _ = result[0]
                length = row['end'] - row['start']
                
                lifted.append({
                    'chr': new_chr,
                    'start': new_start,
                    'end': new_start + length,
                    'symbol': row['symbol'],
                    'score': row['score']
                })
            else:
                failed += 1
        
        self.enhancers_mm39 = pd.DataFrame(lifted)
        
        print(f"LiftOver: {len(lifted)}/{len(combined)} ({len(lifted)/len(combined)*100:.1f}%)")
        print(f"Unique genes: {self.enhancers_mm39['symbol'].nunique()}")
        
        return self.enhancers_mm39
    
    def overlap_with_consensus(self, consensus_peaks):
        """
        Overlap enhancers with consensus peaks.
        Returns formatted overlaps with peak_id.
        """
        if self.enhancers_mm39 is None:
            raise ValueError("Load and liftover enhancers first!")
        
        enh_pr = pr.PyRanges(self.enhancers_mm39.rename(columns={
            'chr': 'Chromosome', 'start': 'Start', 'end': 'End'
        }))
        
        consensus_pr = pr.PyRanges(consensus_peaks)
        overlaps = enh_pr.join(consensus_pr).df
        
        # Format: use consensus peak coordinates, enhancer gene
        overlaps_formatted = pd.DataFrame({
            'chr': overlaps['Chromosome'].astype(str),  # ← Convert to string
            'start': overlaps['Start_b'],
            'end': overlaps['End_b'],
            'gene': overlaps['symbol'],
            'tissue': 'Cortex'
        })
        
        # Add peak_id (now chr is string, so concatenation works)
        overlaps_formatted['peak_id'] = (
            overlaps_formatted['chr'] + ':' + 
            overlaps_formatted['start'].astype(str) + '-' + 
            overlaps_formatted['end'].astype(str)
        )
        
        target_genes = overlaps_formatted['gene'].unique().tolist()
        
        print(f"Consensus peaks overlapping enhancers: {len(overlaps_formatted)}")
        print(f"Target genes with accessible enhancers: {len(target_genes)}")
        
        return target_genes, overlaps_formatted
    
    def get_de_with_accessible_enhancers(self, consensus_targets, de_genes):
        """Get DE genes that have accessible enhancers."""
        concordant = list(set(consensus_targets) & set(de_genes))
        pct = len(concordant) / len(de_genes) * 100 if de_genes else 0
        print(f"DE genes with accessible enhancers: {len(concordant)}/{len(de_genes)} ({pct:.1f}%)")
        return concordant

    def annotate_da_peaks(self, overlaps_df, da_peaks):
        """
        Annotate which consensus peaks overlap DA peaks.
        
        Args:
            overlaps_df: Consensus peak → gene mappings (output from overlap_with_consensus)
            da_peaks: DataFrame with DA peaks ['seqnames', 'start', 'end']
        
        Returns:
            overlaps_df with 'is_DA' column added
        """

        # Make a copy so we don't modify the input
        overlaps_df = overlaps_df.copy()
        # Get unique consensus peaks from overlaps
        consensus_peaks_unique = overlaps_df[['chr', 'start', 'end', 'peak_id']].drop_duplicates()
        
        # Create BED objects
        consensus_bed = pybedtools.BedTool.from_dataframe(
            consensus_peaks_unique[['chr', 'start', 'end', 'peak_id']].rename(columns={'chr': 'chrom', 'peak_id': 'name'})
        )
        
        da_bed = pybedtools.BedTool.from_dataframe(
            da_peaks[['seqnames', 'start', 'end']].rename(columns={'seqnames': 'chrom'})
        )
        
        # Find overlaps
        da_overlaps = consensus_bed.intersect(da_bed, wa=True, u=True)
        
        # Extract DA peak IDs
        da_peak_ids = set(interval.name for interval in da_overlaps)
        
        # Annotate original dataframe
        overlaps_df['is_DA'] = overlaps_df['peak_id'].isin(da_peak_ids)
        
        print(f"\nDA annotation:")
        print(f"  Consensus peaks overlapping DA: {len(da_peak_ids)}")
        print(f"  Genes linked to DA peaks: {overlaps_df[overlaps_df['is_DA']]['gene'].nunique()}")
        
        return overlaps_df

