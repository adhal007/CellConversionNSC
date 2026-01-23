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
    
class RegulatoryRegionMaker:
    def __init__(self):
        # NEW containers
        self.promoters = None
        self.chip_peaks = None


    def _load_data(self):
        # 1. Parse GTF for gene mappings
        print("\n[1/3] Parsing GTF for gene symbol mappings...")
        self.ensembl_to_symbol = self._parse_gtf_gene_mapping()
        
        mapped = sum(1 for g in self.counts.index if g in self.ensembl_to_symbol)
        print(f"      Counts genes with mapping: {mapped} / {len(self.counts)}")
        
        print("\n" + "=" * 60)
        print("Data loaded successfully!")
        print("=" * 60)

        # 2. Generate promoters from GTF
        print("\n[2/3] Generating promoters from GTF...")
        self.promoters = self._get_promoters_from_gtf(window=2000)
        print(f"      {len(self.promoters)} promoters")

        # 3. Load ChIP peaks (if provided)
        if self.paths['chip_peaks'] and self.paths['chip_peaks'].exists():
            print("\n[3/3] Loading ChIP-seq peaks...")
            self.chip_peaks = self._load_chip_peaks()
            print(f"      {len(self.chip_peaks)} peaks")
            print(f"      {self.chip_peaks['TF'].nunique()} unique TFs")
        else:
            print("\n[13/13] No ChIP peaks file provided, skipping...")
            self.chip_peaks = None

    def _parse_gtf_gene_mapping(self) -> Dict[str, str]:
        """Parse GTF file to create Ensembl ID -> Gene Symbol mapping."""
        print("      Parsing GTF for gene mappings...")
        
        ensembl_to_symbol = {}
        
        with open(self.paths['gtf'], 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                # Only parse gene entries
                if fields[2] != 'gene':
                    continue
                
                attributes = fields[8]
                
                # Extract gene_id and gene_name
                gene_id = None
                gene_name = None
                
                for attr in attributes.split(';'):
                    attr = attr.strip()
                    if attr.startswith('gene_id'):
                        # gene_id "ENSMUSG00000000001.5"
                        gene_id = attr.split('"')[1].split('.')[0]  # Remove version
                    elif attr.startswith('gene_name'):
                        # gene_name "Gnai3"
                        gene_name = attr.split('"')[1]
                
                if gene_id and gene_name:
                    ensembl_to_symbol[gene_id] = gene_name
        
        print(f"      Parsed {len(ensembl_to_symbol)} gene mappings from GTF")
        return ensembl_to_symbol
    
    def _get_promoters_from_gtf(self, window=2000):
        """
        Extract promoter regions (UPSTREAM of TSS) from GTF.
        
        Promoter = window bp UPSTREAM of TSS
        """
        promoters = []
        
        with open(self.paths['gtf']) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if fields[2] != 'gene':
                    continue
                
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                
                # Extract gene symbol
                attrs = {}
                for item in fields[8].split(';'):
                    if not item.strip():
                        continue
                    key_val = item.strip().split(' ', 1)
                    if len(key_val) == 2:
                        attrs[key_val[0]] = key_val[1].strip('"')
                
                gene_name = attrs.get('gene_name', '')
                
                # Get TSS and promoter based on strand
                if strand == '+':
                    tss = start
                    # Promoter is UPSTREAM (before gene start)
                    promoter_start = max(0, tss - window)
                    promoter_end = tss
                else:  # - strand
                    tss = end
                    # Promoter is UPSTREAM (after gene end in coordinates)
                    promoter_start = tss
                    promoter_end = tss + window
                
                promoters.append({
                    'chr': chrom,
                    'start': promoter_start,
                    'end': promoter_end,
                    'gene': gene_name,
                    'strand': strand,
                    'tss': tss
                })
        
        return pd.DataFrame(promoters)
##############################################################################################################
##############################################################################################################
    def create_merged_regulatory_regions(self, promoters, enhancers):
        """
        Merge promoters and enhancers into unified regulatory regions.
        
        SIMPLE LOGIC:
        1. Same-gene overlap → Merge into "proximal_regulatory"
        2. Everything else → Keep as-is ("promoter" or "enhancer")
        
        Different-gene overlaps are FINE - both regions stay separate.
        """
        print("\n" + "="*80)
        print("CREATING MERGED REGULATORY REGIONS")
        print("="*80)
        
        print(f"Input:")
        print(f"  Promoters: {len(promoters):,}")
        print(f"  Enhancers: {len(enhancers):,}")
        
        # Convert to PyRanges
        promoters_pr = pr.PyRanges(promoters.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}
        ))
        
        enhancers_pr = pr.PyRanges(enhancers.rename(
            columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End', 'symbol': 'gene'}
        ))
        
        # Find overlaps
        all_overlaps = promoters_pr.join(enhancers_pr).df
        
        # Get SAME-GENE overlaps
        same_gene_overlaps = all_overlaps[
            all_overlaps['gene'] == all_overlaps['gene_b']
        ].copy()
        
        print(f"\nSame-gene promoter-enhancer overlaps: {len(same_gene_overlaps):,}")
        
        # Track which promoters/enhancers are in same-gene overlaps
        promoters_in_proximal = set(
            zip(same_gene_overlaps['gene'], 
                same_gene_overlaps['Chromosome'],
                same_gene_overlaps['Start'], 
                same_gene_overlaps['End'])
        )
        
        enhancers_in_proximal = set(
            zip(same_gene_overlaps['gene_b'],
                same_gene_overlaps['Chromosome'],
                same_gene_overlaps['Start_b'], 
                same_gene_overlaps['End_b'])
        )
        
        # 1. Create proximal regulatory regions (merge same-gene overlaps)
        proximal_regions = []
        for _, row in same_gene_overlaps.iterrows():
            merged_start = min(row['Start'], row['Start_b'])
            merged_end = max(row['End'], row['End_b'])
            
            proximal_regions.append({
                'chr': row['Chromosome'],
                'start': merged_start,
                'end': merged_end,
                'gene': row['gene'],
                'region_type': 'proximal_regulatory'
            })
        
        proximal_df = pd.DataFrame(proximal_regions).drop_duplicates()
        
        # 2. Keep promoters that DON'T have same-gene overlap
        promoter_regions = []
        for _, prom in promoters.iterrows():
            prom_key = (prom['gene'], prom['chr'], prom['start'], prom['end'])
            if prom_key not in promoters_in_proximal:
                promoter_regions.append({
                    'chr': prom['chr'],
                    'start': prom['start'],
                    'end': prom['end'],
                    'gene': prom['gene'],
                    'region_type': 'promoter'
                })
        
        promoter_df = pd.DataFrame(promoter_regions)
        
        # 3. Keep enhancers that DON'T have same-gene overlap
        enhancer_regions = []
        for _, enh in enhancers.iterrows():
            enh_key = (enh['symbol'], enh['chr'], enh['start'], enh['end'])
            if enh_key not in enhancers_in_proximal:
                enhancer_regions.append({
                    'chr': enh['chr'],
                    'start': enh['start'],
                    'end': enh['end'],
                    'gene': enh['symbol'],
                    'region_type': 'enhancer'
                })
        
        enhancer_df = pd.DataFrame(enhancer_regions)
        
        # Combine
        merged_regulatory = pd.concat([
            promoter_df,
            proximal_df,
            enhancer_df
        ], ignore_index=True)
        
        print(f"\nOutput:")
        print(f"  Promoter: {len(promoter_df):,}")
        print(f"  Proximal regulatory: {len(proximal_df):,}")
        print(f"  Enhancer: {len(enhancer_df):,}")
        print(f"  Total: {len(merged_regulatory):,}")
        
        return merged_regulatory

