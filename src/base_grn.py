import pandas as pd
from celloracle import motif_analysis as ma
import genomepy


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
