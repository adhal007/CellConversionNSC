# ==============================================================================
# DiffBind Analysis for ATAC-seq
# ==============================================================================

library(DiffBind)
library(tidyverse)

cat("=== Starting DiffBind Analysis ===\n")
cat("Time:", format(Sys.time()), "\n\n")

# Set working directory
setwd("/home/users/adhal/CorticalNeuronFate/CellConversionNSC")

# Create results directory
dir.create("results/diffbind", recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# 1. Load sample sheet and create DiffBind object
# ==============================================================================

cat("Loading sample sheet...\n")
samples <- read.csv("data/atac_seq/ATAC-seq/samples_clean.csv")
print(samples[, c("SampleID", "Tissue", "Factor", "Condition")])

cat("\nSample distribution:\n")
print(table(samples$Condition, samples$Factor, samples$Tissue))

# Create DiffBind object
cat("\nCreating DiffBind object...\n")
dba_obj <- dba(sampleSheet = samples)
print(dba_obj)

# Plot correlation based on peak overlap
pdf("results/diffbind/01_peak_overlap_correlation.pdf", width = 10, height = 10)
plot(dba_obj)
dev.off()
cat("Saved: 01_peak_overlap_correlation.pdf\n")

# ==============================================================================
# 2. Count reads in consensus peaks
# ==============================================================================

cat("\n=== Counting reads in peaks (this takes ~30-60 min) ===\n")
cat("Start time:", format(Sys.time()), "\n")

dba_obj <- dba.count(dba_obj, 
                      summits = 250,
                      minOverlap = 2,
                      bParallel = TRUE)

cat("End time:", format(Sys.time()), "\n")

# Save checkpoint
save(dba_obj, file = "results/diffbind/dba_counted.RData")
cat("Saved checkpoint: dba_counted.RData\n")

# Normalize
dba_obj <- dba.normalize(dba_obj)

# Plot correlation after counting
pdf("results/diffbind/02_count_correlation.pdf", width = 10, height = 10)
plot(dba_obj)
dev.off()

# PCA
pdf("results/diffbind/03_pca_all_samples.pdf", width = 10, height = 8)
dba.plotPCA(dba_obj, label = DBA_ID)
dev.off()

pdf("results/diffbind/03_pca_by_condition.pdf", width = 10, height = 8)
dba.plotPCA(dba_obj, attributes = c(DBA_CONDITION, DBA_TISSUE), label = DBA_ID)
dev.off()
cat("Saved: PCA plots\n")

# ==============================================================================
# 3. Differential Analysis - TEMPORAL CD133 (E14 vs E18)
# ==============================================================================

cat("\n=== Running Temporal CD133 comparison ===\n")

dba_cd133 <- dba(dba_obj, mask = dba_obj$samples$Factor == "CD133")

dba_cd133 <- dba.contrast(dba_cd133,
                           reorderMeta = list(Condition = "E14"),
                           design = "~Tissue + Condition",
                           minMembers = 2)

dba_cd133 <- dba.analyze(dba_cd133, method = DBA_DESEQ2)
dba.show(dba_cd133, bContrasts = TRUE)

results_temporal_cd133 <- dba.report(dba_cd133, th = 1)
results_temporal_cd133_sig <- dba.report(dba_cd133, th = 0.05, fold = 1)

cat("\nTemporal CD133 (E14 vs E18):\n")
cat("  Total peaks tested:", length(results_temporal_cd133), "\n")
cat("  Significant (FDR<0.05, |LFC|>1):", length(results_temporal_cd133_sig), "\n")
cat("  E14-high:", sum(results_temporal_cd133_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_temporal_cd133_sig$Fold < 0), "\n")

# ==============================================================================
# 4. Differential Analysis - TEMPORAL ALL (E14 vs E18)
# ==============================================================================

cat("\n=== Running Temporal ALL comparison ===\n")

dba_all <- dba.contrast(dba_obj,
                         reorderMeta = list(Condition = "E14"),
                         design = "~Tissue + Factor + Condition",
                         minMembers = 2)

dba_all <- dba.analyze(dba_all, method = DBA_DESEQ2)

results_temporal_all <- dba.report(dba_all, th = 1)
results_temporal_all_sig <- dba.report(dba_all, th = 0.05, fold = 1)

cat("\nTemporal ALL (E14 vs E18):\n")
cat("  Total peaks tested:", length(results_temporal_all), "\n")
cat("  Significant (FDR<0.05, |LFC|>1):", length(results_temporal_all_sig), "\n")
cat("  E14-high:", sum(results_temporal_all_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_temporal_all_sig$Fold < 0), "\n")

# ==============================================================================
# 5. Differential Analysis - REGIONAL E14 (Cortex vs LGE)
# ==============================================================================

cat("\n=== Running Regional E14 comparison ===\n")

dba_regional <- dba(dba_obj, mask = dba_obj$samples$Condition == "E14")

dba_regional <- dba.contrast(dba_regional,
                              reorderMeta = list(Tissue = "Cortex"),
                              design = "~Factor + Tissue",
                              minMembers = 2)

dba_regional <- dba.analyze(dba_regional, method = DBA_DESEQ2)

results_regional <- dba.report(dba_regional, th = 1)
results_regional_sig <- dba.report(dba_regional, th = 0.05, fold = 1)

cat("\nRegional E14 (Cortex vs LGE):\n")
cat("  Total peaks tested:", length(results_regional), "\n")
cat("  Significant (FDR<0.05, |LFC|>1):", length(results_regional_sig), "\n")
cat("  Cortex-high:", sum(results_regional_sig$Fold > 0), "\n")
cat("  LGE-high:", sum(results_regional_sig$Fold < 0), "\n")

# ==============================================================================
# 6. Visualizations
# ==============================================================================

cat("\n=== Generating plots ===\n")

# MA plots
pdf("results/diffbind/04_MA_plots.pdf", width = 12, height = 4)
par(mfrow = c(1, 3))
dba.plotMA(dba_cd133, contrast = 1, main = "Temporal CD133: E14 vs E18")
dba.plotMA(dba_all, contrast = 1, main = "Temporal ALL: E14 vs E18")
dba.plotMA(dba_regional, contrast = 1, main = "Regional E14: Cortex vs LGE")
dev.off()

# Volcano plots
pdf("results/diffbind/05_volcano_plots.pdf", width = 12, height = 4)
par(mfrow = c(1, 3))
dba.plotVolcano(dba_cd133, contrast = 1)
dba.plotVolcano(dba_all, contrast = 1)
dba.plotVolcano(dba_regional, contrast = 1)
dev.off()

# Heatmaps
pdf("results/diffbind/06_heatmap_temporal_cd133.pdf", width = 10, height = 12)
tryCatch({
    dba.plotHeatmap(dba_cd133, contrast = 1, correlations = FALSE)
}, error = function(e) cat("Heatmap error:", e$message, "\n"))
dev.off()

pdf("results/diffbind/06_heatmap_temporal_all.pdf", width = 10, height = 12)
tryCatch({
    dba.plotHeatmap(dba_all, contrast = 1, correlations = FALSE)
}, error = function(e) cat("Heatmap error:", e$message, "\n"))
dev.off()

cat("Saved: MA plots, volcano plots, heatmaps\n")

# ==============================================================================
# 7. Export results
# ==============================================================================

cat("\n=== Exporting results ===\n")

export_results <- function(gr, filename) {
    df <- as.data.frame(gr)
    write.csv(df, filename, row.names = FALSE)
    cat("Saved:", filename, "\n")
}

export_results(results_temporal_cd133, "results/diffbind/DAR_temporal_cd133_all.csv")
export_results(results_temporal_cd133_sig, "results/diffbind/DAR_temporal_cd133_sig.csv")
export_results(results_temporal_all, "results/diffbind/DAR_temporal_all_all.csv")
export_results(results_temporal_all_sig, "results/diffbind/DAR_temporal_all_sig.csv")
export_results(results_regional, "results/diffbind/DAR_regional_e14_all.csv")
export_results(results_regional_sig, "results/diffbind/DAR_regional_e14_sig.csv")

# Save all objects
save(dba_obj, dba_cd133, dba_all, dba_regional,
     results_temporal_cd133, results_temporal_cd133_sig,
     results_temporal_all, results_temporal_all_sig,
     results_regional, results_regional_sig,
     file = "results/diffbind/dba_all_results.RData")

cat("\n=== DONE ===\n")
cat("End time:", format(Sys.time()), "\n")