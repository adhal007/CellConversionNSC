# ==============================================================================
# DiffBind Analysis for ATAC-seq - FIXED
# ==============================================================================

library(DiffBind)
library(tidyverse)

cat("=== Starting DiffBind Analysis ===\n")
cat("Time:", format(Sys.time()), "\n\n")

setwd("/home/users/adhal/CorticalNeuronFate/CellConversionNSC")
dir.create("results/diffbind", recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# 1. Load sample sheet and create DiffBind object
# ==============================================================================

cat("Loading sample sheet...\n")
samples <- read.csv("data/atac_seq/ATAC-seq/samples_clean.csv")

# Fix PSA-NCAM naming (remove hyphen causing issues)
samples$Factor <- gsub("PSA-NCAM", "PSANCAM", samples$Factor)

print(samples[, c("SampleID", "Tissue", "Factor", "Condition")])

cat("\nSample distribution:\n")
print(table(samples$Condition, samples$Factor, samples$Tissue))

# Create DiffBind object
cat("\nCreating DiffBind object...\n")
dba_obj <- dba(sampleSheet = samples)
print(dba_obj)

# ==============================================================================
# 2. Count reads in consensus peaks
# ==============================================================================

cat("\n=== Counting reads in peaks ===\n")
cat("Start time:", format(Sys.time()), "\n")

dba_obj <- dba.count(dba_obj, 
                      summits = 250,
                      minOverlap = 2,
                      bParallel = TRUE)

cat("End time:", format(Sys.time()), "\n")

save(dba_obj, file = "results/diffbind/dba_counted.RData")
cat("Saved checkpoint: dba_counted.RData\n")

dba_obj <- dba.normalize(dba_obj)

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
# 6. CORTEX-ONLY TEMPORAL (E14 vs E18)
# ==============================================================================

cat("\n=== Running Cortex-only Temporal comparison ===\n")

dba_ctx <- dba(dba_obj, mask = dba_obj$samples$Tissue == "Cortex")
print(dba_ctx)

dba_ctx <- dba.contrast(dba_ctx,
                         reorderMeta = list(Condition = "E14"),
                         design = "~Factor + Condition",
                         minMembers = 2)

dba_ctx <- dba.analyze(dba_ctx, method = DBA_DESEQ2)

results_ctx_temporal <- dba.report(dba_ctx, th = 1)
results_ctx_temporal_sig <- dba.report(dba_ctx, th = 0.05, fold = 1)

cat("\nCortex Temporal (E14 vs E18):\n")
cat("  Total peaks tested:", length(results_ctx_temporal), "\n")
cat("  Significant (FDR<0.05, |LFC|>1):", length(results_ctx_temporal_sig), "\n")
cat("  E14-high:", sum(results_ctx_temporal_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_ctx_temporal_sig$Fold < 0), "\n")

# ==============================================================================
# 7. Visualizations (with error handling)
# ==============================================================================

cat("\n=== Generating plots ===\n")

# Correlation heatmaps
tryCatch({
    pdf("results/diffbind/01_correlation_heatmap.pdf", width = 10, height = 10)
    plot(dba_obj)
    dev.off()
    cat("Saved: 01_correlation_heatmap.pdf\n")
}, error = function(e) cat("Correlation plot error:", e$message, "\n"))

# PCA
tryCatch({
    pdf("results/diffbind/02_pca.pdf", width = 10, height = 8)
    dba.plotPCA(dba_obj, label = DBA_ID)
    dev.off()
    cat("Saved: 02_pca.pdf\n")
}, error = function(e) cat("PCA plot error:", e$message, "\n"))

# MA plots - use method without main argument
tryCatch({
    pdf("results/diffbind/03_MA_temporal_cd133.pdf", width = 8, height = 6)
    dba.plotMA(dba_cd133, contrast = 1)
    dev.off()
    cat("Saved: 03_MA_temporal_cd133.pdf\n")
}, error = function(e) cat("MA plot cd133 error:", e$message, "\n"))

tryCatch({
    pdf("results/diffbind/03_MA_temporal_all.pdf", width = 8, height = 6)
    dba.plotMA(dba_all, contrast = 1)
    dev.off()
    cat("Saved: 03_MA_temporal_all.pdf\n")
}, error = function(e) cat("MA plot all error:", e$message, "\n"))

tryCatch({
    pdf("results/diffbind/03_MA_regional.pdf", width = 8, height = 6)
    dba.plotMA(dba_regional, contrast = 1)
    dev.off()
    cat("Saved: 03_MA_regional.pdf\n")
}, error = function(e) cat("MA plot regional error:", e$message, "\n"))

tryCatch({
    pdf("results/diffbind/03_MA_ctx_temporal.pdf", width = 8, height = 6)
    dba.plotMA(dba_ctx, contrast = 1)
    dev.off()
    cat("Saved: 03_MA_ctx_temporal.pdf\n")
}, error = function(e) cat("MA plot ctx error:", e$message, "\n"))

# Volcano plots
tryCatch({
    pdf("results/diffbind/04_volcano_temporal_cd133.pdf", width = 8, height = 6)
    dba.plotVolcano(dba_cd133, contrast = 1)
    dev.off()
    cat("Saved: 04_volcano_temporal_cd133.pdf\n")
}, error = function(e) cat("Volcano plot error:", e$message, "\n"))

tryCatch({
    pdf("results/diffbind/04_volcano_ctx_temporal.pdf", width = 8, height = 6)
    dba.plotVolcano(dba_ctx, contrast = 1)
    dev.off()
    cat("Saved: 04_volcano_ctx_temporal.pdf\n")
}, error = function(e) cat("Volcano ctx plot error:", e$message, "\n"))

# Heatmaps
tryCatch({
    pdf("results/diffbind/05_heatmap_temporal_cd133.pdf", width = 10, height = 12)
    dba.plotHeatmap(dba_cd133, contrast = 1, correlations = FALSE)
    dev.off()
    cat("Saved: 05_heatmap_temporal_cd133.pdf\n")
}, error = function(e) cat("Heatmap error:", e$message, "\n"))

# ==============================================================================
# 8. Export results
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
export_results(results_ctx_temporal, "results/diffbind/DAR_ctx_temporal_all.csv")
export_results(results_ctx_temporal_sig, "results/diffbind/DAR_ctx_temporal_sig.csv")

# Save all objects
save(dba_obj, dba_cd133, dba_all, dba_regional, dba_ctx,
     results_temporal_cd133, results_temporal_cd133_sig,
     results_temporal_all, results_temporal_all_sig,
     results_regional, results_regional_sig,
     results_ctx_temporal, results_ctx_temporal_sig,
     file = "results/diffbind/dba_all_results.RData")

cat("\n=== DONE ===\n")
cat("End time:", format(Sys.time()), "\n")