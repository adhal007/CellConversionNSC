#!/usr/bin/env Rscript

library(DiffBind)
library(tidyverse)

cat("Starting DiffBind analysis...\n")
cat(date(), "\n")

# ==============================================================================
# SETUP
# ==============================================================================

base_dir <- "/mnt/lscratch/users/adhal/CorticalNeuronFate/CellConversionNSC"
output_dir <- file.path(base_dir, "results/diffbind_cd133")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Starting DiffBind Analysis ===\n")
cat("Time:", format(Sys.time()), "\n\n")

# ==============================================================================
# LOAD SAMPLE SHEET
# ==============================================================================

cat("Loading sample sheet...\n")
samples <- read.csv(file.path(base_dir, "data/atac_seq/ATAC-seq/samples_all_fixed.csv"))

# Show what we loaded
print(samples[, c("SampleID", "Tissue", "Factor", "Condition")])

# Sample distribution
cat("\nSample distribution:\n")
print(table(samples$Tissue, samples$Condition, samples$Factor))

# ==============================================================================
# CREATE DIFFBIND OBJECT (ALL CD133 SAMPLES)
# ==============================================================================

cat("\nCreating DiffBind object (all CD133)...\n")

# All CD133 samples
samples_cd133 <- samples

dba_all <- dba(sampleSheet = samples_cd133)
print(dba_all)

# ==============================================================================
# COUNT READS
# ==============================================================================

cat("\n=== Counting reads in peaks ===\n")
cat("Start time:", format(Sys.time()), "\n")

dba_counted <- dba.count(dba_all, summits = 250)

cat("End time:", format(Sys.time()), "\n")

# Save checkpoint
save(dba_counted, file = file.path(output_dir, "dba_counted_cd133_all.RData"))
cat("Saved checkpoint: dba_counted_cd133_all.RData\n")

# ==============================================================================
# ANALYSIS 1: ALL CD133 (E14 vs E18)
# ==============================================================================

cat("\n=== Running CD133 E14 vs E18 (all tissues) ===\n")

# Set contrast
dba_contrast_cd133 <- dba.contrast(dba_counted, 
                                   categories = DBA_CONDITION,
                                   minMembers = 2)

# Show contrasts
print(dba_contrast_cd133)

# Analyze
dba_analyzed_cd133 <- dba.analyze(dba_contrast_cd133, method = DBA_DESEQ2)

# Get results
results_cd133 <- dba.report(dba_analyzed_cd133, th = 1, bCounts = TRUE)
results_cd133_df <- as.data.frame(results_cd133)

# Filter significant
results_cd133_sig <- results_cd133_df[results_cd133_df$FDR < 0.05 & abs(results_cd133_df$Fold) > 1, ]

cat("\nCD133 All Tissues (E14 vs E18):\n")
cat("  Total peaks tested:", nrow(results_cd133_df), "\n")
cat("  Significant (FDR<0.05, |LFC|>1):", nrow(results_cd133_sig), "\n")
cat("  E14-high:", sum(results_cd133_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_cd133_sig$Fold < 0), "\n")

# Save results
write.csv(results_cd133_df, file.path(output_dir, "DAR_cd133_all_tissues.csv"), row.names = FALSE)
write.csv(results_cd133_sig, file.path(output_dir, "DAR_cd133_all_tissues_sig.csv"), row.names = FALSE)

# ==============================================================================
# ANALYSIS 2: CD133 CORTEX ONLY (E14 vs E18)
# ==============================================================================

cat("\n=== Running CD133 Cortex E14 vs E18 ===\n")

# Filter to Cortex only
samples_cortex <- samples[samples$Tissue == "Cortex", ]

cat("\nCortex CD133 samples:\n")
print(samples_cortex[, c("SampleID", "Tissue", "Factor", "Condition")])

# Create DiffBind object for Cortex only
dba_cortex <- dba(sampleSheet = samples_cortex)
print(dba_cortex)

# Count reads
cat("\nCounting reads for Cortex samples...\n")
dba_cortex_counted <- dba.count(dba_cortex, summits = 250)

# Save checkpoint
save(dba_cortex_counted, file = file.path(output_dir, "dba_counted_cd133_cortex.RData"))
cat("Saved checkpoint: dba_counted_cd133_cortex.RData\n")

# Set contrast
dba_cortex_contrast <- dba.contrast(dba_cortex_counted,
                                    categories = DBA_CONDITION,
                                    minMembers = 2)

print(dba_cortex_contrast)

# Analyze
dba_cortex_analyzed <- dba.analyze(dba_cortex_contrast, method = DBA_DESEQ2)

# Get results
results_cortex <- dba.report(dba_cortex_analyzed, th = 1, bCounts = TRUE)
results_cortex_df <- as.data.frame(results_cortex)

# Filter significant
results_cortex_sig <- results_cortex_df[results_cortex_df$FDR < 0.05 & abs(results_cortex_df$Fold) > 1, ]

cat("\nCD133 Cortex (E14 vs E18):\n")
cat("  Total peaks tested:", nrow(results_cortex_df), "\n")
cat("  Significant (FDR<0.05, |LFC|>1):", nrow(results_cortex_sig), "\n")
cat("  E14-high:", sum(results_cortex_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_cortex_sig$Fold < 0), "\n")

# Save results
write.csv(results_cortex_df, file.path(output_dir, "DAR_cd133_cortex.csv"), row.names = FALSE)
write.csv(results_cortex_sig, file.path(output_dir, "DAR_cd133_cortex_sig.csv"), row.names = FALSE)

# ==============================================================================
# GENERATE PLOTS
# ==============================================================================

cat("\n=== Generating plots ===\n")

# MA plot - All CD133
pdf(file.path(output_dir, "MA_plot_cd133_all.pdf"), width = 8, height = 6)
dba.plotMA(dba_analyzed_cd133, main = "CD133 All Tissues: E14 vs E18")
dev.off()

# MA plot - Cortex
pdf(file.path(output_dir, "MA_plot_cd133_cortex.pdf"), width = 8, height = 6)
dba.plotMA(dba_cortex_analyzed, main = "CD133 Cortex: E14 vs E18")
dev.off()

# Volcano plot - All CD133
pdf(file.path(output_dir, "volcano_plot_cd133_all.pdf"), width = 8, height = 6)
dba.plotVolcano(dba_analyzed_cd133, main = "CD133 All Tissues: E14 vs E18")
dev.off()

# Volcano plot - Cortex
pdf(file.path(output_dir, "volcano_plot_cd133_cortex.pdf"), width = 8, height = 6)
dba.plotVolcano(dba_cortex_analyzed, main = "CD133 Cortex: E14 vs E18")
dev.off()

# PCA plot - All CD133
pdf(file.path(output_dir, "PCA_cd133_all.pdf"), width = 8, height = 6)
dba.plotPCA(dba_counted, label = DBA_CONDITION, main = "CD133 All Tissues")
dev.off()

# PCA plot - Cortex
pdf(file.path(output_dir, "PCA_cd133_cortex.pdf"), width = 8, height = 6)
dba.plotPCA(dba_cortex_counted, label = DBA_CONDITION, main = "CD133 Cortex")
dev.off()

# Correlation heatmap - All CD133
pdf(file.path(output_dir, "correlation_heatmap_cd133_all.pdf"), width = 10, height = 10)
dba.plotHeatmap(dba_counted, main = "CD133 All Tissues - Sample Correlation")
dev.off()

# Correlation heatmap - Cortex
pdf(file.path(output_dir, "correlation_heatmap_cd133_cortex.pdf"), width = 8, height = 8)
dba.plotHeatmap(dba_cortex_counted, main = "CD133 Cortex - Sample Correlation")
dev.off()

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("\n" , rep("=", 80), "\n", sep = "")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n", sep = "")

cat("\nResults saved to:", output_dir, "\n")

cat("\nCD133 All Tissues (E14 vs E18):\n")
cat("  Total peaks:", nrow(results_cd133_df), "\n")
cat("  Significant:", nrow(results_cd133_sig), "\n")
cat("  E14-high:", sum(results_cd133_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_cd133_sig$Fold < 0), "\n")

cat("\nCD133 Cortex Only (E14 vs E18):\n")
cat("  Total peaks:", nrow(results_cortex_df), "\n")
cat("  Significant:", nrow(results_cortex_sig), "\n")
cat("  E14-high:", sum(results_cortex_sig$Fold > 0), "\n")
cat("  E18-high:", sum(results_cortex_sig$Fold < 0), "\n")

cat("\nTime:", format(Sys.time()), "\n")
cat("Done!\n")