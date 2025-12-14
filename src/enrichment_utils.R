# ==============================================================================
# Enrichment Analysis Utilities
# src/enrichment_utils.R
# ==============================================================================

library(clusterProfiler)
library(org.Mm.eg.db)
library(enrichplot)
library(ggplot2)
library(DOSE)

# ==============================================================================
# CONVERSION FUNCTIONS
# ==============================================================================

#' Convert gene symbols or Ensembl IDs to Entrez IDs
#'
#' @param genes Character vector of gene identifiers
#' @param from_type "SYMBOL" or "ENSEMBL"
#' @return Character vector of Entrez IDs
convert_to_entrez <- function(genes, from_type = "SYMBOL") {
    if (length(genes) == 0) return(character(0))
    
    # Remove version numbers if Ensembl
    if (from_type == "ENSEMBL") {
        genes <- gsub("\\..*", "", genes)
    }
    
    # Remove NAs
    genes <- genes[!is.na(genes)]
    
    # Convert
    suppressMessages({
        converted <- bitr(genes, 
                          fromType = from_type, 
                          toType = "ENTREZID", 
                          OrgDb = org.Mm.eg.db)
    })
    
    cat("  Converted", nrow(converted), "of", length(genes), "genes to Entrez IDs\n")
    return(converted$ENTREZID)
}


# ==============================================================================
# ENRICHMENT FUNCTIONS
# ==============================================================================

#' Run GO enrichment analysis
#'
#' @param entrez_ids Character vector of Entrez IDs
#' @param ont Ontology: "BP", "MF", or "CC"
#' @param pval_cutoff P-value cutoff
#' @param qval_cutoff Q-value cutoff
#' @return enrichResult object or NULL
run_go_enrichment <- function(entrez_ids, ont = "BP", pval_cutoff = 0.01, qval_cutoff = 0.1) {
    if (length(entrez_ids) == 0) {
        cat("  No genes provided for GO enrichment\n")
        return(NULL)
    }
    
    result <- enrichGO(gene = entrez_ids,
                       OrgDb = org.Mm.eg.db,
                       ont = ont,
                       pAdjustMethod = "BH",
                       pvalueCutoff = pval_cutoff,
                       qvalueCutoff = qval_cutoff,
                       readable = TRUE)
    
    if (!is.null(result) && nrow(result) > 0) {
        cat("  GO", ont, ":", nrow(result), "enriched terms\n")
    } else {
        cat("  GO", ont, ": No significant terms\n")
    }
    
    return(result)
}


#' Run KEGG pathway enrichment
#'
#' @param entrez_ids Character vector of Entrez IDs
#' @param pval_cutoff P-value cutoff
#' @param qval_cutoff Q-value cutoff
#' @return enrichResult object or NULL
run_kegg_enrichment <- function(entrez_ids, pval_cutoff = 0.01, qval_cutoff = 0.1) {
    if (length(entrez_ids) == 0) {
        cat("  No genes provided for KEGG enrichment\n")
        return(NULL)
    }
    
    result <- enrichKEGG(gene = entrez_ids,
                         organism = 'mmu',
                         pAdjustMethod = "BH",
                         pvalueCutoff = pval_cutoff,
                         qvalueCutoff = qval_cutoff)
    
    if (!is.null(result) && nrow(result) > 0) {
        cat("  KEGG:", nrow(result), "enriched pathways\n")
    } else {
        cat("  KEGG: No significant pathways\n")
    }
    
    return(result)
}


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

#' Generate enrichment plots
#'
#' @param enrich_result enrichResult object
#' @param output_prefix Output file prefix (without extension)
#' @param title Plot title
#' @param plot_types Character vector: "dotplot", "cnetplot", "barplot", "emapplot"
#' @param show_category Number of categories to show
plot_enrichment <- function(enrich_result, output_prefix, title, 
                            plot_types = c("dotplot", "cnetplot"), 
                            show_category = 20) {
    
    if (is.null(enrich_result) || nrow(enrich_result) == 0) {
        cat("  Skipping plots for", title, "- no significant terms\n")
        return(invisible(NULL))
    }
    
    n_terms <- nrow(enrich_result)
    
    # Dotplot
    if ("dotplot" %in% plot_types) {
        tryCatch({
            pdf(paste0(output_prefix, "_dotplot.pdf"), width = 10, height = 8)
            p <- dotplot(enrich_result, 
                         showCategory = min(show_category, n_terms), 
                         title = title)
            print(p)
            dev.off()
            cat("  Saved:", basename(paste0(output_prefix, "_dotplot.pdf")), "\n")
        }, error = function(e) cat("  Dotplot error:", e$message, "\n"))
    }
    
    # Cnetplot (gene-concept network)
    if ("cnetplot" %in% plot_types) {
        tryCatch({
            pdf(paste0(output_prefix, "_cnetplot.pdf"), width = 14, height = 12)
            p <- cnetplot(enrich_result, 
                          showCategory = min(10, n_terms),
                          categorySize = "pvalue") +
                 ggtitle(paste0(title, " - Gene-Concept Network"))
            print(p)
            dev.off()
            cat("  Saved:", basename(paste0(output_prefix, "_cnetplot.pdf")), "\n")
        }, error = function(e) cat("  Cnetplot error:", e$message, "\n"))
    }
    
    # Barplot
    if ("barplot" %in% plot_types) {
        tryCatch({
            pdf(paste0(output_prefix, "_barplot.pdf"), width = 10, height = 8)
            p <- barplot(enrich_result, 
                         showCategory = min(show_category, n_terms),
                         title = title)
            print(p)
            dev.off()
            cat("  Saved:", basename(paste0(output_prefix, "_barplot.pdf")), "\n")
        }, error = function(e) cat("  Barplot error:", e$message, "\n"))
    }
    
    # Enrichment map (similarity network)
    if ("emapplot" %in% plot_types && n_terms >= 2) {
        tryCatch({
            enrich_result_pt <- pairwise_termsim(enrich_result)
            pdf(paste0(output_prefix, "_emapplot.pdf"), width = 12, height = 10)
            p <- emapplot(enrich_result_pt, 
                          showCategory = min(30, n_terms)) +
                 ggtitle(paste0(title, " - Enrichment Map"))
            print(p)
            dev.off()
            cat("  Saved:", basename(paste0(output_prefix, "_emapplot.pdf")), "\n")
        }, error = function(e) cat("  Emapplot error:", e$message, "\n"))
    }
    
    return(invisible(NULL))
}


# ==============================================================================
# PIPELINE FUNCTIONS
# ==============================================================================

#' Run complete enrichment analysis for a gene set
#'
#' @param genes Gene identifiers (symbols or Ensembl)
#' @param gene_type "SYMBOL" or "ENSEMBL"
#' @param output_dir Output directory
#' @param prefix File prefix
#' @param title_prefix Title prefix for plots
#' @param ontologies GO ontologies to test: c("BP", "MF", "CC")
#' @param run_kegg Whether to run KEGG analysis
#' @param plot_types Plot types to generate
#' @param simplify_bp Whether to simplify BP terms
#' @return List of enrichment results
run_enrichment_pipeline <- function(genes, 
                                     gene_type = "SYMBOL",
                                     output_dir,
                                     prefix,
                                     title_prefix,
                                     ontologies = c("BP", "MF"),
                                     run_kegg = TRUE,
                                     plot_types = c("dotplot", "cnetplot"),
                                     simplify_bp = TRUE) {
    
    cat("\n", paste(rep("=", 70), collapse = ""), "\n")
    cat("ENRICHMENT ANALYSIS:", title_prefix, "\n")
    cat("  Input genes:", length(genes), "\n")
    cat(paste(rep("=", 70), collapse = ""), "\n")
    
    # Create output directory
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Convert to Entrez
    cat("\nConverting to Entrez IDs...\n")
    entrez_ids <- convert_to_entrez(genes, from_type = gene_type)
    
    if (length(entrez_ids) == 0) {
        cat("ERROR: No genes converted to Entrez IDs\n")
        return(NULL)
    }
    
    results <- list(
        input_genes = genes,
        entrez_ids = entrez_ids
    )
    
    # GO enrichment for each ontology
    for (ont in ontologies) {
        cat("\nRunning GO", ont, "enrichment...\n")
        
        go_result <- run_go_enrichment(entrez_ids, ont = ont)
        results[[paste0("GO_", ont)]] <- go_result
        
        if (!is.null(go_result) && nrow(go_result) > 0) {
            # Save CSV
            write.csv(as.data.frame(go_result), 
                      file.path(output_dir, paste0(prefix, "_GO_", ont, ".csv")),
                      row.names = FALSE)
            
            # Generate plots
            plot_enrichment(go_result,
                           output_prefix = file.path(output_dir, paste0(prefix, "_GO_", ont)),
                           title = paste0(title_prefix, " - GO ", ont),
                           plot_types = plot_types)
            
            # Simplified version for BP
            if (ont == "BP" && simplify_bp) {
                tryCatch({
                    go_simp <- simplify(go_result, cutoff = 0.7, by = "p.adjust")
                    results[[paste0("GO_", ont, "_simplified")]] <- go_simp
                    
                    if (nrow(go_simp) > 0) {
                        write.csv(as.data.frame(go_simp),
                                  file.path(output_dir, paste0(prefix, "_GO_", ont, "_simplified.csv")),
                                  row.names = FALSE)
                        
                        plot_enrichment(go_simp,
                                       output_prefix = file.path(output_dir, paste0(prefix, "_GO_", ont, "_simplified")),
                                       title = paste0(title_prefix, " - GO ", ont, " (Simplified)"),
                                       plot_types = plot_types)
                    }
                }, error = function(e) cat("  Simplify error:", e$message, "\n"))
            }
        }
    }
    
    # KEGG enrichment
    if (run_kegg) {
        cat("\nRunning KEGG enrichment...\n")
        kegg_result <- run_kegg_enrichment(entrez_ids)
        results[["KEGG"]] <- kegg_result
        
        if (!is.null(kegg_result) && nrow(kegg_result) > 0) {
            write.csv(as.data.frame(kegg_result),
                      file.path(output_dir, paste0(prefix, "_KEGG.csv")),
                      row.names = FALSE)
            
            plot_enrichment(kegg_result,
                           output_prefix = file.path(output_dir, paste0(prefix, "_KEGG")),
                           title = paste0(title_prefix, " - KEGG"),
                           plot_types = c("dotplot", "barplot"))
        }
    }
    
    # Save RData
    save(results, file = file.path(output_dir, paste0(prefix, "_enrichment.RData")))
    cat("\nSaved:", file.path(output_dir, paste0(prefix, "_enrichment.RData")), "\n")
    
    return(results)
}


#' Compare multiple gene sets with enrichment
#'
#' @param gene_lists Named list of gene vectors
#' @param gene_type "SYMBOL" or "ENSEMBL"
#' @param output_dir Output directory
#' @param prefix File prefix
#' @param title Plot title
#' @param ontologies GO ontologies to compare
#' @param run_kegg Whether to run KEGG comparison
#' @return List of compareClusterResult objects
compare_enrichment <- function(gene_lists, 
                               gene_type = "SYMBOL",
                               output_dir,
                               prefix,
                               title,
                               ontologies = c("BP", "MF"),
                               run_kegg = TRUE) {
    
    cat("\n", paste(rep("=", 70), collapse = ""), "\n")
    cat("COMPARATIVE ENRICHMENT:", title, "\n")
    cat(paste(rep("=", 70), collapse = ""), "\n")
    
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Convert all gene lists to Entrez
    cat("\nConverting gene lists to Entrez IDs...\n")
    entrez_lists <- lapply(names(gene_lists), function(name) {
        cat("  ", name, ":", length(gene_lists[[name]]), "genes -> ")
        ids <- convert_to_entrez(gene_lists[[name]], from_type = gene_type)
        ids
    })
    names(entrez_lists) <- names(gene_lists)
    
    # Remove empty lists
    entrez_lists <- entrez_lists[sapply(entrez_lists, length) > 0]
    
    if (length(entrez_lists) < 2) {
        cat("ERROR: Need at least 2 non-empty gene lists for comparison\n")
        return(NULL)
    }
    
    results <- list()
    
    # Compare GO for each ontology
    for (ont in ontologies) {
        cat("\nComparing GO", ont, "...\n")
        tryCatch({
            compare_result <- compareCluster(entrez_lists, 
                                              fun = "enrichGO",
                                              OrgDb = org.Mm.eg.db,
                                              ont = ont,
                                              pAdjustMethod = "BH",
                                              pvalueCutoff = 0.01,
                                              readable = TRUE)
            results[[paste0("GO_", ont)]] <- compare_result
            
            if (!is.null(compare_result) && nrow(compare_result) > 0) {
                write.csv(as.data.frame(compare_result),
                          file.path(output_dir, paste0(prefix, "_compare_GO_", ont, ".csv")),
                          row.names = FALSE)
                
                pdf(file.path(output_dir, paste0(prefix, "_compare_GO_", ont, "_dotplot.pdf")), 
                    width = 14, height = 10)
                p <- dotplot(compare_result, showCategory = 10, 
                             title = paste0(title, " - GO ", ont))
                print(p)
                dev.off()
                cat("  Saved: ", prefix, "_compare_GO_", ont, "_dotplot.pdf\n", sep = "")
            }
        }, error = function(e) cat("  Compare GO", ont, "error:", e$message, "\n"))
    }
    
    # Compare KEGG
    if (run_kegg) {
        cat("\nComparing KEGG...\n")
        tryCatch({
            compare_kegg <- compareCluster(entrez_lists,
                                            fun = "enrichKEGG",
                                            organism = 'mmu',
                                            pAdjustMethod = "BH",
                                            pvalueCutoff = 0.01)
            results[["KEGG"]] <- compare_kegg
            
            if (!is.null(compare_kegg) && nrow(compare_kegg) > 0) {
                write.csv(as.data.frame(compare_kegg),
                          file.path(output_dir, paste0(prefix, "_compare_KEGG.csv")),
                          row.names = FALSE)
                
                pdf(file.path(output_dir, paste0(prefix, "_compare_KEGG_dotplot.pdf")),
                    width = 14, height = 10)
                p <- dotplot(compare_kegg, showCategory = 10, 
                             title = paste0(title, " - KEGG"))
                print(p)
                dev.off()
                cat("  Saved:", paste0(prefix, "_compare_KEGG_dotplot.pdf"), "\n")
            }
        }, error = function(e) cat("  Compare KEGG error:", e$message, "\n"))
    }
    
    # Save results
    save(results, file = file.path(output_dir, paste0(prefix, "_compare_enrichment.RData")))
    
    return(results)
}


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

#' Load TF list from CSV file
#'
#' @param filepath Path to CSV file with 'symbol' column
#' @return Character vector of gene symbols
load_tf_list <- function(filepath) {
    if (!file.exists(filepath)) {
        cat("WARNING: File not found:", filepath, "\n")
        return(character(0))
    }
    df <- read.csv(filepath)
    if ("symbol" %in% colnames(df)) {
        return(df$symbol)
    } else {
        return(df[[1]])
    }
}


#' Load significant genes from DESeq2 results
#'
#' @param filepath Path to DESeq2 CSV file
#' @param direction "up", "down", or "both"
#' @param padj_thresh Adjusted p-value threshold
#' @param lfc_thresh Log2 fold change threshold
#' @return Character vector of Ensembl IDs
load_deseq_genes <- function(filepath, direction = "both", padj_thresh = 0.05, lfc_thresh = 1) {
    if (!file.exists(filepath)) {
        cat("WARNING: File not found:", filepath, "\n")
        return(character(0))
    }
    
    deseq <- read.csv(filepath, row.names = 1)
    
    # Handle NA values
    deseq <- deseq[!is.na(deseq$padj), ]
    
    if (direction == "up") {
        sig <- deseq[deseq$padj < padj_thresh & deseq$log2FoldChange > lfc_thresh, ]
    } else if (direction == "down") {
        sig <- deseq[deseq$padj < padj_thresh & deseq$log2FoldChange < -lfc_thresh, ]
    } else {
        sig <- deseq[deseq$padj < padj_thresh & abs(deseq$log2FoldChange) > lfc_thresh, ]
    }
    
    cat("  Loaded", nrow(sig), direction, "genes from", basename(filepath), "\n")
    return(rownames(sig))
}


#' Convert Ensembl IDs to Symbols
#'
#' @param ensembl_ids Character vector of Ensembl IDs
#' @return Character vector of gene symbols
ensembl_to_symbol <- function(ensembl_ids) {
    ensembl_ids <- gsub("\\..*", "", ensembl_ids)
    suppressMessages({
        converted <- bitr(ensembl_ids, 
                          fromType = "ENSEMBL", 
                          toType = "SYMBOL", 
                          OrgDb = org.Mm.eg.db)
    })
    return(converted$SYMBOL)
}