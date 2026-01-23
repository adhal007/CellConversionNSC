# CellConversionNSC

- Pipeline to integrate RNA-Seq Data and ATAC Seq - using PKNs (Enhancers, Promoters and CHIP Seq) with state of the art statistics (Bidirectional Geometric JSD)
- 


## Directory Structure
```
src/
   ├── io/              ← Data loading & parsing
   ├── engine/          ← Main pipeline
   └── models/          ← Analysis components
       ├── stats/       ← Statistical methods
       ├── grn/         ← GRN construction
       ├── expr/        ← (Future: expression analysis)
       └── atac/        ← (Future: ATAC-seq analysis)
```

```

ModuleResponsibilityKey Componentssrc/io/Data loading, file parsingGTF parsing, BED files, regulatory regionssrc/engine/Main analysis pipelineNSCAnalysis orchestrationsrc/models/stats/Statistical analysisDESeq2, GJSD, differential testssrc/models/grn/GRN constructionCellOracle, motif scanningsrc/models/expr/Expression analysis(Future: normalization, QC)src/models/atac/ATAC-seq analysis(Future: peak calling, DA)
```