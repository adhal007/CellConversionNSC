"""GTF parsing utilities."""
import pandas as pd


def parse_gtf_gene_mapping(gtf_path):
    """Parse GTF to create Ensembl ID -> Gene Symbol mapping."""
    ensembl_to_symbol = {}
    
    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            
            attributes = fields[8]
            gene_id = None
            gene_name = None
            
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_id = attr.split('"')[1].split('.')[0]
                elif attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
            
            if gene_id and gene_name:
                ensembl_to_symbol[gene_id] = gene_name
    
    print(f"      Parsed {len(ensembl_to_symbol)} gene mappings")
    return ensembl_to_symbol


def get_promoters_from_gtf(gtf_path, window=2000):
    """Extract promoter regions from GTF."""
    promoters = []
    
    with open(gtf_path) as f:
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
            
            attrs = {}
            for item in fields[8].split(';'):
                if not item.strip():
                    continue
                key_val = item.strip().split(' ', 1)
                if len(key_val) == 2:
                    attrs[key_val[0]] = key_val[1].strip('"')
            
            gene_name = attrs.get('gene_name', '')
            
            if strand == '+':
                tss = start
                promoter_start = max(0, tss - window)
                promoter_end = tss
            else:
                tss = end
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
