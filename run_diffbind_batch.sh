#!/bin/bash -l
#SBATCH --job-name=diffbind_atac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Safer defaults
set -eo pipefail

# ---- Activate micromamba env ----
if command -v micromamba >/dev/null 2>&1; then
  set +u
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate scrna_R
  set -u
else
  export PATH="/mnt/aiongpfs/users/adhal/micromamba/envs/scrna_R/bin:$PATH"
fi

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE || true

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Sanity check
which R
R --version

echo "Starting DiffBind analysis..."
date

# Run
Rscript /home/users/adhal/CorticalNeuronFate/CellConversionNSC/diffbind_analysis.R

echo "Done!"
date