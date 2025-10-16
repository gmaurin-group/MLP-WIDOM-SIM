#!/bin/bash
##!/bin/bash
#SBATCH --partition=workq
#SBATCH --job-name=cp2k
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --account=k10149
#SBATCH --err=std.err
#SBATCH --output=std.out
#----------------------------------------------------------#
module load cp2k/2023.2
source ${CP2K_HOME}/tools/toolchain/install/setup
export FI_CXI_RX_MATCH_MODE=software
export OMP_NUM_THREADS=1
#----------------------------------------------------------#
echo "The job "${SLURM_JOB_ID}" is running on "${SLURM_JOB_NODELIST}
#----------------------------------------------------------#

srun --ntasks=192 --map-by-numa --hint=nomultithread ${CP2K_HOME}/exe/local/cp2k.psmp GEO_OPT.inp > geo_opt.outp
