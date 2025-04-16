#!/bin/bash
#SBATCH --job-name=dns-pcap-stream
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --partition=short   # Or medium / long if needed

# spack
source /nas/eclairnas01/shared/spack/latest/share/spack/setup-env.sh
spack env activate /nas/home/shefalit/wireshark-env
#spack add tcpdump

# Activate your environment
source /nas/eclairnas01/users/shefalit/working/code/myenv/bin/activate   # Or conda activate myenv if using conda
export PATH=$HOME/local/bin:$PATH

# Make sure log/output directories exist
mkdir -p logs csv_files

echo "======== JOB STARTED AT $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a logs/pipeline_status.log

# Run your Python processing pipeline
python3 data_preprocess.py --day 14
status=$?

# Timestamped end
if [ $status -eq 0 ]; then
    echo "======== JOB FINISHED SUCCESSFULLY AT $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a logs/pipeline_status.log
else
    echo "======== JOB FAILED (exit code $status) AT $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a logs/pipeline_status.log
fi
