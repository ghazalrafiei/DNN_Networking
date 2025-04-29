#!/bin/bash
#SBATCH --job-name=dns-pcap-stream
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --time=5:00:00
#SBATCH --mem=32G


# Make sure log/output directories exist
mkdir -p logs csv_files

echo "======== JOB STARTED AT $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a logs/pipeline_status.log

# Run your Python processing pipeline
module load cuda/11.7
module load cudnn/8.3
source activate my_pytorch_gpu_env
PYTHONUNBUFFERED=1 python3 traffic_prediction_using_TransformerEncoder.py
status=$?

# Timestamped end
if [ $status -eq 0 ]; then
    echo "======== JOB FINISHED SUCCESSFULLY AT $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a logs/pipeline_status.log
else
    echo "======== JOB FAILED (exit code $status) AT $(date '+%Y-%m-%d %H:%M:%S') ========" | tee -a logs/pipeline_status.log
fi