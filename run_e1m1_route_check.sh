#!/bin/bash
#SBATCH --job-name=qd_e1m1_route
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_e1m1_route_%j.log
#SBATCH --error=logs/slurm_e1m1_route_%j.err

echo "=== Quantum Doom - E1M1 Route Check (scripted follower) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

cd /tc1home/UG/zgoh033/quantum_doom
set -eo pipefail
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python validate_e1m1_route.py --skills 1 3 --episodes 5

echo "=== Done! End: $(date) ==="
