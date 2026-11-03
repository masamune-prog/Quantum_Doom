#!/bin/bash
#SBATCH --job-name=qd_teacher_test
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_teacher_test_%j.log
#SBATCH --error=logs/slurm_teacher_test_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
# CPU only. Scripted teacher exit rate from the real start at skills 3 and 1.
python test_e1m1_teacher.py 3 100
python test_e1m1_teacher.py 1 30
