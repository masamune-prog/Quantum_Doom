#!/bin/bash
#SBATCH --job-name=qd_ablation_test
#SBATCH --partition=UGGPU-TC1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_ablation_test_%j.log
#SBATCH --error=logs/slurm_ablation_test_%j.err
cd /tc1home/UG/zgoh033/quantum_doom && source .venv/bin/activate && python test_circuit_ablations.py
