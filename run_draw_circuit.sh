#!/bin/bash
#SBATCH --job-name=qd_draw_circuit
#SBATCH --partition=UGGPU-TC1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_draw_circuit_%j.log
#SBATCH --error=logs/slurm_draw_circuit_%j.err
cd /tc1home/UG/zgoh033/quantum_doom && source .venv/bin/activate
export MPLCONFIGDIR=/tmp/matplotlib-qd-$SLURM_JOB_ID
python draw_circuit.py circuit_diagram
