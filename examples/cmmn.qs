#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --job-name=cmmn_mapping_cue_to_emotion
# The below is maximum time for the job.
#SBATCH --time=7-00:00:00
# SBATCH --time-min=0-01:00:00
#SBATCH --mail-user='ajmeek@udel.edu'
# this could be --mail-type=END, FAIL, TIME_LIMIT_90. but I thought it was too many emails
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --export=NONE
#SBATCH --mem=128G
# SBATCH --array=1-12
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"

vpkg_devrequire intel-python/2022u1:python3
# source activate /work/cniel/ajmeek/bowaves_cmmn/convolutional-monge-mapping-normalization/venv/
# pip install -r ../requirements.txt

# check version
python --version # why is this python 2??

# Run bash / python script below

python cmmn_hpc.py