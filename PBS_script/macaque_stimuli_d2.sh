#!/bin/bash
#PBS -P vortex_dl
#PBS -l select=1:ncpus=1:ngpus=0:mem=32gb
#PBS -l walltime=47:59:59
#PBS -e /project/PDLAI/project2_data/macaque_stimuli/job
#PBS -o /project/PDLAI/project2_data/macaque_stimuli/job

PBS_O_WORKDIR="/project/phys_DL/extended-criticality-dnn"
cd ${PBS_O_WORKDIR}
source virt-test-qu/bin/activate

python fewshot-learning/pretrained_macaque_stimuli.py snr_d2_mbatch wide_resnet50_2 False