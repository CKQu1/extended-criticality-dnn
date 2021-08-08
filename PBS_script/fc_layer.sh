#!/bin/bash
#PBS -P phys_DL
#PBS -N fc_layer_10-20
#PBS -q defaultQ

## GPU
#PBS -l select=1:ncpus=2:ngpus=2:mem=32gb			

## CPU
##PBS -l select=1:ncpus=4:mem=64gb

#PBS -l walltime=47:59:59
#PBS -o PBSout/
#PBS -e PBSout/
##PBS -J 1-4

# GPU (don't need if CPU)
#module load cuda/9.1.85

# python
#module load python/3.6.5 cuda/10.0.130 openmpi-gcc/3.1.3-cuda10
#source ~/pytorch/bin/activate

DATA_DIR="/project/phys_DL/Anomalous-diffusion-dynamics-of-SGD"
cd ${DATA_DIR}

#params=`sed "${PBS_ARRAY_INDEX}q;d" job_params_bs`
#param_array=( $params )

#params=`sed "${PBS_ARRAY_INDEX}q;d" job_params_model`
#param_array=( $params )

# matlab
#module load matlab/R2019a
#--batch_size=${param_array[0]}
# training DNN
# --save_epoch=5
# --loss_name=mse 
# --optimizer=adam
# --batch_size=128
# --momentum=0.9 
# --lr_decay=0.5
# --dataset=mnist
# --weight_decay=0.0005
# --epoch_start=1  --epoch_last=250

# python modules
module load zlib/1.2.8 python/3.7.7 gmp/5.1.3 mpc/1.0.3 openmpi-gcc/3.1.5 magma/2.5.3 git/2.25.0 sqlite/3080802  cuda/10.2.89 mpfr/3.1.4 gcc/7.4.0 lapack/3.9.0 openssl/1.1.1d

# fc15 gaussian

python -m main_last_epoch_2  --ngpu=2  --model=fc15_mnist_tanh  --w_std=0.1  --save_epoch=5  --epochs=650  --batch_size=64  --lr=0.001  --dataset=mnist

# fc15 stable

python -m main_last_epoch_2  --ngpu=2  --model=fc15_mnist_tanh  --init_dist=stable  --init_alpha=1.5  --init_scale_multiplier=0.1  --save_epoch=5  --epochs=650  --batch_size=64  --lr=0.001  --dataset=mnist




