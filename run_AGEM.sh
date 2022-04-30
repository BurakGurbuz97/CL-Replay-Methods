#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research

#---------------------------------- A-GEM Experiments ------------------------------------#

#---------------------------------- DENSE Experiments ------------------------------------#


for i in 1 2 
do
    # #---------------------------------- EF-MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-efmnist" --model "agem" --optim "sgd" --lr 0.05 --n_epochs 25 --batch_size 256 \
    # --seed 0 --notes "AGEM for EFMNIST" --non_verbose  --csv_log  \
    # --buffer_size 5700  --minibatch_size 256 --optim_wd 0  --prune_perc 0.1

    # #---------------------------------- MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-mnist" --model "agem" --optim "sgd" --lr 0.1 --n_epochs 10 --batch_size 512 \
    # --seed 0 --notes "AGEM for MNIST" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 512 --optim_wd 0  --prune_perc 0.1

    #---------------------------------- Cifar10 Experiments ------------------------------------#
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    --dataset "seq-cifar10" --model "agem" --optim "sgd" --lr 0.01 --n_epochs 50 --batch_size 32 \
    --seed 0 --notes "AGEM for Cifar10" --non_verbose  --csv_log  \
    --buffer_size 1000  --minibatch_size 32 --optim_wd 0  --prune_perc 0.1


    #---------------------------------- SPARSE Experiments ------------------------------------#

    # #---------------------------------- EF-MNIST Experiments ------------------------------------#
    #  MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    #  --dataset "seq-efmnist" --model "agem" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 256 \
    #  --seed 0 --notes "AGEM for EFMNIST" --non_verbose  --csv_log  \
    #  --buffer_size 5700  --minibatch_size 256 --optim_wd 0  --prune_perc 70

    # #---------------------------------- MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-mnist" --model "agem" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 512 \
    # --seed 0 --notes "AGEM for MNIST" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 512 --optim_wd 0  --prune_perc 90

    #---------------------------------- Cifar10 Experiments ------------------------------------#
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    --dataset "seq-cifar10" --model "agem" --optim "sgd" --lr 0.01 --n_epochs 100 --batch_size 32 \
    --seed 0 --notes "AGEM for Cifar10" --non_verbose  --csv_log  \
    --buffer_size 1000  --minibatch_size 32 --optim_wd 0  --prune_perc 90
done



exit 0