#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research




#---------------------------------- ICARL Experiments ------------------------------------#

for i in 1 
do
    #---------------------------------- Dense Experiments ------------------------------------#

    # #---------------------------------- MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-mnist" --model "icarl" --optim "sgd" --lr 0.1 --n_epochs 20 --batch_size 128 \
    # --seed 0 --notes "iCarl for MNIST" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 128 --optim_wd 0.00001 --prune_perc 0.1

    # #---------------------------------- EF-MNIST Experiments ------------------------------------#
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    --dataset "seq-efmnist" --model "icarl" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 32 \
    --seed 0 --notes "iCarl for EFMNIST" --non_verbose  --csv_log  \
    --buffer_size 5700  --minibatch_size 32 --optim_wd 0.00001 --prune_perc 0.1

    #---------------------------------- Cifar10 Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-cifar10" --model "icarl" --optim "sgd" --lr 0.05 --n_epochs 50 --batch_size 64 \
    # --seed 0 --notes "iCarl for cifar10" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 64 --optim_wd 0.00001  --prune_perc 0.1


    #---------------------------------- Sparse Experiments ------------------------------------#

    #---------------------------------- MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-mnist" --model "icarl" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 256 \
    # --seed 0 --notes "iCarl for MNIST" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 256 --optim_wd 0.00001 --prune_perc 90

    # #---------------------------------- EF-MNIST Experiments ------------------------------------#
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    --dataset "seq-efmnist" --model "icarl" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 32 \
    --seed 0 --notes "iCarl for EFMNIST" --non_verbose  --csv_log  \
    --buffer_size 5700  --minibatch_size 32 --optim_wd 0.00001  --prune_perc 70


    #---------------------------------- Cifar10 Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-cifar10" --model "icarl" --optim "sgd" --lr 0.1 --n_epochs 100 --batch_size 128 \
    # --seed 0 --notes "iCarl for cifar10" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 128 --optim_wd 0.00001  --prune_perc 90
done



exit 0