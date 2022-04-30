#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research

#---------------------------------- DER Experiments ------------------------------------#

for i in 1 2
do
    #---------------------------------- Dense Experiments ------------------------------------#
    # #---------------------------------- MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-mnist" --model "der" --optim "sgd" --lr 0.1 --n_epochs 10 --batch_size 256 \
    # --seed 0 --notes "DER for MNIST" --non_verbose  --csv_log --buffer_size 1000  --minibatch_size 256 --alpha 1 --prune_perc 0.1

    # #---------------------------------- EF-MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-efmnist" --model "der" --optim "sgd" --lr 0.1 --n_epochs 25 --batch_size 256 \
    # --seed 0 --notes "DER for EF-MNIST" --non_verbose  --csv_log  \
    # --buffer_size 5700  --minibatch_size 256 --alpha 0.5  --prune_perc 0.1


    #---------------------------------- Cifar10 Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-cifar10" --model "der" --optim "sgd" --lr 0.03 --n_epochs 50 --batch_size 32 \
    # --seed 0 --notes "DER for Cifar10" --non_verbose  --csv_log  \
    # --buffer_size 1000  --minibatch_size 32 --alpha 1  --prune_perc 0.1



    #---------------------------------- Sparse Experiments ------------------------------------#
    #---------------------------------- MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-mnist" --model "der" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 256 \
    # --seed 0 --notes "DER for MNIST" --non_verbose  --csv_log --buffer_size 1000  --minibatch_size 256 --alpha 1 --prune_perc 90 

    # #---------------------------------- EF-MNIST Experiments ------------------------------------#
    # MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    # --dataset "seq-efmnist" --model "der" --optim "sgd" --lr 0.1 --n_epochs 50 --batch_size 256 \
    # --seed 0 --notes "DER for EF-MNIST" --non_verbose  --csv_log  \
    # --buffer_size 5700  --minibatch_size 256 --alpha 0.5 --prune_perc 70


    #---------------------------------- Cifar10 Experiments ------------------------------------#
    MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
    --dataset "seq-cifar10" --model "der" --optim "sgd" --lr 0.03 --n_epochs 100 --batch_size 32\
    --seed 0 --notes "DER for Cifar10" --non_verbose  --csv_log  \
    --buffer_size 1000  --minibatch_size 32 --alpha 1 --prune_perc 90
done
exit 0