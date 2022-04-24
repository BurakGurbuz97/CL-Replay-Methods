#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research




#---------------------------------- ICARL Experiments ------------------------------------#


for LR in  0.05 0.1
do
    for BS in 128 256 512 1024
    do
        MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
        --dataset "seq-efmnist" --model "icarl" --optim "sgd" --lr ${LR} --n_epochs 25 --batch_size ${BS} \
        --seed 0 --notes "iCarl for EFMNIST" --non_verbose  --csv_log  \
        --buffer_size 5700  --minibatch_size ${BS} --optim_wd 0
    done
done


for LR in  0.05 0.1
do
    for BS in 128 256 512
    do
        MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
        --dataset "seq-mnist" --model "icarl" --optim "sgd" --lr ${LR} --n_epochs 10 --batch_size ${BS} \
        --seed 0 --notes "iCarl for MNIST" --non_verbose  --csv_log  \
        --buffer_size 1000  --minibatch_size ${BS} --optim_wd 0
    done
done

exit 0