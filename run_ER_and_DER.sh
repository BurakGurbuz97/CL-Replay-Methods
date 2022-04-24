#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research

for OPTIM in "sgd" "adam"
do
    for LR in  0.05 0.10 0.01
    do
        for BS in 32 64
        do
            MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
            --dataset "seq-cifar10" --model "er" --optim ${OPTIM} --lr ${LR} --n_epochs 30 --batch_size ${BS} \
            --seed 0 --notes "Simple ER for CIFAR10" --non_verbose  --csv_log  \
            --buffer_size 1000  --minibatch_size ${BS}
        done
    done
done

#---------------------------------- DER Experiments ------------------------------------#

for OPTIM in "sgd" "adam"
do
    for A in  1 0.3
        do
        for LR in  0.03 0.1 0.01
        do
            for BS in 32 64
            do
                MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
                --dataset "seq-cifar10" --model "der" --optim ${OPTIM} --lr ${LR} --n_epochs 30 --batch_size ${BS} \
                --seed 0 --notes "DER for Cifar10" --non_verbose  --csv_log  \
                --buffer_size 1000  --minibatch_size ${BS} --alpha ${A}
            done
        done
    done
done

exit 0