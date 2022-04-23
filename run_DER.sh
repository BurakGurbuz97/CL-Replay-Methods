#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research

MKL_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1 CUBLAS_WORKSPACE_CONFIG=:16:8  python main.py \
--dataset "seq-mnist" --model "der" --optim "sgd" --lr 0.01 --n_epochs 10 --batch_size 128 \
--seed 0 --notes "Simple ER for MNIST" --non_verbose  --csv_log  \
--buffer_size 1000  --minibatch_size 128 --alpha 1.0
  

exit 0