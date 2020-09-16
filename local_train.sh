#!/usr/bin/env bash
gcloud compute ssh --zone europe-west1-b $1 \
--command "cd /home/k1804053/FL-SNN && /opt/conda/bin/python mnist_online_distributed.py --dist_url='tcp://10.132.0.9:23456' \
 --world_size=$2 --node_rank=$3 --processes_per_node=$4 --tau=1 --test_interval=500 --num_samples_train=5000 \
  --n_hidden_neurons=128 --num_ite=1 --save_path=/home/k1804053/results/results_flsnn/ --lr=0.0001 --home=/home/k1804053/"
