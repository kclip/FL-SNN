#!/usr/bin/env bash
# Get the number of nodes
NUM_WORKER=$(gcloud compute instances list | grep -E '^worker-[0-9]+ ' | wc -l)
WORLD_SIZE=$(($NUM_WORKER+1))
PROCESSES_PER_NODE=1

# Launch workers
for  i in $(seq 1 $NUM_WORKER); do
  j=$(( $i - 1 ))
  sh local_train.sh worker-${j} $WORLD_SIZE ${i} $PROCESSES_PER_NODE &
done

# Launch master node
sh local_train.sh master-0 $WORLD_SIZE 0 $PROCESSES_PER_NODE
