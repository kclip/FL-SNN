# FL-SNN
Code for Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence (arXiv version: https://arxiv.org/abs/1910.09594)

To run, this code requires our `snn` package, which can be found at https://github.com/kclip/snn
This repo will not be maintained actively, the latest working version of the snn is commit f352ea5042375d8c779f0ef893386d032c2b4c73

# Run example
An experiment can be run on the MNIST-DVS dataset by launching on each node

`python mnist_online_distributed.py --dist_url='tcp://master-ip --world_size=3 --node_rank=#rank --processes_per_node=1 `

Make sure to first download and preprocess the MNIST-DVS dataset using the script in `snn/data_preprocessing/process_mnistdvs.py` and change your home directory in `snn/launch_experiment.py`.

