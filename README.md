# FL-SNN
Code for Federated Neuromorphic Learning of Spiking Neural Networks for Low-Power Edge Intelligence (arXiv version: https://arxiv.org/abs/1910.09594)

After a recent refactor, to run, this code requires to install our `snn` package, which can be found at https://github.com/kclip/snn

# Run example
An experiment can be run on the MNIST-DVS dataset by launching on each node

`python mnist_online_distributed.py --dist_url='tcp://master-ip --world_size=3 --node_rank=#rank --processes_per_node=1 `

Make sure to first download and preprocess the MNIST-DVS dataset using the script in `snn/data_preprocessing/process_mnistdvs.py` and change your home directory in `snn/launch_experiment.py`.

