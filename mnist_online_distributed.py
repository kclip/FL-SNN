from __future__ import print_function
import datetime
import os

import numpy as np
import tables
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import snn.utils.filters as filters
from snn.utils.utils_snn import refractory_period
from snn.utils.misc import save_results

from utils.training_fl_snn import feedforward_sampling, local_feedback_and_update
from utils.distributed_utils import init_processes, init_training, global_update, global_update_subset, get_acc_and_loss, init_test

""""

Runs FL-SNN using two devices. 

"""


def train_fixed_rate(rank, num_nodes, net_params, args):
    # Create network groups for communication
    all_nodes = dist.new_group([0, 1, 2], timeout=datetime.timedelta(0, 360000))

    # Setup training parameters
    args.dataset = tables.open_file(args.dataset)

    train_data = args.dataset.root.train
    test_data = args.dataset.root.test

    S_prime = int(args.sample_length * 1000 / args.dt)
    S = args.num_samples_train * S_prime

    args, test_indices, test_dict, test_save_path = init_test(rank, args)

    for tau in args.tau_list:
        n_weights_to_send = int(tau * args.rate)

        for _ in range(args.num_ite):
            # Initialize main parameters for training
            network, indices_local, weights_list, eligibility_trace, et_temp, learning_signal, ls_temp = init_training(rank, num_nodes, all_nodes, dataset, labels, net_params)
            samples_indices_train = np.random.choice(indices_local, [args.num_samples_train], replace=True)

            # Gradients accumulator
            gradients_accum = torch.zeros(network.feedforward_weights.shape, dtype=torch.float)
            dist.barrier(all_nodes)

            for s in range(S):
                if rank != 0:
                    if s % S_prime == 0:  # Reset internal state for each example
                        refractory_period(network)
                        inputs, label = get_example(train_data, s // S_prime, S_prime, args.n_classes, args.input_shape, args.dt, args.dataset.root.stats.train_data[1],
                                                    args.polarity)
                        sample = torch.cat((inputs, label), dim=0).to(network.device)

                    # lr decay
                    # if (s + 1) % int(S / 4) == 0:
                    #     args.lr /= 2

                    # Feedforward sampling
                    log_proba, ls_temp, et_temp, gradients_accum = feedforward_sampling(network, sample[:, s % S_prime], ls_temp, et_temp, args, gradients_accum)

                    # Local feedback and update
                    eligibility_trace, et_temp, learning_signal, ls_temp = local_feedback_and_update(network, eligibility_trace, et_temp, learning_signal, ls_temp, s, args)

                # Global update
                if (s + 1) % (tau * args.deltas) == 0:
                    dist.barrier(all_nodes)
                    global_update_subset(all_nodes, rank, network, weights_list, gradients_accum, n_weights_to_send)
                    gradients_accum = torch.zeros(network.feedforward_weights.shape, dtype=torch.float)
                    dist.barrier(all_nodes)

            if rank == 0:
                global_acc, _ = get_acc_and_loss(network, args.dataset, test_indices)
                test_dict[tau].append(global_acc)
                save_results(test_dict, test_save_path)
                print('Tau: %d, final accuracy: %f' % (tau, global_acc))

    if rank == 0:
        save_results(test_dict, test_save_path)
        print('Training finished and accuracies saved to ' + test_save_path)


def train(rank, num_nodes, args):
    # Create network groups for communication
    all_nodes = dist.new_group([0, 1, 2], timeout=datetime.timedelta(0, 360000))

    # Setup training parameters
    args.dataset = tables.open_file(args.dataset)

    train_data = args.dataset.root.train
    test_data = args.dataset.root.test

    S_prime = int(args.sample_length * 1000 / args.dt)
    S = args.num_samples_train * S_prime

    args, test_indices, test_dict, test_save_path = init_test(rank, args)

    for i in range(args.num_ite):
        # Initialize main parameters for training
        network, indices_local, weights_list, eligibility_trace, et_temp, learning_signal, ls_temp = init_training(rank, num_nodes, all_nodes, args)

        dist.barrier(all_nodes)

        # Test loss at beginning + selection of training indices
        if rank != 0:
            print('Node %d' % rank, indices_local)
        else:
            acc, _ = get_acc_and_loss(network, args.dataset, test_indices)
            test_dict[0].append(acc)
            network.train()

        dist.barrier(all_nodes)

        for s in range(S):
            if rank == 0:
                if s % S_prime == 0:
                    if (1 + (s // S_prime)) % args.test_interval == 0:
                        acc, _ = get_acc_and_loss(network, args.dataset, test_indices)
                        test_dict[1 + (s // S_prime)].append(acc)
                        network.train()
                        print('Acc at step %d : %f' % (s, acc))

            dist.barrier(all_nodes)

            if rank != 0:
                if s % S_prime == 0:  # at each example
                    refractory_period(network)
                    inputs, label = get_example(train_data, s // S_prime, S_prime, args.n_classes, args.input_shape, args.dt, args.dataset.root.stats.train_data[1], args.polarity)
                    sample = torch.cat((inputs, label), dim=0).to(network.device)

                # lr decay
                # if s % S / 4 == 0:
                #     args.lr /= 2

                # Feedforward sampling
                log_proba, ls_temp, et_temp, _ = feedforward_sampling(network, sample[:, s % S_prime], ls_temp, et_temp, args)

                # Local feedback and update
                eligibility_trace, et_temp, learning_signal, ls_temp = local_feedback_and_update(network, eligibility_trace, et_temp, learning_signal, ls_temp, s, args)

            # Global update
            if (s + 1) % (args.tau * args.deltas) == 0:
                dist.barrier(all_nodes)
                global_update(all_nodes, rank, network, weights_list)
                dist.barrier(all_nodes)

        # Final global update
        dist.barrier(all_nodes)
        global_update(all_nodes, rank, network, weights_list)
        dist.barrier(all_nodes)

        if rank == 0:
            global_acc, _ = get_acc_and_loss(network, args.dataset, test_indices)
            print('Iteration: %d, final accuracy: %f' % (i, global_acc))
            test_dict[args.num_samples_train].append(global_acc)
        else:
            _, loss = get_acc_and_loss(network, args.dataset, test_indices)
            test_dict[args.num_samples_train].append(loss)
        save_results(test_dict, test_save_path)

    if rank != 0:
        save_results(test_dict, test_save_path)
        print('Training finished and test loss saved to ' + test_save_path)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Train probabilistic SNNs in a distributed fashion using Pytorch')
    # Mandatory arguments
    parser.add_argument('--dist_url', type=str, help='URL to specify the initialization method of the process group')
    parser.add_argument('--node_rank', type=int, help='Rank of the current node')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes to run')
    parser.add_argument('--processes_per_node', default=1, type=int, help='Number of processes in the node')
    parser.add_argument('--dataset', help='Path to the dataset', default='/home/k1804053/datasets/mnist-dvs/mnist_dvs_events.hdf5')
    parser.add_argument('--labels', nargs='+', default=None, type=int)

    # Pytorch arguments
    parser.add_argument('--backend', default='gloo', choices=['gloo', 'nccl', 'mpi', 'tcp'], help='Communication backend to use')

    # Training arguments
    parser.add_argument('--num_ite', default=10, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--num_samples_train', default=200, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--num_samples_test', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--test_interval', default=40, type=int, help='Test interval')
    parser.add_argument('--rate', default=None, type=float, help='Fixed communication rate')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--dt', default=25000, type=int, help='')
    parser.add_argument('--sample_length', default=2000, type=int, help='')
    parser.add_argument('--input_shape', nargs='+', default=[676], type=int, help='Shape of an input sample')
    parser.add_argument('--polarity', default='false', type=str, help='Use polarity or not')

    # SNN arguments
    parser.add_argument('--n_hidden_neurons', default=0, type=int)
    parser.add_argument('--n_basis_ff', default=8, type=int)
    parser.add_argument('--n_basis_fb', default=1, type=int)
    parser.add_argument('--topology_type', default='fully_connected', choices=['fully_connected', 'sparse', 'feedforward'], type=str)
    parser.add_argument('--tau_ff', default=10, type=int, help='Feedforward filter length')
    parser.add_argument('--tau_fb', default=10, type=int, help='Feedback filter length')
    parser.add_argument('--filter', default='raised_cosine_pillow_08', help='filter type')
    parser.add_argument('--mu', default=1.5, type=float, help='Filters width')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--tau', default=1, type=int, help='Global update period.')
    parser.add_argument('--tau_list', nargs='+', default=None, type=int, help='List of update period.')
    parser.add_argument('--kappa', default=0.2, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--beta', default=0.05, type=float, help='Baseline decay coefficient')
    parser.add_argument('--deltas', default=1, type=int, help='Local update period')
    parser.add_argument('--alpha', default=1, type=float, help='KL regularization strength')
    parser.add_argument('--r', default=0.3, type=float, help='Desired hidden neurons spiking rate')
    parser.add_argument('--weights_magnitude', default=0.05, type=float)

    args = parser.parse_args()
    print(args)

    node_rank = args.node_rank + args.node_rank*(args.processes_per_node - 1)
    n_processes = args.processes_per_node
    assert (args.world_size % n_processes == 0), 'Each node must have the same number of processes'
    assert (node_rank + n_processes) <= args.world_size, 'There are more processes specified than world_size'

    args.n_input_neurons = 26**2
    args.n_output_neurons = 10
    args.n_hidden_neurons = args.n_hidden_neurons
    args.n_neurons = args.n_input_neurons + args.n_output_neurons + args.n_hidden_neurons

    filters_dict = {'base_ff_filter': filters.base_feedforward_filter, 'base_fb_filter': filters.base_feedback_filter, 'cosine_basis': filters.cosine_basis,
                    'raised_cosine': filters.raised_cosine, 'raised_cosine_pillow_05': filters.raised_cosine_pillow_05, 'raised_cosine_pillow_08': filters.raised_cosine_pillow_08}

    tau = args.tau
    if args.rate is not None:
        assert args.tau_list is not None, 'rate and tau_list must be specified together'
        tau = None
    if args.tau_list is not None:
        assert args.rate is not None, 'rate and tau_list must be specified together'
        tau = None

    args.synaptic_filter = filters_dict[args.filter]
    args.n_basis_fb = 1

    processes = []
    for local_rank in range(n_processes):
        if args.tau_list is not None:
            p = mp.Process(target=init_processes, args=(node_rank + local_rank, args.world_size, args.backend, args.dist_url, args, train_fixed_rate))
        else:
            p = mp.Process(target=init_processes, args=(node_rank + local_rank, args.world_size, args.backend, args.dist_url, args, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
