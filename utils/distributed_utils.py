import datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from snn.models.SNN import BinarySNN
from snn.utils import misc, filters, utils_snn



def init_processes(rank, world_size, backend, url, args, train_func):
    """"
    Initialize process group and launches training on the nodes
    """
    dist.init_process_group(backend=backend, init_method=url, timeout=datetime.timedelta(0, 360000), world_size=world_size, rank=rank)
    print('Process %d started' % rank)
    train_func(rank, world_size, args)
    return


def init_test(args):
    args.num_samples_test = args.dataset.root.stats.test_data[0]
    if args.labels is not None:
        print(args.labels)
        num_samples_test = min(args.num_samples_test, len(misc.find_indices_for_labels(args.dataset.root.test, args.labels)))
        test_indices = np.random.choice(misc.find_indices_for_labels(args.dataset.root.test, args.labels), [num_samples_test], replace=False)
    else:
        test_indices = np.random.choice(np.arange(args.dataset.root.stats.test_data[0]), [args.num_samples_test], replace=False)
        args.labels = [i for i in range(10)]

    args.save_path = misc.mksavedir(pre=args.home + args.results_path, exp_dir=args.name)

    S = args.num_samples_train * args.S_prime
    accs = {i: [] for i in range(0, S, args.test_interval)}
    accs[S] = []

    losses = {i: [] for i in range(0, S, args.test_interval)}
    losses[S] = []

    return args, test_indices, losses, accs


def init_training(rank, num_nodes, nodes_group, args):
    """"
    Initializes the different parameters for distributed training
    """
    # Initialize an SNN
    network = BinarySNN(**misc.make_network_parameters(network_type='snn',
                                                       n_input_neurons=args.n_input_neurons,
                                                       n_output_neurons=args.n_output_neurons,
                                                       n_hidden_neurons=args.n_hidden_neurons,
                                                       topology_type='fully_connected',
                                                       topology=None,
                                                       n_neurons_per_layer=0,
                                                       density=1,
                                                       weights_magnitude=0.05,
                                                       initialization='uniform',
                                                       synaptic_filter=args.synaptic_filter,
                                                       n_basis_ff=args.n_basis_ff,
                                                       n_basis_fb=args.n_basis_fb,
                                                       tau_ff=args.tau_ff,
                                                       tau_fb=args.tau_fb,
                                                       mu=args.mu
                                                       ),
                        device='cpu')
    network.train()

    # At the beginning, the master node:
    # - transmits its weights to the workers
    # - distributes the samples among workers
    if rank == 0:
        # Initializing an aggregation list for future weights collection
        weights_list = [[torch.zeros(network.feedforward_weights.shape, dtype=torch.float) for _ in range(num_nodes)],
                        [torch.zeros(network.feedback_weights.shape, dtype=torch.float) for _ in range(num_nodes)],
                        [torch.zeros(network.bias.shape, dtype=torch.float) for _ in range(num_nodes)],
                        [torch.zeros(1, dtype=torch.float) for _ in range(num_nodes)]]
    else:
        weights_list = []

    indices_local = distribute_samples(nodes_group, rank, args)
    dist.barrier(nodes_group)

    # Master node sends its weights
    for parameter in network.get_parameters():
        dist.broadcast(network.get_parameters()[parameter], 0, group=nodes_group)
    if rank == 0:
        print('Node 0 has shared its model and training data is partitioned among workers')

    # The nodes initialize their eligibility trace and learning signal
    learning_signal = 0
    ls_temp = 0
    eligibility_trace = {parameter: network.get_gradients()[parameter] for parameter in network.get_gradients()}
    et_temp = {parameter: network.get_gradients()[parameter] for parameter in network.get_gradients()}

    return network, indices_local, weights_list, eligibility_trace, et_temp, learning_signal, ls_temp


def distribute_samples(nodes, rank, args):
    """
    The master node (rank 0) randomly chooses and transmits samples indices to each device for training.
    Upon reception of their assigned samples, the nodes create their training dataset
    """

    if rank == 0:
        # Indices corresponding to each class
        indices_worker_0 = np.zeros([args.num_samples_train])
        indices_worker_1 = np.zeros([args.num_samples_train])

        num_samples_per_class = int(args.num_samples_train / (len(args.labels)/2))

        for i, label in enumerate(args.labels[:int(len(args.labels)/2)]):
            indices_worker_0[i * num_samples_per_class: (i + 1) * num_samples_per_class] =\
                np.random.choice(misc.find_indices_for_labels(args.dataset.root.train, [label]), [num_samples_per_class], replace=True)
        for i, label in enumerate(args.labels[int(len(args.labels)/2):]):
            indices_worker_1[i * num_samples_per_class: (i + 1) * num_samples_per_class] =\
                np.random.choice(misc.find_indices_for_labels(args.dataset.root.train, [label]), [num_samples_per_class], replace=True)

        random.shuffle(indices_worker_0)
        random.shuffle(indices_worker_1)

        # Send samples to the workers
        indices_local = torch.zeros([args.num_samples_train], dtype=torch.int)
        indices = [indices_local, torch.IntTensor(indices_worker_0), torch.IntTensor(indices_worker_1)]
        dist.scatter(tensor=indices_local, src=0, scatter_list=indices, group=nodes)

        # Save samples sent to the workers at master to evaluate train loss and accuracy later
        indices_local = torch.IntTensor(np.hstack((indices_worker_0, indices_worker_1)))

    else:
        args.local_labels = args.labels[int(len(args.labels)/2) * (rank - 1): int(len(args.labels)/2) * rank]
        indices_local = torch.zeros([args.num_samples_train], dtype=torch.int)
        dist.scatter(tensor=indices_local, src=0, scatter_list=[], group=nodes)

    return indices_local


def global_update(nodes, rank, network, weights_list):
    """"
    Global update step for distributed learning.
    """

    for j, parameter in enumerate(network.get_parameters()):
        if rank != 0:
            dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=[], dst=0, group=nodes)
        else:
            dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=weights_list[j], dst=0, group=nodes)
            network.get_parameters()[parameter].data = torch.mean(torch.stack(weights_list[j][1:]), dim=0)
        dist.broadcast(network.get_parameters()[parameter], 0, group=nodes)


def global_update_subset(nodes, rank, network, weights_list, gradients_accum, n_weights_to_send):
    """"
    Global update step for distributed learning when transmitting only a subset of the weights.
    Each worker node transmits a tensor in which only the indices corresponding to the synapses with the largest n_weights_to_send accumulated gradients are kept nonzero
    """

    for j, parameter in enumerate(network.get_parameters()):
        if j == 0:
            if rank != 0:
                to_send = network.get_parameters()[parameter].data  # each worker node copies its weights in a new vector
                # Selection of the indices to set to zero before transmission
                indices_not_to_send = [i for i in range(network.n_basis_feedforward) if i not in torch.topk(torch.sum(gradients_accum, dim=(0, 1)), n_weights_to_send)[1]]
                to_send[:, :, indices_not_to_send] = 0

                # Transmission of the quantized weights
                dist.gather(tensor=to_send, gather_list=[], dst=0, group=nodes)
            else:
                dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=weights_list[j], dst=0, group=nodes)

                indices_received = torch.bincount(torch.nonzero(torch.sum(torch.stack(weights_list[j][1:]), dim=(1, 2)))[:, 1])
                multiples = torch.zeros(network.n_basis_feedforward)  # indices of weights transmitted by two devices at once: those will be averaged
                multiples[:len(indices_received)] = indices_received
                multiples[multiples == 0] = 1

                # Averaging step
                network.get_parameters()[parameter].data = torch.sum(torch.stack(weights_list[j][1:]), dim=0) / multiples.type(torch.float)

        else:
            if rank != 0:
                dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=[], dst=0, group=nodes)
            else:
                dist.gather(tensor=network.get_parameters()[parameter].data, gather_list=weights_list[j], dst=0, group=nodes)
                network.get_parameters()[parameter].data = torch.mean(torch.stack(weights_list[j][1:]), dim=0)
        dist.broadcast(network.get_parameters()[parameter], 0, group=nodes)

