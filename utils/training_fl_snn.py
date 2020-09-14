import torch
import tables


def feedforward_sampling(network, example, ls, et, args, gradients_accum=None):
    log_proba = network(example)

    # Accumulate learning signal
    proba_hidden = torch.sigmoid(network.potential[network.hidden_neurons - network.n_input_neurons])
    ls += torch.sum(log_proba[network.output_neurons - network.n_input_neurons]) \
          - args.alpha * torch.sum(network.spiking_history[network.hidden_neurons, -1]
          * torch.log(1e-12 + proba_hidden / args.r)
          + (1 - network.spiking_history[network.hidden_neurons, -1]) * torch.log(1e-12 + (1. - proba_hidden) / (1 - args.r)))

    for parameter in network.get_gradients():
        # Only when the comm. rate is fixed
        if (parameter == 'ff_weights') & (gradients_accum is not None):
            gradients_accum += torch.abs(network.get_gradients()[parameter])

        et[parameter] += network.get_gradients()[parameter]

    return log_proba, ls, et, gradients_accum


def local_feedback_and_update(network, eligibility_trace, et_temp, learning_signal, ls_temp, s, args):
    """"
    Runs the local feedback and update steps:
    - computes the learning signal
    - updates the learning parameter
    """
    # At local algorithmic timesteps, do a local update
    if (s + 1) % args.deltas == 0:
        # local feedback
        learning_signal = args.kappa * learning_signal + (1 - args.kappa) * ls_temp
        ls_temp = 0

        # Update parameter
        for parameter in network.get_gradients():
            eligibility_trace[parameter].mul_(args.kappa).add_(1 - args.kappa, et_temp[parameter])
            et_temp[parameter] = 0

            network.get_parameters()[parameter] += args.lr * eligibility_trace[parameter]

    return eligibility_trace, et_temp, learning_signal, ls_temp


def init_training(network, indices, args):
    eligibility_trace_hidden = {parameter: network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.get_gradients()}
    eligibility_trace_output = {parameter: network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.get_gradients()}

    et_temp_hidden = {parameter: network.get_gradients()[parameter][network.hidden_neurons - network.n_input_neurons] for parameter in network.get_gradients()}
    et_temp_output = {parameter: network.get_gradients()[parameter][network.output_neurons - network.n_input_neurons] for parameter in network.get_gradients()}

    learning_signal = 0
    ls_temp = 0

    baseline_num = {parameter: eligibility_trace_hidden[parameter].pow(2) * learning_signal for parameter in eligibility_trace_hidden}
    baseline_den = {parameter: eligibility_trace_hidden[parameter].pow(2) for parameter in eligibility_trace_hidden}

    S_prime = tables.open_file(args.dataset).root.train.label[:].shape[-1]
    S = len(indices[args.start_idx:]) * S_prime

    return eligibility_trace_hidden, eligibility_trace_output, et_temp_hidden, et_temp_output, learning_signal, ls_temp, baseline_num, baseline_den, S_prime, S
