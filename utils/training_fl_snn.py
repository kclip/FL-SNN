import torch
import tables


def feedforward_sampling(network, inputs, outputs, ls, et, args, gradients_accum=None):
    log_proba = network(inputs, outputs)

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
