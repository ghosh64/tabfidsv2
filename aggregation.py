import torch
import numpy as np
import itertools
from tqdm import tqdm


def fed_avg(state_dicts, n_devices, dataset_lengths, device):
    updated_dict = state_dicts[0].copy()
    for layer in state_dicts[0].keys():
        averaged_params = torch.zeros(state_dicts[0][layer].size(), device=device)
        for dev in range(n_devices):
            averaged_params += dataset_lengths[dev] * state_dicts[dev][layer]
        updated_dict[layer] = averaged_params / dataset_lengths.sum()
    return updated_dict


def per_layer_aggregation(device_model, val_data_loader, criterion, statedicts,
                          dev, comm_round, device, temporal_avg_fn=None,
                          use_temporal_avg=False, temporal_window=3):
    # Original code was hardcoded for CLDNN (12 layers): 6 groups repeated to 12.
    # If using CLDNN, you can replace the below with:
    #   pl_weights = list(itertools.product(*[np.arange(0, 1.01, 1)]*6))
    #   pl_weights = [np.repeat(pl_wt, 2) for pl_wt in pl_weights]
    n_layers = len(list(device_model.state_dict().keys()))
    n_groups = min(6, n_layers)
    repeat_factor = max(1, n_layers // n_groups)

    pl_weights = list(itertools.product(*[np.arange(start=0, stop=1.01, step=1)] * n_groups))
    pl_weights = [np.repeat(pl_wt, repeat_factor)[:n_layers] for pl_wt in pl_weights]

    trained_global_model = device_model.state_dict().copy()
    target_local_model = device_model.state_dict().copy()
    last_local_model = statedicts[-1][dev].copy()

    tloss_previous = float('inf')
    best_wt = None
    best_model = None

    for wt in tqdm(pl_weights, desc=f"PL search device {dev}"):
        for i, layer in enumerate(trained_global_model.keys()):
            target_local_model[layer] = (
                wt[i] * trained_global_model[layer] +
                (1 - wt[i]) * last_local_model[layer]
            )

        device_model.load_state_dict(target_local_model)
        testing_loss = _evaluate(
            device_model, val_data_loader, criterion, device,
            temporal_avg_fn, use_temporal_avg, temporal_window
        )

        if testing_loss < tloss_previous:
            tloss_previous = testing_loss
            best_wt = wt
            best_model = {k: v.clone() for k, v in target_local_model.items()}

    device_model.load_state_dict(best_model)
    return best_wt


def _evaluate(model, data_loader, criterion, device, temporal_avg_fn,
              use_temporal_avg, temporal_window):
    model.eval()
    closs = 0
    with torch.no_grad():
        for data, labels in data_loader:
            if use_temporal_avg and temporal_avg_fn is not None:
                data, labels = temporal_avg_fn(data, labels, temporal_window)
            data = data.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            predictions = model(data)
            loss = criterion(predictions, labels)
            closs += loss.item()
    return closs / len(data_loader)
