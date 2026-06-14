import torch
import numpy as np


def temporal_avg(data, labels, n=3):
    data, labels = data.numpy(), labels.numpy()
    ind_attack = [i for i in range(len(labels)) if labels[i] == 0]
    ind_benign = [i for i in range(len(labels)) if labels[i] == 1]

    data_att = [np.array(data[i]) for i in ind_attack]
    data_ben = [np.array(data[i]) for i in ind_benign]

    label_att = [labels[i] for i in ind_attack]
    label_ben = [labels[i] for i in ind_benign]

    attack, labels_attack = _get_averaged_data(data_att, label_att, n)
    benign, labels_benign = _get_averaged_data(data_ben, label_ben, n)

    attack = attack + benign
    labels_attack = labels_attack + labels_benign

    return torch.FloatTensor(attack), torch.FloatTensor(labels_attack)


def _get_averaged_data(data, labels, n):
    n = n if len(labels) > n else len(labels)
    data_avg, labels_avg = [], []
    for i in range(len(data)):
        lst = [data[j] if j < len(data) else data[j - len(data)] for j in range(i, i + n)]
        data_avg.append(list(sum(lst) / n))
        labels_avg.append(labels[i])
    return data_avg, labels_avg
