import numpy as np
import os

import matplotlib.pyplot as plt

def noise_mask(X, masking_ratio, lm=3):
    mask = np.ones(X.shape, dtype=bool)
    for m in range(X.shape[0]):  # 
        mask[m, :] = geom_noise_mask_single(X.shape[1], lm, masking_ratio)  # time dimension
    
    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def positive_negative_sampling(x, y, all_data, num_samples=8):
    positive_samples = []
    negative_samples = []
    normal_samples = all_data[all_data[:, -1] == 0]
    abnormal_samples = all_data[all_data[:, -1] == 1]
    for i in range(len(y)):
        if y[i] == 0:
            positive_samples_index = np.random.choice(normal_samples.shape[0], num_samples, replace=False)
            positive_samples.append(normal_samples[positive_samples_index, :-1])
            negative_samples_index = np.random.choice(abnormal_samples.shape[0], num_samples, replace=False)
            negative_samples.append(abnormal_samples[negative_samples_index, :-1])
        else:
            positive_samples_index = np.random.choice(abnormal_samples.shape[0], num_samples, replace=False)
            positive_samples.append(abnormal_samples[positive_samples_index, :-1])
            negative_samples_index = np.random.choice(normal_samples.shape[0], num_samples, replace=False)
            negative_samples.append(normal_samples[negative_samples_index, :-1])

    positive_samples = np.array(positive_samples)
    negative_samples = np.array(negative_samples)
    return positive_samples, negative_samples

def performance_display(metric_value, metric_name, output_path):
    color = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    for index, (name, values) in enumerate(metric_value.items()):
        plt.plot(list(range(1, len(values)+1)), 
                    values, 
                    color=color[index], 
                    linewidth=1.5, 
                    label=name)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel(metric_name)
    plt.grid(linestyle='--')
    fig_path = os.path.join(output_path, metric_name+'.png')
    plt.savefig(fig_path, dpi=500, bbox_inches = 'tight')
    plt.cla()