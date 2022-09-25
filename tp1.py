import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split


def genData():
    # Generate dataset
    n_data=1000000
    dataset = np.zeros([n_data, 3])
    cpt = 0
    for u_iter in np.arange(-5,5,0.01):
        for v_iter in  np.arange(-5,5,0.01):
            dataset[cpt,0] = u_iter
            dataset[cpt,1] = v_iter
            dataset[cpt,2] = np.sin(u_iter-v_iter)
            cpt+=1
    # Split and shuffle dataset
    split_percent = [0.7, 0.15, 0.15]
    split_length = []
    for percent in split_percent:
        split_length.append(int(n_data*percent))
    train_data, val_data, test_data = random_split(dataset, split_length)
    return train_data, val_data, test_data


def main():
    train_data, val_data, test_data = genData()


if __name__ == "__main__":
    main()