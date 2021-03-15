import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import json
import os
from random import shuffle
import math


PATH = os.path.join(os.getcwd(), "data4.0")

SPACEGROUP_FILE = {
    0: "Triclinic.txt",
    1: "Monoclinic.txt",
    2: "Orthorhombic.txt",
    3: "Tetragonal.txt",
    4: "Trigonal.txt",
    5: "Hexagonal.txt",
    6: "Cubic.txt"
}

SPACEGROUP_LABEL_OFFSET = {
        0: 1,
        1: 3,
        2: 16,
        3: 75,
        4: 143,
        5: 168,
        6: 195
}

SPACEGROUP_SHAPE = {
        0: 2,
        1: 13,
        2: 59,
        3: 68,
        4: 25,
        5: 27,
        6: 36
}


class Cs2Sg(Dataset):
    def __init__(self, crystal_system, valid_size):
        print("Preparing dataset")
        data_input_valid = []
        data_label_valid = []
        data_input_train = []
        data_label_train = []

        with open(SPACEGROUP_FILE[crystal_system], "r") as f:
            path_list = f.readlines()
        shuffle(path_list)
        valid_len = math.floor(len(path_list) * valid_size)

        for i, path in enumerate(path_list):
            data_path = os.path.join(PATH, path.rstrip())
            with open(data_path, "r") as f:
                data = json.load(f)
                data_input = np.array(data["bands"]).T
                data_label = np.array([data["number"]]) - SPACEGROUP_LABEL_OFFSET[crystal_system]

                if i < valid_len:
                    data_input_valid.append(torch.from_numpy(data_input).float())
                    data_label_valid.append(torch.from_numpy(data_label).long())

                else:
                    data_input_train.append(torch.from_numpy(data_input).float())
                    data_label_train.append(torch.from_numpy(data_label).long())

        print("valid length:", len(data_input_valid))
        print("train length:", len(data_input_train))

        self.data_inputs = data_input_valid + data_input_train
        self.data_labels = data_label_valid + data_label_train
        self.length = len(self.data_labels)
        self.valid_size = valid_len

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data_inputs[item], self.data_labels[item]


def get_valid_train_loader(dataset, batch_size):
    num = len(dataset)
    indices = [i for i in range(num)]
    split = dataset.valid_size

    valid_idx, train_idx = indices[:split], indices[split:]

    valid_sampler = SubsetRandomSampler(valid_idx)
    train_sampler = SubsetRandomSampler(train_idx)

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    return train_loader, valid_loader

