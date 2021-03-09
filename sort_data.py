import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os


SPACEGROUP_TAG = {
        0: (1, 2),
        1: (3, 15),
        2: (16, 74),
        3: (75, 142),
        4: (143, 167),
        5: (168, 194),
        6: (195, 230)
    }

SPACEGROUP_NAME = {
    0: "Triclinic",
    1: "Monoclinic",
    2: "Orthorhombic",
    3: "Tetragonal",
    4: "Trigonal",
    5: "Hexagonal",
    6: "Cubic"
}


def Totxt(crystal_system):

    lower_bound = SPACEGROUP_TAG[crystal_system][0]
    upper_bound = SPACEGROUP_TAG[crystal_system][1]
    sorted = []

    folder = os.path.join(os.getcwd(), "data4.0")
    n = 0
    for path in os.listdir(folder):
        n += 1
        print(n)
        data_path = os.path.join(folder, path)
        with open(data_path) as f:
            data = json.load(f)
            if lower_bound <= data["number"] <= upper_bound:
                sorted.append(path)

        with open(SPACEGROUP_NAME[crystal_system] + ".txt", "w+") as f:
            f.writelines(i + "\n" for i in sorted)


if __name__ == "__main__":
    pass
