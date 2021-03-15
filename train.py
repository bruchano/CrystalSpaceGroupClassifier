import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from CNN_model import *
from NN_model import *
import data_loader
import data_processing
import crystal
import network


"""
range		count	name
1 - 2		2		Triclinic
3 - 15		13		Monoclinic
16 - 74		59		Orthorhombic
75 - 142	68		Tetragonal
143 - 167	25		Trigonal
168 - 194	27		Hexagonal
195 - 230	36		Cubic
"""

SPACEGROUP_NAME = {
    0: "Triclinic",
    1: "Monoclinic",
    2: "Orthorhombic",
    3: "Tetragonal",
    4: "Trigonal",
    5: "Hexagonal",
    6: "Cubic"
}

CRYSTAL_SYSTEM = 3
NET = "NN"
N_CNN = 2
CHANNELS = 6
OUT_FEATURES = 68
N_FC = [1024, 512, 128]

epoch = 10
lr = 5e-2

MODEL_PATH = None

GPU = "GPU4"
VER = "_1"
SAVE_PATH = "models/" + SPACEGROUP_NAME[CRYSTAL_SYSTEM] + "_" + NET + "_ch_" + str(CHANNELS) + "_lr_" + str(lr) + "_" + GPU + VER + ".pt"
FIGURE_PATH = "figures/" + NET + "_" + GPU + VER + ".png"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-ep", "--ep", type=int)
parser.add_argument("-lr", "--lr", type=float)
parser.add_argument("-p", "--path", type=str)
parser.add_argument("-m", "--model", type=str)
args = parser.parse_args()

if args.ep:
    epoch = args.ep
if args.lr:
    lr = args.lr
if args.path:
    SAVE_PATH = args.path
if args.model:
    MODEL_PATH = args.model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if NET == "NN":
        model = SpaceGroupNN(OUT_FEATURES, N_FC).to(device)
    else:
        model = SpaceGroupCNN(N_CNN, CHANNELS, OUT_FEATURES, N_FC).to(device)

    if MODEL_PATH:
        model.load_state_dict(torch.load(MODEL_PATH))

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = data_loader.Cs2Sg(CRYSTAL_SYSTEM, 0.1)
    train_loader, valid_loader = data_loader.get_valid_train_loader(dataset, 32)

    network.validate_train_loop(device, model, optimizer, scheduler, criterion, valid_loader, train_loader, epoch, SAVE_PATH, FIGURE_PATH)




if __name__ == "__main__":
    train()
