import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os

epoch = 3
lr = 1e-3
MODEL_PATH = None
SAVE_PATH = None

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
    pass


def evaluate():
    pass




if __name__ == "__main__":
    pass
