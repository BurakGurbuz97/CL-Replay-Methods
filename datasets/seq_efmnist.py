# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from continuum.datasets import  FashionMNIST, EMNIST, Fellowship
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MLPCustom import MLP
import torch.nn.functional as F
from conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from continuum import ClassIncremental
from typing import Tuple

def store_masked_loaders_custom(train_dataset, test_dataset, setting):
    scenario_train = ClassIncremental(train_dataset, increment=setting.T)
    scenario_test = ClassIncremental(test_dataset, increment=setting.T)

    train_loader = DataLoader(scenario_train[setting.i],
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(scenario_test[setting.i],
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=1)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i = setting.i + 1
    return train_loader, test_loader


class SequentialEFMNIST(ContinualDataset):
    T = [10, 13, 13, 11, 10]
    NAME = 'seq-efmnist'
    SETTING = 'class-il'
    N_TASKS = 5
    TRANSFORM = None

    def get_data_loaders(self):
        train_dataset = Fellowship([EMNIST(base_path() + 'EMNIST', train = True, download=True, split='balanced'), 
                                    FashionMNIST(base_path() + 'FMNIST', train = True, download=True)])
       
        test_dataset = Fellowship([EMNIST(base_path() + 'EMNIST', train = False, download=True, split='balanced'), 
                                    FashionMNIST(base_path() + 'FMNIST', train = False, download=True)])

        train, test = store_masked_loaders_custom(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return MLP(28 * 28, 57)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None