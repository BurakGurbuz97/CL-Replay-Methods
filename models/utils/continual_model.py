# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import Adam, SGD
import torch
import torchvision
from argparse import Namespace
from conf import get_device
import numpy as np
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#No Bias pruning
def random_prune(model, pruning_perc, skip_first = True):
    if pruning_perc > 0.0:
        model = copy.deepcopy(model)
        pruning_perc = pruning_perc / 100.0
        weight_masks = []
        bias_masks = []
    first_conv = skip_first
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight_mask = torch.from_numpy(np.random.choice([0, 1],
                                            module.weight.shape,
                                            p =  [pruning_perc, 1 - pruning_perc]))
            weight_masks.append(weight_mask.to(DEVICE))
            #do not prune biases
            bias_mask = torch.from_numpy(np.random.choice([0, 1],
                                            module.bias.shape,
                                            p =  [0, 1]))
            bias_masks.append(bias_mask.to(DEVICE))
        #Channel wise pruning Conv Layer
        elif isinstance(module, nn.Conv2d):
           if first_conv:
               connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                   (module.weight.shape[0],  module.weight.shape[1]),
                                                    p =  [0, 1]))
               first_conv = False
           else:
               connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                   (module.weight.shape[0],  module.weight.shape[1]),
                                                    p =  [pruning_perc, 1 - pruning_perc]))
           filter_masks = []
           for conv_filter in range(module.weight.shape[0]):
              
               filter_mask = []
               for inp_channel  in range(module.weight.shape[1]):
                   if connectivity_mask[conv_filter, inp_channel] == 1:
                       filter_mask.append(np.ones((module.weight.shape[2], module.weight.shape[3])))
                   else:
                       filter_mask.append(np.zeros((module.weight.shape[2], module.weight.shape[3])))
               filter_masks.append(filter_mask)
               
           weight_masks.append(torch.from_numpy(np.array(filter_masks)).to(torch.float32).to(DEVICE))
           
           #do not prune biases
           bias_mask = torch.from_numpy(np.random.choice([0, 1],
                                            module.bias.shape,
                                            p =  [0, 1])).to(torch.float32).to(DEVICE)
           bias_masks.append(bias_mask)
    model.to(DEVICE)
    model.set_masks(weight_masks, bias_masks)
    return model

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.net = random_prune(self.net, args.prune_perc)
        if args.optim == "sgd":
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        else:
            self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
