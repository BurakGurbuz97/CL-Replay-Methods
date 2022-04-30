# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from status import progress_bar, create_stash
from loggers import *
from loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
import copy


# Calculate the sparsity of the model
def compute_weight_sparsity(model):
    parameters = 0
    ones = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            shape = module.weight.data.shape
            parameters += torch.prod(torch.tensor(shape))
            w_mask, _ = copy.deepcopy(module.get_mask())
            ones += torch.count_nonzero(w_mask)
    return float((parameters - ones) / parameters) * 100


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    if not hasattr(dataset, "T"):
        outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
        outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')




def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)


    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)


    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        print('Model sparsity before training:', compute_weight_sparsity(model.net))
        train_loader, _ = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)

        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                if hasattr(dataset.train_loader.dataset, 'logits'):
                     #We do not perform any augmentation of seq-efmnist
                    if args.dataset == "seq-efmnist":
                        inputs, labels, _ = data
                        not_aug_inputs = copy.deepcopy(inputs)
                    else:
                        inputs, labels, not_aug_inputs, _ = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    #logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits = None)
                else:
                    #We do not perform any augmentation of seq-efmnist
                    if args.dataset == "seq-efmnist":
                        inputs, labels, _ = data
                        not_aug_inputs = copy.deepcopy(inputs)
                    else:
                        inputs, labels, not_aug_inputs = data
                        
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(i, len(train_loader), epoch, t, loss)
            
            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if args.csv_log:
            csv_logger.log(mean_acc)


    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.csv_log:
        csv_logger.write(vars(args))
