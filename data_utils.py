# Author: Ghada Sokar et al.
# This is the implementation for the Learning Invariant Representation for Continual Learning paper in AAAI workshop on Meta-Learning for Computer Vision

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
from scipy.misc import imsave
from scipy.misc import imresize

def get_train_loader(train_dataset,batch_size):
    train_loader = DataLoader(
    train_dataset,
    batch_size,
    num_workers=0,
    pin_memory=True, shuffle=True)
    return train_loader

def get_test_loader(test_dataset,test_batch_size):
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return full_dataset,test_dataset

def task_construction(task_labels):
    full_dataset,test_dataset = load_data()
    train_dataset = split_dataset_by_labels(full_dataset, task_labels)
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset,test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    task_idx = 0
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.targets = splited_dataset.targets[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
        task_idx += 1
    return datasets

