import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import time


SEED = 14
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cpu"
torch.set_num_threads(4)
def ddp_setup(rank, world_size, init_method):
    #ip address of node 0    
    init_process_group(backend="gloo", rank=rank, init_method=init_method, world_size=world_size)



batch_size = 256 # batch for one node
def train_model(rank, model, train_loader, optimizer, criterion, world_size, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Your code goes here!
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()

        params = model.parameters()

        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad
                if rank == 0:
                    grad_list = [torch.zeros_like(grad) for _ in range(world_size)]
                else:
                    grad_list = None
                dist.gather(grad, gather_list=grad_list, dst=0)
                if rank == 0:
                    grad_avg = grad_avg = torch.stack(grad_list, dim=0).mean(dim=0)
                    scatter_list = [grad_avg.clone() for _ in range(world_size)]
                else:
                    scatter_list = None
                aggregated_grad = torch.zeros_like(grad)
                dist.scatter(aggregated_grad, scatter_list=scatter_list, src=0)
                param.grad = aggregated_grad
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Bacth Size: {data.shape[0]}, Loss: {running_loss / (batch_idx + 1):.4f}")
        
        if batch_idx == 0:
            start_time = time.time()
        if batch_idx == 39:
            end_time = time.time()
            avg_time = (end_time - start_time) / 39
            print(f"Average time per iteration after 40 iterations: {avg_time:.4f} sec")

    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main(rank, world_size, master_ip):


    ddp_setup(rank, world_size, master_ip)

    global_batch_size = 256
    local_batch_size = global_batch_size // world_size

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=local_batch_size,
                                                    sampler=DistributedSampler(training_set, num_replicas=world_size, rank=rank),
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=local_batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(rank, model, train_loader, optimizer, training_criterion, world_size, epoch)
        test_model(model, test_loader, training_criterion)
    destroy_process_group()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed training with PyTorch and Gloo")
    parser.add_argument("--master-ip", type=str, required=True, help="IP address of master node")
    parser.add_argument("--num-nodes", type=int, required=True, help="Total number of nodes")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this node")
    args = parser.parse_args()
    main(args.rank, args.num_nodes, args.master_ip)
