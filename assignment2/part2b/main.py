import torch
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import model as mdl
import time
import os
import argparse

device = "cpu"
torch.set_num_threads(4)

batch_size = 64 # batch for one node

NUM_EPOCHS = 1

SEED = 14
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    running_loss = 0.0

    group = dist.new_group([0, 1, 2, 3])

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, target)
        loss.backward()

        # sync gradient with allreduce
        for param in model.parameters():
            param.grad /= 4
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group, async_op=False)

        optimizer.step()

        running_loss += loss.item()
        
        # FIXME: running loss or loss.item()
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - Loss: {running_loss / (batch_idx + 1):.4f}")
        
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
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", type=str, required=True, help="IP address of the master node")
    parser.add_argument("--num-nodes", type=int, required=True, help="Total number of nodes")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the current node")
    args = parser.parse_args()

    # FIXME:ip, port?
    init_method = f"tcp://{args.master_ip}:6585"
    dist.init_process_group(backend='gloo', init_method=init_method, rank=args.rank, world_size=args.num_nodes)

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
    # use the distributed sampler to distribute the data among workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=True,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} complete time: {epoch_time:.4f} sec")

if __name__ == "__main__":
    main()

