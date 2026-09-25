import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import math

def set_seed(seed=42):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CutMix:
    """CutMix augmentation"""
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch, labels):
        if random.random() > self.prob:
            return batch, labels, labels, 1.0
        
        batch_size = batch.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size).to(batch.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch.size(), lam)
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size(-1) * batch.size(-2)))
        return batch, labels, labels[rand_index], lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

class MixUp:
    """MixUp augmentation"""
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch, labels):
        if random.random() > self.prob:
            return batch, labels, labels, 1.0
        
        batch_size = batch.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size).to(batch.device)
        mixed_batch = lam * batch + (1 - lam) * batch[rand_index]
        return mixed_batch, labels, labels[rand_index], lam

def get_dataloaders(batch_size=128, num_workers=2):
    """Get CIFAR-10 dataloaders with augmentations."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    def __init__(self, num_classes=10, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class WarmupCosineScheduler:
    """Cosine scheduler with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr