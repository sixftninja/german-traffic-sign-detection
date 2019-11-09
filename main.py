from __future__ import print_function
from collections import namedtuple
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

from data import data_transforms, train_data_transforms, ImbalancedDatasetSampler
from model import Net1, Net2, Net3, Net4, Net5

parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_workers', type=int, default=4, metavar='W',
                    help='number of workers (default: 4)')
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='O',
                    help='name of output file, default is gtsrb_kaggle.csv')
parser.add_argument('--checkpoint', type=str, default='', metavar='C',
                    help='checkpoint, default:empty')
args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    gpu = True
    print("Using GPU")
else:
    gpu = False
    print("using CPU")

FloatTensor = torch.cuda.FloatTensor if gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if gpu else torch.ByteTensor
Tensor = FloatTensor

model = Net4()
if gpu: model.cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

train_dataset = datasets.ImageFolder(args.data + '/train_images', transform=train_data_transforms)
val_dataset = datasets.ImageFolder(args.data + '/val_images', transform=data_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    #shuffle=True,
    sampler=ImbalancedDatasetSampler(train_dataset),
    num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

'''
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        # from visdom import Visdom
        # self.viz = Visdom(server='log-0')
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        return
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(
                X=np.array([x]), Y=np.array([y]), env=self.env,
                win=self.plots[var_name], name=split_name, update = 'append')
'''

def calc_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#plotter = VisdomLinePlotter()


def train(epoch):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        if gpu:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss = F.nll_loss(output, target)

        losses.update(loss.item(), data.size(0))
        accuracy = calc_accuracy(output, target)[0]
        accuracies.update(accuracy, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time = epoch - 1 + (1 + batch_idx) / len(train_loader)
        # plotter.plot('loss', 'train', 'Class Loss', time, losses.avg)
        # plotter.plot('acc', 'train', 'Class Accuracy', time, accuracies.avg)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), losses.val))

    #plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)
    #plotter.plot('acc', 'train', 'Class Accuracy', epoch, accuracies.avg)


def validation(epoch):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    error_cases = []
    with torch.no_grad():
        for data, target in val_loader:
            if gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            loss = F.nll_loss(output, target)

            losses.update(loss.item(), data.size(0))
            accuracy = calc_accuracy(output, target)[0]
            accuracies.update(accuracy, data.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            error_cases.append(data[(pred != target.data.view_as(pred)).nonzero(as_tuple=True)[0]])

        images = torch.cat(error_cases)
        grid = torchvision.utils.make_grid(images, normalize=True)
        to_pil_image = transforms.ToPILImage()
        plt.imshow(to_pil_image(grid.cpu()))
        # plotter.viz.image(grid.cpu(), caption="Error Cases – Epoch {}".format(epoch))

        # plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
        # plotter.plot('acc', 'val', 'Class Accuracy', epoch, accuracies.avg)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            losses.avg, accuracies.sum/100, accuracies.count, accuracies.avg))

        return losses.avg, accuracies.avg

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation_loss, accuracy = validation(epoch)
    scheduler.step(round(validation_loss, 3))
    model_file = 'checkpoints/{}_epoch{}_val{:.6f}_acc{:.3f}.pth'.format(
        model.__class__.__name__, epoch, validation_loss, accuracy)
    torch.save(model.state_dict(), model_file)
