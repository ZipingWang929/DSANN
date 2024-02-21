import torch
import torch.nn as nn

from torch.nn import functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b


def get_optimizer(model, args,initial_lr):
    initial_lr = args.lr0 if not args.schuse else initial_lr
    params = model.get_parameters1(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr0, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    if not args.schuse:
        return None
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model,eval_loader,mode='weight'):
    correct = 0
    total = 0
    for data in eval_loader:
        x = data[0].cuda().float()
        y = data[1].cuda().long()
        cls,dis=model(x,train=False)
        if mode=='weight':
            weight = torch.sum(torch.stack(dis, dim=0),dim=0)
            p=torch.stack([dis[i]/weight*cls[i] for i in range(len(dis))],dim=0)
            p=torch.sum(p, dim=0)
        else:
            cls = torch.stack(cls, dim=0)
            p=torch.mean(cls,dim=0)
        if p.size(1) == 1:
            correct += (p.gt(0).eq(y).float()).sum().item()
        else:
            correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)

    return correct / total