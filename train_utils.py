import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import math
import time
import os
from copy import deepcopy
from torch.optim.optimizer import Optimizer, required
import copy

# from custom_writer import CustomWriter

class SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                d_p.mul_(group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-1)

        return loss


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name, writer_type="custom"):
        self.tb_dir = tb_dir
        self.writer_type = writer_type
        if self.writer_type == "tensorboard":
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
        # elif self.writer_type == "custom":
        #     self.writer = CustomWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None, mode="train"):

        if suffix is None:
            suffix = ''
        if self.writer_type == "tensorboard":
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value, it)
        elif self.writer_type == "custom":
            self.writer.set_epoch(it, mode)
            for key, value in tb_dict.items():
                self.writer.add_scalar(suffix + key, value)
            if it < 1000000:
                return
            self.writer.plot_stats()
            self.writer.dump_stats()


class AverageMeter(object):


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


def wd_loss(net):
    loss = 0
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name):
            continue
        elif ('weight' in name):
            loss = loss + torch.sum(param ** 2) / 2
    return loss


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, nesterov=True, bn_wd_skip=True):

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name or 'bias' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):

    def _lr_lambda(current_step):

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_imagenet_schedule(optimizer, num_training_steps, num_labels, batch_size):
    def iter2epoch(iter):
        iter_per_ep = num_labels // batch_size
        ep = iter // iter_per_ep
        return ep

    def epoch2iter(epoch):
        iter_per_ep = num_labels // batch_size
        iter = epoch * iter_per_ep
        return iter

    def _lr_lambda(iter):
        return None


def accuracy(output, target, topk=(1,)):


    with torch.no_grad():
        maxk = max(topk)  # get k in top-k
        batch_size = target.size(0)  # get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)  # pred: [num of batch, k]
        pred = pred.t()  # pred: [k, num of batch]

        # [1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # np.shape(res): [k, 1]
        return res


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):

    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """Multi-class focal loss with per-class alpha weights."""
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            alpha = [1.0]
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha_tensor)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha
        num_classes = pred.size(1)
        if alpha.numel() == 1:
            alpha = alpha.expand(num_classes)
        elif alpha.numel() != num_classes:
            raise ValueError(f"alpha expects {num_classes} entries, got {alpha.numel()}")
        alpha = alpha.to(pred.device)
        alpha = alpha[target]

        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1)).view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=None, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = float(smoothing)
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.ndim < 2:
            raise ValueError('pred tensor must have shape [B, num_classes]')
        n_classes = pred.size(1) if self.num_classes is None else self.num_classes
        if n_classes <= 1:
            raise ValueError('Label smoothing requires num_classes > 1')

        smoothing = min(max(self.smoothing, 0.0), 0.499)
        log_probs = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)

        loss = (-true_dist * log_probs).sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class EMA:

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Bn_Controller:
    def __init__(self):

        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}
