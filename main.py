# -*- coding: utf-8 -*-


import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import net_builder, get_logger, count_parameters, over_write_args_from_file
from train_utils import get_optimizer, get_cosine_schedule_with_warmup
from models.main.main import S2_VER
from datasets.ssl_dataset import ImageNetLoader, Emotion_SSL_Dataset
from datasets.data_utils import get_data_loader
import json


# -------------------------------
# Utilities
# -------------------------------

def setup_seed(seed: int, deterministic: bool = True):
    assert seed is not None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN flags
    cudnn.deterministic = bool(deterministic)
    cudnn.benchmark = not deterministic


def get_warmup_steps(total_steps: int, warmup_ratio: float, warmup_steps=None) -> int:
    if warmup_steps is not None and warmup_steps > 0:
        return int(warmup_steps)
    ratio = max(0.0, min(1.0, float(warmup_ratio)))
    return int(total_steps * ratio)


# -------------------------------
# Main entry points
# -------------------------------

def main(args):
    if getattr(args, 'c', None):
        args = over_write_args_from_file(args, args.c)

    # Save path checks
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite and not args.resume:
        raise Exception(f'already existing model: {save_path}. Use --overwrite to overwrite.')

    os.makedirs(save_path, exist_ok=True)

    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading paths are the same. Use --overwrite to allow it.')

    if args.seed is not None and args.deterministic:
        warnings.warn('Deterministic mode is ON. This may slow down training and change cuDNN behavior.')

    if args.gpu is not None:
        warnings.warn('Using a specific GPU id disables DataParallel.')

    # Save current args for reproducibility
    try:
        with open(os.path.join(save_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to dump args.json: {e}")

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    setup_seed(args.seed, deterministic=args.deterministic)

    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    try:
        cudnn.allow_tf32 = bool(args.tf32)
    except Exception:
        pass

    # -----------------
    # Logger
    # -----------------
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name), "INFO")
    logger.warning(f"USE GPU: {args.gpu} for training")
    logger.info(f"Task = {args.dataset}@{args.num_labels}")

    # -----------------
    # Build model
    # -----------------
    args.bn_momentum = 1.0 - 0.999
    _net_builder = net_builder(args.net, False, None, is_remix=False, dim=args.low_dim, proj=True)

    tb_log = None  # Keep as None unless you wire a TB writer

    model = S2_VER(
        _net_builder,
        args.num_classes,
        args.ema_m,
        args.T,
        args.p_cutoff,
        args.ulb_loss_ratio,
        args.hard_label,
        tb_log=tb_log,
        args=args,
        logger=logger,
    )

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # -----------------
    # Optimizer & LR schedule
    # -----------------
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)

    total_steps = int(args.num_train_iter * args.epoch)
    warmup_steps = get_warmup_steps(total_steps, args.warmup_ratio, args.warmup_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        total_steps,
        num_warmup_steps=warmup_steps,
    )

    model.set_optimizer(optimizer, scheduler)

    # -----------------
    # Device setup
    # -----------------
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')

    if args.gpu is not None:
        ndev = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= ndev:
            raise ValueError(f'Invalid GPU id {args.gpu}; available: 0..{ndev-1}')
        torch.cuda.set_device(args.gpu)
        model.model = model.model.cuda(args.gpu)
    else:
        # Fallback to DataParallel over all visible devices
        model.model = torch.nn.DataParallel(model.model).cuda()

    if args.torch_compile and hasattr(torch, 'compile'):
        try:
            compile_mode = 'max-autotune' if hasattr(torch._dynamo, 'config') else 'default'
            model.model = torch.compile(model.model, mode=compile_mode, fullgraph=False)
            logger.info(f"torch.compile enabled (mode={compile_mode}).")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without it. Reason: {e}")

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {vars(args)}")

    cudnn.benchmark = not args.deterministic

    # -----------------
    # Datasets & Loaders
    # -----------------
    if args.dataset != "imagenet":
        train_dset = Emotion_SSL_Dataset(
            args, alg='comatch', name=args.dataset, train=True,
            num_classes=args.num_classes, data_dir=args.train_data_dir
        )
        lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)

        _eval_dset = Emotion_SSL_Dataset(
            args, alg='comatch', name=args.dataset, train=False,
            num_classes=args.num_classes, data_dir=args.test_data_dir
        )
        eval_dset = _eval_dset.get_dset()
    else:
        image_loader = ImageNetLoader(
            root_path=args.data_dir, num_labels=args.num_labels, num_class=args.num_classes
        )
        lb_dset = image_loader.get_lb_train_data()
        ulb_dset = image_loader.get_ulb_train_data()
        eval_dset = image_loader.get_lb_test_data()

    logger.info(f"Dataset sizes -> LB: {len(lb_dset)} | ULB: {len(ulb_dset)} | EVAL: {len(eval_dset)}")

    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    loader_dict['train_lb'] = get_data_loader(
        dset_dict['train_lb'],
        args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
    )

    loader_dict['train_ulb'] = get_data_loader(
        dset_dict['train_ulb'],
        args.batch_size * args.uratio,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
    )

    loader_dict['eval'] = get_data_loader(
        dset_dict['eval'],
        args.eval_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    logger.info(
        f"Loader iters/epoch -> LB: {len(loader_dict['train_lb'])} | ULB: {len(loader_dict['train_ulb'])} | EVAL: {len(loader_dict['eval'])}"
    )

    model.set_data_loader(loader_dict)
    model.set_dset(ulb_dset)

    # -----------------
    # Resume
    # -----------------
    if args.resume:
        model.load_model(args.load_path)
        # scheduler state lives inside model.scheduler now
        try:
            last_ep = getattr(model.scheduler, 'last_epoch', 'NA')
            logger.info(f"Scheduler state restored. last_epoch={last_ep}")
        except Exception:
            pass
        logger.info(f"Resumed from {args.load_path}")

    # -----------------
    # Train loop with best checkpointing & early stopping
    # -----------------
    trainer = model.train
    best_eval_acc = -float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epoch):
        eval_acc = trainer(args, epoch, best_eval_acc, logger=logger)

        if eval_acc > best_eval_acc + 1e-9:
            best_eval_acc = eval_acc
            epochs_no_improve = 0
            if args.save_best:
                best_ckpt = {
                    'model': model.model.state_dict(),
                    'args': vars(args),
                    'epoch': epoch,
                    'best_eval_acc': best_eval_acc,
                }
                torch.save(best_ckpt, os.path.join(args.save_dir, args.save_name, 'best.pth'))
                logger.info(f"New best acc {best_eval_acc:.4f} @ epoch {epoch}. Saved best.pth")
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            logger.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break

    logger.info("Training finished.")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse as _argparse
        raise _argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Improved main for S2_VER/CoMatch')

    # Saving & loading
    parser.add_argument('--save_dir', type=str, default='eccv_result/single/sgd/add')
    parser.add_argument('-sn', '--save_name', type=str, default='main')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--save_best', type=str2bool, default=True)

    # Training configuration
    parser.add_argument('--epoch', type=int, default=1500)
    parser.add_argument('--num_train_iter', type=int, default=1024, help='iterations per epoch')
    parser.add_argument('-nl', '--num_labels', type=int, default=3)
    parser.add_argument('-bsz', '--batch_size', type=int, default=2)
    parser.add_argument('--uratio', type=int, default=4, help='ratio of unlabeled to labeled in a batch')
    parser.add_argument('--eval_batch_size', type=int, default=128)

    # CoMatch / S2_VER params
    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.7)          # tuned default
    parser.add_argument('--p_cutoff', type=float, default=0.90)  # tuned default
    parser.add_argument('--p_cutoff_end', type=float, default=None,
                        help='target cutoff for ramp-up (None keeps constant)')
    parser.add_argument('--p_rampup_ratio', type=float, default=0.2,
                        help='fraction of total iters to reach p_cutoff_end')
    parser.add_argument('--noise_th', type=float, default=0.3)
    parser.add_argument('--ema_m', type=float, default=0.999)
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--ldl_ratio', type=float, default=0.2)
    parser.add_argument('--low_dim', type=int, default=2816)
    parser.add_argument('--lam_c', type=float, default=3)
    parser.add_argument('--lam_d', type=float, default=3)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--dynamic_th', type=float, default=0.7)
    parser.add_argument('--dis_ce', action='store_true')
    parser.add_argument('--update_m', type=str, default='L2')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--label_filter_min', type=float, default=0.8,
                        help='minimum CE threshold when filtering early epochs')
    parser.add_argument('--gate_clean_start', type=int, default=5,
                        help='epoch to start early gate filtering (<=10 applies)')
    parser.add_argument('--add_ulb', type=str2bool, default=False)

    # Optimizer
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # LR warmup controls
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='fraction of total steps for warmup')
    parser.add_argument('--warmup_steps', type=int, default=0, help='override warmup steps if > 0')
    parser.add_argument('--clip', type=float, default=0.0)

    # AMP / compile / TF32 / determinism
    parser.add_argument('--amp', type=str2bool, default=False, help='(handled in model/train loop if supported)')
    parser.add_argument('--torch_compile', type=str2bool, default=False, help='enable torch.compile if available')
    parser.add_argument('--tf32', type=str2bool, default=True, help='allow TF32 on Ampere+ GPUs')
    parser.add_argument('--deterministic', type=str2bool, default=False)  # faster by default

    # Backbone
    parser.add_argument('--net', type=str, default='ResNet50')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Data
    parser.add_argument('--data_dir', type=str, default='datasets/MVSA_Single')
    parser.add_argument('--train_data_dir', type=str, default='datasets/MVSA_Single')
    parser.add_argument('--test_data_dir', type=str, default='datasets/MVSA_Single')
    parser.add_argument('-ds', '--dataset', type=str, default='mvsa-s')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)

    # Early stopping
    parser.add_argument('--patience', type=int, default=0, help='stop if no improvement for this many epochs (0=off)')
    parser.add_argument('--eval_every', type=int, default=1, help='evaluate every N epochs (currently unused)')

    # GPU & seed
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    # Config file
    parser.add_argument('--c', type=str, default='')
    parser.add_argument('--ulb_rampup_ratio', type=float, default=0.1,
                        help='fraction of total iters for unlabeled loss ramp-up')
    parser.add_argument('--split_path', type=str, default=None,
                        help='Path to the json file containing fixed data split for reproducibility')
    parser.add_argument('--enable_motivation_log', type=str2bool, default=False,
                        help='record pseudo-label stats for motivation scatter plots')
    parser.add_argument('--motivation_samples_per_batch', type=int, default=32,
                        help='max samples recorded per unlabeled batch when logging motivation stats')
    parser.add_argument('--motivation_flush_size', type=int, default=2048,
                        help='number of records cached before flushing to CSV')
    parser.add_argument('--enable_cpl', type=str2bool, default=True,
                        help='enable conflict-aware pseudo-label filtering')
    parser.add_argument('--enable_mco', type=str2bool, default=True,
                        help='enable multi-branch consistency weighting')
    parser.add_argument('--enable_drr', type=str2bool, default=True,
                        help='enable decoupled representation regularization')

    args = parser.parse_args()
    main(args)
