
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import numpy as np
import os

from datasets.DistributedProxySampler import DistributedProxySampler


def split_ssl_data(args, img, text, img_label, text_label, label,
                   num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    Separate labeled and unlabeled data
    """

    def _as_str_list(seq):
        out = []
        for t in seq:
            if isinstance(t, str):
                out.append(t)
            elif isinstance(t, (list, tuple)):
                out.append(" ".join(map(str, t)))
            elif t is None:
                out.append("")
            else:
                out.append(str(t))
        return out

    def _as_int_labels(seq):
        import numpy as _np
        res = []
        for x in seq:
            if isinstance(x, (int, _np.integer)):
                res.append(int(x))
            elif isinstance(x, str):
                res.append(int(x))
            elif isinstance(x, (list, tuple, _np.ndarray)):
                x_arr = _np.array(x)
                res.append(int(x_arr.argmax()) if x_arr.size else 0)
            elif x is None:
                res.append(0)
            else:
                res.append(int(x))
        return _np.array(res, dtype=int)

    def _as_img_list(seq):
        out = []
        for p in seq:
            if isinstance(p, (str, np.str_)):
                out.append(p)
            elif isinstance(p, np.ndarray):
                out.append(p)
            elif isinstance(p, (list, tuple)):
                out.append(np.array(p))
            else:
                out.append(p)
        return np.array(out, dtype=object)

    img        = _as_img_list(img)
    text       = np.array(_as_str_list(text), dtype=object)
    img_label  = _as_int_labels(img_label)
    text_label = _as_int_labels(text_label)
    label      = _as_int_labels(label)

    lb_img, lb_text, lbs, lb_idx = sample_labeled_data(
        args, img, text, label, num_labels, num_classes, args.dataset, index
    )
    ulb_idx = np.array(sorted(list(set(range(len(img))) - set(lb_idx))))

    if include_lb_to_ulb:
        return lb_img, lb_text, lbs, img, text, label
    else:
        return lb_img, lb_text, lbs, img[ulb_idx], text[ulb_idx], label[ulb_idx]


def sample_labeled_data(args, img, text, label,
                        num_labels, num_classes,
                        dataset, index=None, name=None):
    """
    Sample labeled data with class balance.
    """
    assert num_labels % num_classes == 0
    if index is not None:
        index = np.array(index, dtype=np.int32)
        return img[index], text[index], label[index], index

    dump_file = f"sampled_label_idx_{dataset}_{num_labels}.npy"
    dump_path = os.path.join(args.save_dir, args.save_name, dump_file)

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_img = img[lb_idx]
        lb_text = text[lb_idx]
        lbs = label[lb_idx]
        return lb_img, lb_text, lbs, lb_idx

    samples_per_class = int(num_labels / num_classes)

    lb_img, lb_text, lbs, lb_idx = [], [], [], []
    for c in range(num_classes):
        idx = np.where(label == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)
        lb_img.extend(img[idx])
        lb_text.extend(text[idx])
        lbs.extend(label[idx])

    np.save(dump_path, np.array(lb_idx))
    return (np.array(lb_img, dtype=object),
            np.array(lb_text, dtype=object),
            np.array(lbs, dtype=int),
            np.array(lb_idx, dtype=int))


def get_sampler_by_name(name):
    sampler_name_list = sorted(
        name for name in torch.utils.data.sampler.__dict__
        if not name.startswith('_') and callable(sampler.__dict__[name])
    )
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_data_loader(dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler=None,
                    replacement=True,
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True):
    """
    Package the DataLoader
    """
    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    if isinstance(data_sampler, str):
        data_sampler = get_sampler_by_name(data_sampler)

    num_replicas = 1
    if (num_epochs is not None) and (num_iters is None):
        num_samples = len(dset) * num_epochs
    elif (num_epochs is None) and (num_iters is not None):
        num_samples = batch_size * num_iters * num_replicas
    else:
        num_samples = len(dset)

    if data_sampler.__name__ == 'RandomSampler':
        data_sampler = data_sampler(dset, replacement, num_samples, generator)
    else:
        raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

    batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
    return DataLoader(dset, batch_sampler=batch_sampler,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
