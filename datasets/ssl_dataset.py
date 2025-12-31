# -*- coding: utf-8 -*-
from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot, split_ssl_data, sample_labeled_data
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
from PIL import ImageFile
import numpy as np
import copy
import os
import json
import random
import gc
from collections import Counter

# 防止图片截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
#  Mean / Std Config
# ==============================================================================
mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
mean['imagenet'] = [0.485, 0.456, 0.406]
mean['fi'] = [0.485, 0.456, 0.406]
mean['SE30K8'] = [0.485, 0.456, 0.406]
mean['mvsa-s'] = [0.485, 0.456, 0.406]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['svhn'] = [0.1751, 0.1771, 0.1744]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
std['imagenet'] = [0.229, 0.224, 0.225]
std['fi'] = [0.229, 0.224, 0.225]
std['SE30K8'] = [0.229, 0.224, 0.225]
std['mvsa-s'] = [0.229, 0.224, 0.225]


# ==============================================================================
#  Helper Functions
# ==============================================================================

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def _ensure_pil(img_obj, base_dir_candidates=None):
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, np.ndarray):
        if img_obj.ndim == 2:
            img_obj = np.stack([img_obj] * 3, axis=-1)
        return Image.fromarray(img_obj.astype(np.uint8)).convert("RGB")
    if isinstance(img_obj, (str, np.str_)):
        path = img_obj
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")
        if base_dir_candidates:
            for base in base_dir_candidates:
                cand = os.path.join(base, path)
                if os.path.isfile(cand):
                    return Image.open(cand).convert("RGB")
        raise FileNotFoundError(f"Image path not found: {img_obj}")
    raise TypeError(f"Unsupported image type: {type(img_obj)}")


def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_emotion_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size, padding=int(crop_size * 0.125),
                                                         padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.CenterCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


# ==============================================================================
#  BasicDataset (Inlined & Patched for DMD)
# ==============================================================================

class BasicDataset(Dataset):
    """
    兼容弱/强增广，返回 (idx, img_w, [img_s0, img_s1], text, target)
    """

    def __init__(self, alg, img, text, targets=None, num_classes=None, transform=None,
                 is_ulb=False, strong_transform=None, onehot=False, data_root=None, *args, **kwargs):
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.img = img
        self.text = text
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.data_root = data_root

        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
            else:
                self.strong_transform = strong_transform
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        if self.targets is None:
            target = None
        else:
            t = self.targets[idx]
            target = t if not self.onehot else get_onehot(self.num_classes, t)

        img_obj = self.img[idx]
        text = self.text[idx]

        img_pil = _ensure_pil(img_obj, base_dir_candidates=[self.data_root] if self.data_root else None)

        if self.transform is None:
            img_w = transforms.ToTensor()(img_pil)
            return idx, img_w, text, target

        img_w = self.transform(img_pil)

        if not self.is_ulb:
            return idx, img_w, text, target

        # ★★★ Patch: Added 'dmd' support here ★★★
        if self.alg in ['comatch', 'main', 'dmd']:
            img_s0 = self.strong_transform(img_pil)
            img_s1 = self.strong_transform(img_pil)
            return idx, img_w, img_s0, img_s1, text, target

        # Original Logic for other algorithms
        elif self.alg in ['fixmatch', 'uda', 'flexmatch']:
            return idx, img_w, self.strong_transform(img_pil), text, target
        elif self.alg in ['pimodel', 'meanteacher', 'mixmatch']:
            return idx, img_w, self.transform(img_pil), text, target
        elif self.alg == 'pseudolabel' or self.alg == 'vat':
            return idx, img_w, text, target
        elif self.alg == 'remixmatch':
            rotate_v_list = [0, 90, 180, 270]
            rotate_v1 = np.random.choice(rotate_v_list, 1).item()
            img_s1 = self.strong_transform(img_pil)
            img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
            img_s2 = self.strong_transform(img_pil)
            return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1), text, target
        else:
            img_s0 = self.strong_transform(img_pil)
            img_s1 = self.strong_transform(img_pil)
            return idx, img_w, img_s0, img_s1, text, target

    def __len__(self):
        return len(self.img)


# ==============================================================================
#  ImageNetDataset / ImageNetLoader (Restored)
# ==============================================================================

class ImagenetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ulb, num_labels=-1):
        super().__init__(root, transform)
        self.ulb = ulb
        self.num_labels = num_labels
        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.num_labels != -1:
                    fnames = fnames[:self.num_labels]
                if self.num_labels != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        if self.num_labels != -1:
            with open('./sampled_label_idx.json', 'w') as f:
                json.dump(lb_idx, f)
        del lb_idx
        gc.collect()
        return instances


class ImageNetLoader:
    def __init__(self, root_path, num_labels=-1, num_class=1000):
        self.root_path = os.path.join(root_path, 'imagenet')
        self.num_labels = num_labels // num_class

    def get_transform(self, train, ulb):
        if train:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        else:
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        return transform

    def get_lb_train_data(self):
        transform = self.get_transform(train=True, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=False,
                               num_labels=self.num_labels)
        return data

    def get_ulb_train_data(self):
        transform = self.get_transform(train=True, ulb=True)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=True)
        return data

    def get_lb_test_data(self):
        transform = self.get_transform(train=False, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "val"), transform=transform, ulb=False)
        return data


# ==============================================================================
#  SSL_Dataset (Restored)
# ==============================================================================

class SSL_Dataset:
    """
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self, args, alg='fixmatch', name='cifar10', train=True, num_classes=10, data_dir='./data'):
        self.args = args
        self.alg = alg
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        self.transform = get_transform(mean[name], std[name], crop_size, train)

    def get_data(self, svhn_extra=True):
        dset = getattr(torchvision.datasets, self.name.upper())
        if 'CIFAR' in self.name.upper():
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
            return data, targets
        elif self.name.upper() == 'SVHN':
            if self.train:
                if svhn_extra:  # train+extra
                    dset_base = dset(self.data_dir, split='train', download=True)
                    data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
                    dset_extra = dset(self.data_dir, split='extra', download=True)
                    data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
                    data = np.concatenate([data_b, data_e])
                    targets = np.concatenate([targets_b, targets_e])
                else:  # train_only
                    dset = dset(self.data_dir, split='train', download=True)
                    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            else:  # test
                dset = dset(self.data_dir, split='test', download=True)
                data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            return data, targets
        elif self.name.upper() == 'STL10':
            split = 'train' if self.train else 'test'
            dset_lb = dset(self.data_dir, split=split, download=True)
            dset_ulb = dset(self.data_dir, split='unlabeled', download=True)
            data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
            ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
            return data, targets, ulb_data

    def get_dset(self, is_ulb=False, strong_transform=None, onehot=False):
        if self.name.upper() == 'STL10':
            data, targets, _ = self.get_data()
        else:
            data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform

        return BasicDataset(self.alg, data, targets, num_classes, transform,
                            is_ulb, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     strong_transform=None, onehot=False):
        if self.alg == 'fullysupervised':
            lb_data, lb_targets = self.get_data()
            lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                                   self.transform, False, None, onehot)
            return lb_dset, None

        if self.name.upper() == 'STL10':
            lb_data, lb_targets, ulb_data = self.get_data()
            if include_lb_to_ulb:
                ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
            lb_data, lb_targets, _ = sample_labeled_data(self.args, lb_data, lb_targets, num_labels, self.num_classes)
            ulb_targets = None
        else:
            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
                                                                        num_labels, self.num_classes,
                                                                        index, include_lb_to_ulb)
        count = [0 for _ in range(self.num_classes)]
        for c in lb_targets:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)

        lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                               self.transform, False, None, onehot)

        ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes,
                                self.transform, True, strong_transform, onehot)
        return lb_dset, ulb_dset


# ==============================================================================
#  Emotion_SSL_Dataset (Improved with Sorting & Split Saving)
# ==============================================================================

class Emotion_SSL_Dataset:
    def __init__(self, args, alg='fixmatch', name='fi', train=True, num_classes=10, data_dir=None):
        self.args = args
        self.alg = alg
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        crop_size = 224
        base_transform = get_emotion_transform(mean[name], std[name], crop_size, train)

        self.randaugment_n = int(getattr(args, 'randaugment_n', 3))
        self.randaugment_m = int(getattr(args, 'randaugment_m', 5))
        self.weak_aug_policy = getattr(args, 'weak_aug_policy', 'default')
        self.strong_aug_policy = getattr(args, 'strong_aug_policy', 'randaugment')

        if train:
            self.transform = self._apply_policy(base_transform, self.weak_aug_policy)
            self.ulb_strong_transform = self._apply_policy(copy.deepcopy(base_transform), self.strong_aug_policy)
        else:
            self.transform = base_transform
            self.ulb_strong_transform = None

    def _apply_policy(self, base_transform, policy):
        if base_transform is None: return None
        transform_copy = copy.deepcopy(base_transform)
        if policy in (None, 'none', 'default'): return transform_copy
        if not isinstance(transform_copy, transforms.Compose): return transform_copy
        ops = copy.deepcopy(transform_copy.transforms)
        if policy == 'randaugment':
            ops.insert(0, RandAugment(self.randaugment_n, self.randaugment_m))
        return transforms.Compose(ops)

    def get_data(self):
        if self.data_dir is None:
            print('The path of dataset is empty')
            exit(-1)

        json_file = 'train.json' if self.train else 'test.json'
        with open(os.path.join(self.data_dir, json_file), "r") as f:
            anno = json.load(f)

        img = []
        text = []
        img_label = []
        text_label = []
        label = []
        ids = []

        # ★★★ 关键修改：强制排序 keys，保证顺序一致性 ★★★
        sorted_keys = sorted(list(anno.keys()))

        for id in sorted_keys:
            img_path = os.path.join(self.data_dir, 'data', id + '.jpg')
            txt_path = os.path.join(self.data_dir, 'data', id + '.txt')
            try:
                img_pil = Image.open(img_path).convert('RGB')
                img.append(np.array(img_pil))
                with open(txt_path, 'r', encoding='ISO-8859-1') as file:
                    text.append(file.read())
                img_label.append(anno[id]['img_label'])
                text_label.append(anno[id]['text_label'])
                label.append(anno[id]['label'])
                ids.append(id)
            except Exception as e:
                print(f"[Warn] Skipping {id}: {e}")
                continue

        return img, text, img_label, text_label, label, ids

    def get_dset(self, is_ulb=False, strong_transform=None, onehot=False):
        img, text, _, _, label, _ = self.get_data()
        transform = self.transform
        ulb_strong = strong_transform if strong_transform is not None else self.ulb_strong_transform
        return BasicDataset(self.alg, img, text, label, self.num_classes, transform,
                            is_ulb, ulb_strong, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     strong_transform=None, onehot=False):
        """
        支持加载固定划分的 SSL Dataset 切分逻辑
        """
        img, text, img_label, text_label, label, ids = self.get_data()

        # 检查是否有指定的 split 文件
        split_path = getattr(self.args, 'split_path', None)

        lb_img, lb_text, lb_targets = [], [], []
        ulb_img, ulb_text, ulb_targets = [], [], []

        if split_path and os.path.exists(split_path):
            print(f"Loading fixed split from: {split_path}")
            with open(split_path, 'r') as f:
                split_info = json.load(f)

            lb_id_set = set(split_info['labeled_ids'])
            for i, uid in enumerate(ids):
                if uid in lb_id_set:
                    lb_img.append(img[i])
                    lb_text.append(text[i])
                    lb_targets.append(label[i])
                else:
                    ulb_img.append(img[i])
                    ulb_text.append(text[i])
                    ulb_targets.append(label[i])

            lb_targets = np.array(lb_targets)
            ulb_targets = np.array(ulb_targets)

            if include_lb_to_ulb:
                ulb_img = ulb_img + lb_img
                ulb_text = ulb_text + lb_text
                ulb_targets = np.concatenate([ulb_targets, lb_targets])

        else:
            print(f"Generating random split (seed={self.args.seed})...")
            total_num = len(ids)
            idxs = np.arange(total_num)

            if hasattr(self.args, 'seed'):
                np.random.seed(self.args.seed)
                random.seed(self.args.seed)

            np.random.shuffle(idxs)

            lb_idxs = []
            classes = np.unique(label)
            n_labels_per_cls = num_labels // len(classes)

            for c in classes:
                cls_idxs = np.where(np.array(label) == c)[0]
                c_lb_idxs = np.random.choice(cls_idxs, n_labels_per_cls, replace=False)
                lb_idxs.extend(c_lb_idxs)

            lb_idxs = np.array(lb_idxs)
            ulb_idxs = np.setdiff1d(idxs, lb_idxs)

            for i in lb_idxs:
                lb_img.append(img[i])
                lb_text.append(text[i])
                lb_targets.append(label[i])

            for i in ulb_idxs:
                ulb_img.append(img[i])
                ulb_text.append(text[i])
                ulb_targets.append(label[i])

            lb_targets = np.array(lb_targets)
            ulb_targets = np.array(ulb_targets)

            # 保存划分
            save_dir = './splits'
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"split_{self.name}_{num_labels}_seed{self.args.seed}.json"
            save_path = os.path.join(save_dir, save_name)

            lb_ids_to_save = [ids[i] for i in lb_idxs]
            with open(save_path, 'w') as f:
                json.dump({'labeled_ids': lb_ids_to_save}, f)
            print(f"Split saved to {save_path}")

            if include_lb_to_ulb:
                ulb_img = ulb_img + lb_img
                ulb_text = ulb_text + lb_text
                ulb_targets = np.concatenate([ulb_targets, lb_targets])

        lb_dset = BasicDataset(self.alg, lb_img, lb_text, lb_targets, self.num_classes,
                               self.transform, False, None, onehot)

        ulb_strong = strong_transform if strong_transform is not None else self.ulb_strong_transform
        ulb_dset = BasicDataset(self.alg, ulb_img, ulb_text, ulb_targets, self.num_classes,
                                self.transform, True, ulb_strong, onehot)

        return lb_dset, ulb_dset
