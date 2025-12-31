from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy
import os


def _ensure_pil(img_obj, base_dir_candidates=None):
    """
    img_obj 可能是：
      - 路径字符串（相对/绝对）
      - numpy.ndarray（H,W,3 或 H,W）
      - PIL.Image
    """
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")

    if isinstance(img_obj, np.ndarray):
        if img_obj.ndim == 2:
            img_obj = np.stack([img_obj] * 3, axis=-1)
        return Image.fromarray(img_obj.astype(np.uint8)).convert("RGB")

    if isinstance(img_obj, (str, np.str_)):
        path = img_obj
        # 尝试原路径
        if os.path.isfile(path):
            return Image.open(path).convert("RGB")
        # 尝试加上候选 base_dir
        if base_dir_candidates:
            for base in base_dir_candidates:
                cand = os.path.join(base, path)
                if os.path.isfile(cand):
                    return Image.open(cand).convert("RGB")
        # 实在找不到就抛
        raise FileNotFoundError(f"Image path not found: {img_obj}")

    # 兜底
    raise TypeError(f"Unsupported image type: {type(img_obj)}")


class BasicDataset(Dataset):
    """
    兼容弱/强增广，返回 (idx, img_w, [img_s0, img_s1], text, target)
    """

    def __init__(self,
                 alg,
                 img,
                 text,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 data_root=None,
                 *args, **kwargs):
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.img = img            # dtype=object: 路径(str) 或 ndarray
        self.text = text          # dtype=object: 文本(str)
        self.targets = targets    # None 或 int labels

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.data_root = data_root  # 供相对路径补全

        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
            else:
                self.strong_transform = strong_transform
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        # 1) label
        if self.targets is None:
            target = None
        else:
            t = self.targets[idx]
            target = t if not self.onehot else get_onehot(self.num_classes, t)

        # 2) image / text
        img_obj = self.img[idx]
        text = self.text[idx]  # 已经是 str（来自 data_utils）

        img_pil = _ensure_pil(
            img_obj,
            base_dir_candidates=[self.data_root] if self.data_root else None
        )

        if self.transform is None:
            img_w = transforms.ToTensor()(img_pil)
            return idx, img_w, text, target

        img_w = self.transform(img_pil)

        if not self.is_ulb:
            return idx, img_w, text, target

        # 无标注分支：按算法要求返回
        if self.alg in ['comatch', 'main']:
            img_s0 = self.strong_transform(img_pil)
            img_s1 = self.strong_transform(img_pil)
            return idx, img_w, img_s0, img_s1, text, target
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
            # 默认与 comatch 对齐（两张强增广）
            img_s0 = self.strong_transform(img_pil)
            img_s1 = self.strong_transform(img_pil)
            return idx, img_w, img_s0, img_s1, text, target

    def __len__(self):
        return len(self.img)
