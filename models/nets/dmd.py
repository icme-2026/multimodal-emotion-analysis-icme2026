# -*- coding: utf-8 -*-
"""
DMD backbone (feature decoupling + multimodal attention)
"""

import os
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, D]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Modal_Select(nn.Module):
    """3-way gating logits (text-specific / vision-specific / common)."""
    def __init__(self, input_size, hidden_size, num_classes=3):
        super(Modal_Select, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, 512*3]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # [B,3] (logits)
        return out


class Attention(nn.Module):
    """Single-head dot-product attention over 3 items (t-specific, v-specific, common)."""
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.query_weight = nn.Linear(in_dim, in_dim)
        self.key_weight   = nn.Linear(in_dim, in_dim)
        self.value_weight = nn.Linear(in_dim, in_dim)
        # use constant scale = sqrt(d)
        self.scale = float(in_dim ** 0.5)

    def forward(self, x):
        """
        x: [B, N(=3), C]
        return: [B, N(=3), C]
        """
        q = self.query_weight(x)        # [B,3,C]
        k = self.key_weight(x)          # [B,3,C]
        v = self.value_weight(x)        # [B,3,C]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / max(self.scale, 1e-6)  # [B,3,3]
        attn_weights = torch.softmax(attn_scores, dim=-1)                           # [B,3,3]
        out = torch.matmul(attn_weights, v)                                         # [B,3,C]
        return out


class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()

        # -------- Text encoder --------
        _here = os.path.dirname(__file__)
        _default_bert_dir = os.path.abspath(os.path.join(_here, '..', 'bert-base-uncased'))
        bert_dir = os.environ.get('HF_BERT_DIR', _default_bert_dir)
        try:
            self.text_model = BertModel.from_pretrained(bert_dir, local_files_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load local BERT at {bert_dir}. "
                f"Ensure it contains config.json + (pytorch_model.bin|model.safetensors) + vocab.txt"
            ) from e

        # -------- Vision backbone --------
        try:
            self.visual_model = models.resnet18(weights=None)  # torchvision>=0.13
        except Exception:
            self.visual_model = models.resnet18(pretrained=False)
        # remove fc: output [B,512,1,1]
        self.visual_model = nn.Sequential(*list(self.visual_model.children())[:-1])

        # heads
        self.v_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.t_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)
        self.m_classifier = Classifier(input_size=512, hidden_size=1024, num_classes=args.num_classes)

        self.attention = Attention(in_dim=512)
        self.modal_select_layer = Modal_Select(input_size=512 * 3, hidden_size=512, num_classes=3)

        # 1) initial projection (t:768->512, v:512->512) with Conv1d over len=1
        self.proj_l = nn.Conv1d(768, 512, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 2.1) modality-specific encoders
        self.encoder_s_l = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)
        self.encoder_s_v = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 2.2) modality-invariant encoder (shared)
        self.encoder_c = nn.Conv1d(512, 512, kernel_size=1, padding=0, bias=False)

        # 3) decoders for reconstruction
        self.decoder_l = nn.Conv1d(512 * 2, 512, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(512 * 2, 512, kernel_size=1, padding=0, bias=False)

    def forward(self, image, text):
        """
        image: [B,3,H,W]
        text : dict(inputs) for BERT
        """
        B = image.size(0)

        # ----- Text -----
        txt_feat = self.text_model(**text).pooler_output   # [B,768]
        txt_feat = self.proj_l(txt_feat.unsqueeze(-1))     # [B,512,1]

        # ----- Image -----
        img_feat4d = self.visual_model(image)              # [B,512,1,1]
        img_vec = img_feat4d.flatten(1)                    # [B,512]
        img_feat = self.proj_v(img_vec.unsqueeze(-1))      # [B,512,1]

        # ----- Modality-specific -----
        s_l = self.encoder_s_l(txt_feat).squeeze(-1)       # [B,512]
        s_v = self.encoder_s_v(img_feat).squeeze(-1)       # [B,512]
        pre_t = self.t_classifier(s_l)                     # [B,C]
        pre_v = self.v_classifier(s_v)                     # [B,C]

        # ----- Modality-common -----
        c_l = self.encoder_c(txt_feat).squeeze(-1)         # [B,512]
        c_v = self.encoder_c(img_feat).squeeze(-1)         # [B,512]
        c_m = c_l + c_v                                    # [B,512]
        pre_m = self.m_classifier(c_m)                     # [B,C]

        # Cross-heads (用于训练时的互验证/辅助约束)
        pre_m_in_v = self.v_classifier(c_m)                # [B,C]
        pre_m_in_t = self.t_classifier(c_m)                # [B,C]
        pre_v_in_m = self.m_classifier(s_v)                # [B,C]
        pre_t_in_m = self.m_classifier(s_l)                # [B,C]

        # 记录“原始公共特征”，便于 decouple 损失
        c_l_sim = c_l
        c_v_sim = c_v

        # ----- Reconstruction -----
        recon_l = self.decoder_l(torch.cat([s_l, c_l], dim=1).unsqueeze(-1))   # [B,512,1]
        recon_v = self.decoder_v(torch.cat([s_v, c_v], dim=1).unsqueeze(-1))   # [B,512,1]
        s_l_r = self.encoder_s_l(recon_l).squeeze(-1)                          # [B,512]
        s_v_r = self.encoder_s_v(recon_v).squeeze(-1)                          # [B,512]

        # ----- Gating: logits + hard index -----
        # 输入 = (t-specific, v-specific, common)
        gate_in = torch.cat([s_l, s_v, c_m], dim=1)                             # [B,512*3]
        modal_logits = self.modal_select_layer(gate_in)                         # [B,3]  (新增：返回到 res)
        modal_gates = torch.softmax(modal_logits, dim=1)                        # [B,3]
        modal_index = torch.argmax(modal_gates, dim=1)                          # [B]

        # ----- Attention over {s_l, s_v, c_m} -----
        s_l_att = s_l.unsqueeze(1)      # [B,1,512]
        s_v_att = s_v.unsqueeze(1)      # [B,1,512]
        c_m_att = c_m.unsqueeze(1)      # [B,1,512]
        att_tensor = torch.cat([s_l_att, s_v_att, c_m_att], dim=1)  # [B,3,512]
        att_m = self.attention(att_tensor)                           # [B,3,512]

        # 使用 soft gate 做可微组合，仍然保留 modal_index 供日志观察
        gate_weights = modal_gates.unsqueeze(-1)                      # [B,3,1]
        select_m = (att_m * gate_weights).sum(dim=1)                  # [B,512]
        pre_m_att = self.m_classifier(select_m)                         # [B,C]

        # for reconstruction losses / origin terms
        res = {
            'origin_l': txt_feat,           # [B,512,1]
            'origin_v': img_feat,           # [B,512,1]

            's_l': s_l, 's_v': s_v,         # [B,512]
            'c_l': c_l, 'c_v': c_v,         # [B,512]

            's_l_r': s_l_r, 's_v_r': s_v_r, # [B,512]
            'recon_l': recon_l, 'recon_v': recon_v,  # [B,512,1]

            'c_l_sim': c_l_sim, 'c_v_sim': c_v_sim,  # [B,512]

            'att_m': att_m,                 # [B,3,512]
            'modal_index': modal_index,     # [B]
            'modal_logits': modal_logits,   # [B,3]  ★ 新增：供 soft gate 使用

            'pre_t': pre_t, 'pre_v': pre_v, 'pre_m': pre_m,       # [B,C]
            'pre_m_att': pre_m_att,                              # [B,C]

            'pre_m_in_t': pre_m_in_t, 'pre_m_in_v': pre_m_in_v,   # [B,C]
            'pre_v_in_m': pre_v_in_m, 'pre_t_in_m': pre_t_in_m,   # [B,C]
        }
        return res
