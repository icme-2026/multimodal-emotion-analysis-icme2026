# -*- coding: utf-8 -*-

import pickle
import os
from copy import deepcopy
from collections import Counter
import contextlib

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from train_utils import AverageMeter
from .main_utils import Get_Scalar, LinearWarmupScalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller, MultiClassFocalLossWithAlpha


from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

from models.nets import fusion_model, dmd
from transformers import BertModel, BertTokenizer


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1) + 1e-8)
        x_norm = torch.max(x_norm, 1e-8 * torch.ones_like(x_norm))
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1) + 1e-8)
        y_norm = torch.max(y_norm, 1e-8 * torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)
        return cosine

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape

        s = feats.repeat(1, B).view(-1, F)  # B**2 x F
        s_ids = ids.view(B, 1).repeat(1, B)  # B x B

        t = feats.repeat(B, 1)  # B**2 x F
        t_ids = ids.view(1, B).repeat(B, 1)  # B x B

        cosine = self.compute_cosine(s, t)  # B**2

        equal_mask = torch.eye(B, dtype=torch.bool, device=feats.device)
        s_ids = s_ids[~equal_mask].view(B, B - 1)  # B x (B-1)
        t_ids = t_ids[~equal_mask].view(B, B - 1)  # B x (B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B - 1)  # B x (B-1)

        sim_mask = (s_ids == t_ids)  # B x (B-1)
        margin = 0.15 * abs(s_ids - t_ids)

        loss = 0
        loss_num = 0

        for i in range(B):
            sim_num = int(sim_mask[i].sum().item())
            dif_num = B - 1 - sim_num
            if not sim_num or not dif_num:
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1

        loss = loss / loss_num
        return loss


class S2_VER:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,
                 hard_label=True, t_fn=None, p_fn=None, it=0, tb_log=None, args=None, logger=None):

        super(S2_VER, self).__init__()

        self.loader = {}
        self.num_classes = int(num_classes)
        self.ema_m = ema_m

        # model
        self.model = dmd.DMD(args)
        self.ema_model = None
        self.ema = None
        self._ema_initialized = False
        self.enable_cpl = bool(getattr(args, "enable_cpl", True))
        self.enable_mco = bool(getattr(args, "enable_mco", True))
        self.enable_drr = bool(getattr(args, "enable_drr", True))
        self._motivation_log_enabled = bool(getattr(args, "enable_motivation_log", False))
        self._motivation_records = []
        self._motivation_flush_size = int(getattr(args, "motivation_flush_size", 2048))
        self._motivation_samples_per_batch = int(getattr(args, "motivation_samples_per_batch", 32))
        self._motivation_dir = None
        self._motivation_csv_path = None
        self._motivation_header_written = False
        if self._motivation_log_enabled and args is not None:
            run_dir = os.path.join(args.save_dir, args.save_name)
            self._motivation_dir = os.path.join(run_dir, "motivation")
            os.makedirs(self._motivation_dir, exist_ok=True)
            self._motivation_csv_path = os.path.join(self._motivation_dir, "records.csv")

        self.t_fn = t_fn if t_fn is not None else Get_Scalar(T)
        self.p_fn = p_fn if p_fn is not None else Get_Scalar(p_cutoff)

        self.lambda_u = lambda_u
        self.base_p_cutoff = float(p_cutoff)
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = int(it)
        self.lst = [[] for _ in range(10)]
        self.abs_lst = [[] for _ in range(10)]
        self.clsacc = [[] for _ in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

        self.consistency_floor = float(getattr(args, "consistency_floor", 0.0)) if args is not None else 0.0
        # teacher warmup
        self.teacher_warmup_epochs = max(0, int(getattr(args, "teacher_warmup_epochs", 0))) if args is not None else 0

        # tokenizer
        _here = os.path.dirname(__file__)                # .../models/main
        _default_bert_dir = os.path.abspath(os.path.join(_here, '..', 'bert-base-uncased'))
        bert_dir = os.environ.get('HF_BERT_DIR', _default_bert_dir)
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir, local_files_only=True)

        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _texts_to_list(self, texts):

        out = []
        for t in texts:
            if isinstance(t, str):
                out.append(t)
            elif hasattr(t, "item"):
                out.append(str(t.item()))
            else:
                out.append(str(t))
        return out

    def train(self, args, epoch, best_eval_acc, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        if self.ema is None:
            self.ema = EMA(self.model, self.ema_m)
        if not self._ema_initialized:
            self.ema.register()
            self._ema_initialized = True

        # profiling events
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        sup_losses = AverageMeter()
        unsup_losses = AverageMeter()
        decouple_losses = AverageMeter()
        total_losses = AverageMeter()
        mask_ratios = AverageMeter()
        pseudo_true_ratios = AverageMeter()
        lr_last = 0
        batch_data_time = AverageMeter()
        batch_model_time = AverageMeter()

        start_batch.record()

        scaler = GradScaler(enabled=args.amp)
        amp_cm = autocast if args.amp else contextlib.nullcontext

        if args.resume is True:
            eval_dict = self.evaluate(args=args)
            self.print_fn(eval_dict)

        total_iters = max(1, int(args.epoch * len(self.loader_dict['train_ulb'])))

        ulb_rampup_ratio = getattr(args, "ulb_rampup_ratio", 0.1)
        ulb_rampup_iters = max(1, int(ulb_rampup_ratio * total_iters))

        p_cutoff_end = getattr(args, "p_cutoff_end", None)
        p_rampup_ratio = getattr(args, "p_rampup_ratio", 0.2)
        p_rampup_iters = max(1, int(p_rampup_ratio * total_iters)) if p_cutoff_end is not None else 1
        p_scheduler = self.p_fn
        if p_cutoff_end is not None and isinstance(self.p_fn, Get_Scalar):
            start_cutoff = float(self.p_fn(0))
            p_scheduler = LinearWarmupScalar(start_cutoff, p_cutoff_end, p_rampup_iters)

        label_filter_floor = float(getattr(args, "label_filter_min", 0.8))
        gate_clean_start = int(getattr(args, "gate_clean_start", 5))

        for lb_batch, ulb_batch in tqdm(
            zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']),
            total=len(self.loader_dict['train_ulb'])
        ):
            # labeled batch
            _, x_lb, t_lb, y_lb = lb_batch

            # unlabeled batch
            if len(ulb_batch) >= 6:
                x_ulb_idx, x_ulb_w, x_ulb_s0, x_ulb_s1, t_ulb, y_ulb = ulb_batch[:6]
            else:
                x_ulb_idx, x_ulb_w, x_ulb_s0, x_ulb_s1, t_ulb = ulb_batch
                y_ulb = None

            self.it += 1
            end_batch.record()
            torch.cuda.synchronize()
            batch_data_time.update(start_batch.elapsed_time(end_batch) / 1000)
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s0.shape[0] and num_ulb == x_ulb_s1.shape[0]

            x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1 = (
                x_lb.cuda(args.gpu),
                x_ulb_w.cuda(args.gpu),
                x_ulb_s0.cuda(args.gpu),
                x_ulb_s1.cuda(args.gpu),
            )
            y_lb = y_lb.cuda(args.gpu)
            if y_ulb is not None:
                y_ulb = y_ulb.cuda(args.gpu)

            img_inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s0, x_ulb_s1))

            t_lb_list = self._texts_to_list(t_lb)
            t_ulb_list = self._texts_to_list(t_ulb)
            text_input_list = t_lb_list + t_ulb_list + t_ulb_list + t_ulb_list
            text_input = self.tokenizer(text_input_list, return_tensors='pt', padding=True, truncation=True)
            text_input = {k: v.cuda(args.gpu) for k, v in text_input.items()}

            with amp_cm():
                output = self.model(img_inputs, text_input)
                logits_m = output['pre_m_att']
                logits_t = output['pre_t']
                logits_v = output['pre_v']

                # Shared features
                features_m = output['c_l'] + output['c_v']

                #labeled / unlabeled
                logits_x_lb = logits_m[:num_lb]
                logits_t_lb = logits_t[:num_lb]
                logits_v_lb = logits_v[:num_lb]

                logits_x_ulb_w_student, logits_x_ulb_s0, logits_x_ulb_s1 = torch.split(logits_m[num_lb:], num_ulb)
                logits_t_ulb_w_student, logits_t_ulb_s0, logits_t_ulb_s1 = torch.split(logits_t[num_lb:], num_ulb)
                logits_v_ulb_w_student, logits_v_ulb_s0, logits_v_ulb_s1 = torch.split(logits_v[num_lb:], num_ulb)

                features_lb = features_m[:num_lb]
                features_ulb_w, features_ulb_s0, features_ulb_s1 = torch.split(features_m[num_lb:], num_ulb)

                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                pre_v_in_m = output['pre_v_in_m'][:num_lb]
                pre_t_in_m = output['pre_t_in_m'][:num_lb]
                sup_loss_scalar = float(sup_loss.detach().cpu().item())

                if self.enable_cpl and (gate_clean_start <= epoch <= 10):
                    select_v, select_t = [], []
                    v_sup_loss, t_sup_loss = 0, 0
                    label_filter_threshold = torch.tensor(
                        max(label_filter_floor, 0.5 * sup_loss_scalar),
                        device=logits_x_lb.device,
                        dtype=logits_x_lb.dtype,
                    )
                    for i in range(len(pre_v_in_m)):
                        if (ce_loss(pre_v_in_m[i], y_lb[i]) < label_filter_threshold):
                            select_v.append(i)
                        if (ce_loss(pre_t_in_m[i], y_lb[i]) < label_filter_threshold):
                            select_t.append(i)
                    if len(select_v) > 0:
                        v_sup_loss = ce_loss(logits_v_lb[select_v], y_lb[select_v], reduction='mean')
                    if len(select_t) > 0:
                        t_sup_loss = ce_loss(logits_t_lb[select_t], y_lb[select_t], reduction='mean')
                    sup_loss = sup_loss + v_sup_loss + t_sup_loss

                modal_logits = output.get('modal_logits', None)
                if modal_logits is not None:
                    modal_gates = torch.softmax(modal_logits, dim=1)  # (B,3)
                    modal_gates_ulb_w, _, _ = torch.split(modal_gates[num_lb:], num_ulb)
                else:
                    modal_index = output['modal_index']
                    modal_index_ulb_w, _, _ = torch.split(modal_index[num_lb:], num_ulb)
                    modal_gates_ulb_w = F.one_hot(modal_index_ulb_w, num_classes=3).float()

                pre_m_in_t = output['pre_m_in_t']
                pre_m_in_t, _, _ = torch.split(pre_m_in_t[num_lb:], num_ulb)
                pre_m_in_v = output['pre_m_in_v']
                pre_m_in_v, _, _ = torch.split(pre_m_in_v[num_lb:], num_ulb)

                with torch.no_grad():
                    T_now = float(self.t_fn(self.it))
                    p_now = float(p_scheduler(self.it))

                    # EMA Teacher
                    use_teacher = self._should_use_teacher(epoch)
                    if use_teacher:
                        x_ulb_w_imgs = img_inputs[num_lb:num_lb + num_ulb]
                        text_ulb_w = {k: v[num_lb:num_lb + num_ulb] for k, v in text_input.items()}

                        self.ema.apply_shadow()
                        prev_training = self.model.training
                        self.model.eval()
                        with torch.no_grad():
                            teacher_out = self.model(x_ulb_w_imgs, text_ulb_w)
                        if prev_training:
                            self.model.train()
                        self.ema.restore()

                        logits_x_ulb_w = teacher_out['pre_m_att'].detach()
                        logits_t_ulb_w = teacher_out['pre_t'].detach()
                        logits_v_ulb_w = teacher_out['pre_v'].detach()
                    else:
                        logits_x_ulb_w = logits_x_ulb_w_student.detach()
                        logits_t_ulb_w = logits_t_ulb_w_student.detach()
                        logits_v_ulb_w = logits_v_ulb_w_student.detach()

                    features_lb = features_lb.detach()
                    features_ulb_w = features_ulb_w.detach()
                    pre_m_in_t = pre_m_in_t.detach()
                    pre_m_in_v = pre_m_in_v.detach()

                    # softmax
                    ulb_probs = torch.softmax(logits_x_ulb_w / max(T_now, 1e-6), dim=1)
                    t_ulb_probs = torch.softmax(logits_t_ulb_w / max(T_now, 1e-6), dim=1)
                    v_ulb_probs = torch.softmax(logits_v_ulb_w / max(T_now, 1e-6), dim=1)

                    scores, lbs_u_guess = torch.max(ulb_probs, dim=1)
                    t_scores, t_lbs_u_guess = torch.max(t_ulb_probs, dim=1)
                    v_scores, v_lbs_u_guess = torch.max(v_ulb_probs, dim=1)

                    if self.enable_cpl and (gate_clean_start <= epoch <= 10):
                        gate_tau = getattr(args, 'dynamic_th', 0.4)
                        gate_filter_threshold = torch.tensor(
                            max(label_filter_floor, sup_loss_scalar),
                            device=logits_x_lb.device,
                            dtype=logits_x_lb.dtype,
                        )
                        for i in range(num_ulb):
                            gate_common = modal_gates_ulb_w[i, 2]
                            if (gate_common >= gate_tau) and (gate_common >= modal_gates_ulb_w[i, :2].max()):
                                loss_m_in_t = ce_loss(pre_m_in_t[i], t_lbs_u_guess[i])
                                loss_m_in_v = ce_loss(pre_m_in_v[i], v_lbs_u_guess[i])
                                if (loss_m_in_t > gate_filter_threshold) or (loss_m_in_v > gate_filter_threshold):
                                    scores[i] = 0.0  # 过滤该样本

                    if self.enable_cpl:
                        threshold = p_now
                    else:
                        threshold = self.base_p_cutoff
                    mask = scores.ge(threshold)
                    t_mask = t_scores.ge(threshold)
                    v_mask = v_scores.ge(threshold)

                    if y_ulb is not None:
                        mask_float = mask.float()
                        denom = mask_float.sum().clamp_min(1.0)
                        pseudo_acc = ((lbs_u_guess == y_ulb).float() * mask_float).sum() / denom
                        pseudo_true_ratios.update(pseudo_acc.detach().cpu())

                if self.enable_mco:
                    w_m = (scores * mask.float()).detach()
                    w_t = (t_scores * t_mask.float()).detach()
                    w_v = (v_scores * v_mask.float()).detach()
                else:
                    w_m = mask.float().detach()
                    w_t = t_mask.float().detach()
                    w_v = v_mask.float().detach()

                #Vision–language consistency
                with torch.no_grad():
                    cons = (t_ulb_probs * v_ulb_probs).sum(dim=1).detach()  # [B] in [0,1]
                if self.enable_mco:
                    if self.consistency_floor > 0:
                        cons = cons * (1.0 - self.consistency_floor) + self.consistency_floor
                    cons = cons.clamp(0.0, 1.0)
                    w_m = w_m * cons
                    w_t = w_t * cons
                    w_v = w_v * cons
                else:
                    cons = torch.ones_like(scores)

                self._record_motivation_stats(
                    epoch=epoch,
                    iter_idx=self.it,
                    sample_ids=x_ulb_idx,
                    t_labels=t_lbs_u_guess,
                    t_scores=t_scores,
                    t_mask=t_mask,
                    w_t=w_t,
                    v_labels=v_lbs_u_guess,
                    v_scores=v_scores,
                    v_mask=v_mask,
                    w_v=w_v,
                    m_labels=lbs_u_guess,
                    m_scores=scores,
                    m_mask=mask,
                    w_m=w_m,
                    modal_gates=modal_gates_ulb_w,
                    consistency=cons,
                    threshold=p_now,
                    use_teacher=use_teacher,
                )

                #Unsupervised loss (weighted).
                unsup_loss_m = (F.cross_entropy(logits_x_ulb_s0, lbs_u_guess, reduction='none') * w_m)
                unsup_loss_t = (F.cross_entropy(logits_t_ulb_s0, t_lbs_u_guess, reduction='none') * w_t)
                unsup_loss_v = (F.cross_entropy(logits_v_ulb_s0, v_lbs_u_guess, reduction='none') * w_v)

                unsup_loss_m = unsup_loss_m.sum() / w_m.sum().clamp_min(1e-6)
                unsup_loss_t = unsup_loss_t.sum() / w_t.sum().clamp_min(1e-6)
                unsup_loss_v = unsup_loss_v.sum() / w_v.sum().clamp_min(1e-6)

                unsup_loss = unsup_loss_m + unsup_loss_t + unsup_loss_v

                # ====== decouple losses ======
                if self.enable_drr:
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon = loss_recon_l + loss_recon_v

                    loss_sl_slr = self.MSE(output['s_l'], output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'], output['s_v_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv

                    target = -torch.ones(output['s_l'].size(0), device=output['s_l'].device)
                    cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'], target)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'], target)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v

                    c_l, c_v = output['c_l_sim'], output['c_v_sim']
                    c_l_lb, c_v_lb = c_l[:num_lb], c_v[:num_lb]
                    c_l_ulb_w, c_l_ulb_s0, c_l_ulb_s1 = torch.split(c_l[num_lb:], num_ulb)
                    c_v_ulb_w, c_v_ulb_s0, c_v_ulb_s1 = torch.split(c_v[num_lb:], num_ulb)
                    ids, feats = [], []
                    for i in range(y_lb.size(0)):
                        feats.append(c_l_lb[i].view(1, -1))
                        feats.append(c_v_lb[i].view(1, -1))
                        ids.append(y_lb[i].view(1, -1))
                        ids.append(y_lb[i].view(1, -1))

                    #Pseudo-labeling on the unlabeled set
                    for i in range(num_ulb):
                        feats.append(c_l_ulb_w[i].view(1, -1))
                        feats.append(c_v_ulb_w[i].view(1, -1))
                        ids.append(lbs_u_guess[i].view(1, -1))
                        ids.append(lbs_u_guess[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    decouple_loss = loss_s_sr + loss_recon + (loss_sim + loss_ort) * 0.1
                else:
                    decouple_loss = torch.zeros(1, device=logits_x_lb.device, dtype=logits_x_lb.dtype)

                # λ_u ramp-up
                ulb_scale = min(1.0, float(self.it) / float(ulb_rampup_iters))
                lambda_u_now = self.lambda_u * ulb_scale

                total_loss = sup_loss + lambda_u_now * unsup_loss + decouple_loss

            #Counts
            sup_losses.update(sup_loss.detach().cpu())
            unsup_losses.update(unsup_loss.detach().cpu())
            decouple_losses.update(decouple_loss.detach().cpu())
            total_losses.update(total_loss.detach().cpu())
            mask_ratios.update(mask.float().mean().detach().cpu())

            lr_last = self.optimizer.param_groups[0]['lr']

            #“Backprop
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()
            batch_model_time.update(start_run.elapsed_time(end_run) / 1000)

            start_batch.record()

        #Training logs
        pseudo_ratio_str = "N/A"
        if pseudo_true_ratios.count > 0:
            pseudo_ratio_str = f"{pseudo_true_ratios.avg:.4f}"

        data_time_avg = float(batch_data_time.avg)
        model_time_avg = float(batch_model_time.avg)
        lr_last_val = float(lr_last)
        sup_avg = float(sup_losses.avg)
        unsup_avg = float(unsup_losses.avg)
        decouple_avg = float(decouple_losses.avg)
        total_avg = float(total_losses.avg)
        mask_ratio_avg = float(mask_ratios.avg)

        self.print_fn(
            "Epoch {}/{} train: data time: {:.4f}, model time: {:.4f}, last lr: {:.8f}, labeled loss: {:.4f}, "
            "unlabeled loss: {:.4f}, decouple_loss: {:.4f}, total_loss: {:.4f}, mask ratio: {:.4f}, pseudo label correct ratio: {}".format(
                epoch, args.epoch, data_time_avg, model_time_avg, lr_last_val,
                sup_avg, unsup_avg, decouple_avg, total_avg,
                mask_ratio_avg, pseudo_ratio_str
            )
        )

        # evaluate
        eval_dict = self.evaluate(args=args)
        best_eval_acc = max(best_eval_acc, eval_dict['eval/top-1-acc'])

        k = min(5, max(1, self.num_classes - 1))
        topk_key = f'eval/top-{k}-acc'
        macro_f1 = eval_dict['eval/macro-f1']

        self.print_fn(
            "Epoch {}/{} test: test loss: {:.4f}, top-1 acc: {:.4f}, top-{} acc: {:.4f}, macro-F1: {:.4f}, best top-1 acc: {:.4f}".format(
                epoch, args.epoch,
                float(eval_dict['eval/loss']),
                float(eval_dict['eval/top-1-acc']),
                k, float(eval_dict[topk_key]),
                float(macro_f1),
                float(best_eval_acc),
            )
        )

        save_path = os.path.join(args.save_dir, args.save_name)
        if eval_dict['eval/top-1-acc'] >= best_eval_acc:
            self.save_model('model_best.pth', save_path)
        self._flush_motivation_records()
        return eval_dict['eval/top-1-acc']

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        if self.ema is None:
            self.ema = EMA(self.model, self.ema_m)
        if not self._ema_initialized:
            self.ema.register()
            self._ema_initialized = True
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']

        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        num_classes = int(self.num_classes)
        k = min(5, max(1, num_classes - 1))

        for _, x, text_input, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)

            text_input = self.tokenizer(text_input, return_tensors='pt', padding=True, truncation=True)
            text_input = {key: value.cuda(args.gpu) for key, value in text_input.items()}
            num_batch = x.shape[0]
            total_num += num_batch
            output = self.model(x, text_input)
            logits = output['pre_m_att']
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().detach().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().detach().tolist())
            total_loss += loss.cpu().detach() * num_batch

        #“Metrics
        top1 = accuracy_score(y_true, y_pred)
        topk = top_k_accuracy_score(y_true, y_logits, k=k) if k > 1 else top1

        # Macro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        self.ema.restore()
        self.model.train()
        return {
            'eval/loss': (total_loss / total_num),
            'eval/top-1-acc': top1,
            f'eval/top-{k}-acc': topk,
            'eval/macro-f1': macro_f1,
        }

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)
        if self.num_classes == 10:
            tb_path = os.path.join(save_path, 'tensorboard')
            if not os.path.exists(tb_path):
                os.makedirs(tb_path, exist_ok=True)
            with open(os.path.join(save_path, 'tensorboard', 'lst_fix.pkl'), 'wb') as f:
                pickle.dump(self.lst, f)
            with open(os.path.join(save_path, 'tensorboard', 'abs_lst.pkl'), 'wb') as h:
                pickle.dump(self.abs_lst, h)
            with open(os.path.join(save_path, 'tensorboard', 'clsacc.pkl'), 'wb') as g:
                pickle.dump(self.clsacc, g)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        if self.ema is None:
            self.ema = EMA(self.model, self.ema_m)
        self.ema.load(self.ema_model)
        self._ema_initialized = True
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def _record_motivation_stats(self, epoch, iter_idx, sample_ids, t_labels, t_scores, t_mask, w_t,
                                 v_labels, v_scores, v_mask, w_v, m_labels, m_scores, m_mask, w_m,
                                 modal_gates, consistency, threshold, use_teacher):
        if not self._motivation_log_enabled or self._motivation_samples_per_batch <= 0:
            return
        num_samples = len(sample_ids)
        if num_samples == 0:
            return
        record_count = min(num_samples, self._motivation_samples_per_batch)
        if record_count <= 0:
            return
        if num_samples > record_count:
            sel = torch.randperm(num_samples)[:record_count].tolist()
        else:
            sel = list(range(num_samples))

        def _to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x)

        sample_ids = _to_list(sample_ids)
        t_labels = _to_list(t_labels)
        t_scores = _to_list(t_scores)
        t_mask = _to_list(t_mask)
        w_t = _to_list(w_t)
        v_labels = _to_list(v_labels)
        v_scores = _to_list(v_scores)
        v_mask = _to_list(v_mask)
        w_v = _to_list(w_v)
        m_labels = _to_list(m_labels)
        m_scores = _to_list(m_scores)
        m_mask = _to_list(m_mask)
        w_m = _to_list(w_m)
        modal_gates = _to_list(modal_gates)
        consistency = _to_list(consistency)

        for idx in sel:
            record = {
                'epoch': int(epoch),
                'iter': int(iter_idx),
                'sample_id': int(sample_ids[idx]),
                'text_label': int(t_labels[idx]),
                'vision_label': int(v_labels[idx]),
                'fusion_label': int(m_labels[idx]),
                'text_conf': float(t_scores[idx]),
                'vision_conf': float(v_scores[idx]),
                'fusion_conf': float(m_scores[idx]),
                'text_mask': int(t_mask[idx]),
                'vision_mask': int(v_mask[idx]),
                'fusion_mask': int(m_mask[idx]),
                'text_weight': float(w_t[idx]),
                'vision_weight': float(w_v[idx]),
                'fusion_weight': float(w_m[idx]),
                'modal_gate_text': float(modal_gates[idx][0]),
                'modal_gate_vision': float(modal_gates[idx][1]),
                'modal_gate_common': float(modal_gates[idx][2]),
                'consistency': float(consistency[idx]),
                'is_conflict': int(t_labels[idx] != v_labels[idx]),
                'p_threshold': float(threshold),
                'use_teacher': int(use_teacher),
            }
            self._motivation_records.append(record)
            if len(self._motivation_records) >= self._motivation_flush_size:
                self._flush_motivation_records()

    def _flush_motivation_records(self):
        if (not self._motivation_log_enabled) or (not self._motivation_records):
            return
        if self._motivation_csv_path is None:
            return
        df = pd.DataFrame(self._motivation_records)
        header = (not self._motivation_header_written) or (not os.path.exists(self._motivation_csv_path))
        df.to_csv(self._motivation_csv_path, mode='a', index=False, header=header)
        self._motivation_records = []
        self._motivation_header_written = True

    def _should_use_teacher(self, epoch: int) -> bool:

        if not self._ema_initialized:
            return False
        return epoch >= self.teacher_warmup_epochs

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
