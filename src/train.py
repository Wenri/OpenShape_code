import torch
import numpy as np
import wandb
import logging
import os
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm
import sys

class Trainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale']) #module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))


    def _contrastive_from_logits(self, logits):
        # logits: (N, N) already scaled
        device = logits.device
        N = logits.size(0)
        labels = torch.arange(N, device=device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss   = 0.5 * (loss_i + loss_t)
        acc_i  = (logits.argmax(dim=1) == labels).float().mean()
        acc_t  = (logits.argmax(dim=0) == labels).float().mean()
        acc    = 0.5 * (acc_i + acc_t)
        return loss, acc

    @torch.no_grad()
    def _pad_tokens(self, toks_list, mask_list, max_T=None, device='cuda'):
        # toks_list: list of (Ti, D); mask_list: list of (Ti,) bool
        T_max = max(len(m) for m in mask_list)
        if (max_T is not None) and (max_T > 0):
            T_max = min(T_max, max_T)
        D = toks_list[0].shape[-1]
        B = len(toks_list)
        toks = torch.zeros(B, T_max, D, device=device, dtype=toks_list[0].dtype)
        mask = torch.zeros(B, T_max,   device=device, dtype=torch.bool)
        for i, (t, m) in enumerate(zip(toks_list, mask_list)):
            t = t[:T_max].to(device)
            m = m[:T_max].to(device)
            toks[i, :t.size(0)] = t
            mask[i, :m.size(0)] = m
        return toks, mask


    # def _sparc_local_loss(self, v_tok, t_tok, t_mask, logit_scale, sigma=0.0, eps=1e-6):
    #     """
    #     v_tok:  (B, P, C)  L2-normalized 3D patch tokens
    #     t_tok:  (B, T, C)  L2-normalized text tokens
    #     t_mask: (B, T)     bool mask of valid text tokens
    #     returns: loss (scalar), tok_acc (scalar), keep_frac, patches_per_tok, weights_entropy
    #     """
        
    #     device = v_tok.device
    #     B, P, C = v_tok.shape
    #     _, T, _ = t_tok.shape

    #     # 1) token–patch sims: (B, T, P)
    #     sim = torch.einsum('btc,bpc->btp', t_tok, v_tok)

    #     # 2) per-token min–max normalization across patches
    #     smin = sim.amin(dim=-1, keepdim=True)
    #     smax = sim.amax(dim=-1, keepdim=True)
    #     s_hat = (sim - smin) / (smax - smin + eps)

    #     # 3) hard threshold (default sigma = 1/P if not given)
    #     if sigma <= 0.0:
    #         sigma = 1.0 / float(P)
    #     keep = (s_hat >= sigma)
    #     s_sparse = s_hat * keep

    #     # 4) weights & language-grouped vision per token
    #     denom = s_sparse.sum(dim=-1, keepdim=True).clamp_min(eps)
    #     w = s_sparse / denom                                # (B, T, P)
    #     c = torch.einsum('btp,bpc->btc', w, v_tok)          # (B, T, C)

    #     # normalize c and t along channel
    #     c = F.normalize(c, dim=-1)
    #     t = F.normalize(t_tok, dim=-1)

    #     # 5) per-sample token↔token logits: (B, T, T)
    #     # use batched matmul to avoid einsum label errors
    #     sim_tt = logit_scale * (c @ t.transpose(1, 2))      # (B, T, C) @ (B, C, T)

    #     # mask out padded tokens on rows & cols
    #     valid = t_mask.to(device)
    #     if valid.dtype != torch.bool:
    #         valid = valid.bool()
    #     large_neg = torch.finfo(sim_tt.dtype).min
    #     row_mask = valid.unsqueeze(2)                       # (B, T, 1)
    #     col_mask = valid.unsqueeze(1)                       # (B, 1, T)
    #     keep_mat = row_mask & col_mask                      # (B, T, T)
    #     sim_tt = sim_tt.masked_fill(~keep_mat, large_neg)

    #     # CE targets are 0..T-1 (diagonal)
    #     labels = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    #     # c -> t (rows)
    #     loss_rows = F.cross_entropy(sim_tt.reshape(B*T, T), labels.reshape(-1), reduction='none')
    #     loss_rows = (loss_rows.view(B, T) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    #     # t -> c (cols) == CE on transpose
    #     sim_tt_T = sim_tt.transpose(1, 2)
    #     loss_cols = F.cross_entropy(sim_tt_T.reshape(B*T, T), labels.reshape(-1), reduction='none')
    #     loss_cols = (loss_cols.view(B, T) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    #     loss = 0.5 * (loss_rows + loss_cols)

    #     # simple accuracy proxy over valid tokens
    #     with torch.no_grad():
    #         pred = sim_tt.argmax(dim=-1)                                # (B, T)
    #         diag = torch.arange(T, device=device).unsqueeze(0)          # (1, T)
    #         tok_acc = (((pred == diag) & valid).float().sum()
    #                 / valid.float().sum().clamp_min(1.0))

    #         keep_frac = keep.float().mean()
    #         patches_per_tok = keep.sum(dim=-1)[valid].float().mean() if valid.any() else torch.tensor(0., device=device)
    #         weights_entropy = (-(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1))
    #         weights_entropy = weights_entropy[valid].mean() if valid.any() else torch.tensor(0., device=device)

    #     return loss, tok_acc, keep_frac.detach(), patches_per_tok.detach(), weights_entropy.detach()

    def _sparc_local_loss(self, v_tok, t_tok, t_mask, logit_scale, sigma=0.0, eps=1e-10):
        """
        v_tok:  (B, P, C)  L2-normalized 3D patch tokens
        t_tok:  (B, T, C)  L2-normalized text tokens
        t_mask: (B, T)     bool mask of valid text tokens
        returns: loss (scalar), tok_acc (scalar), keep_frac, patches_per_tok, weights_entropy
        """
        device = v_tok.device
        B, P, C = v_tok.shape
        _, T, _ = t_tok.shape

        # 1) token–patch sims: (B, T, P)
        # sim = torch.einsum('btc,bpc->btp', t_tok, v_tok)
        sim = torch.einsum('btc,bpc->btp', t_tok, v_tok)

        # 2) per-token min–max normalization across patches (SPARC Eq. 2) 
        smin = sim.amin(dim=-1, keepdim=True)
        smax = sim.amax(dim=-1, keepdim=True)
        s_hat = (sim - smin) / (smax - smin)

        # 3) sparsify: per-token Top-K (preferred) or ratio; else fallback to sigma (SPARC Eq. 3)
        topk_cfg   = int(getattr(self.config.training, "sparc_topk", 0))
        keep_ratio = float(getattr(self.config.training, "sparc_keep_ratio", 0.0))

        if topk_cfg > 0 or keep_ratio > 0:
            K = topk_cfg if topk_cfg > 0 else max(1, int(P * keep_ratio))
            K = max(1, min(P, K))
            # per-token threshold = K-th largest value
            thresh_vals, _ = torch.topk(s_hat, K, dim=-1)
            sigma_tok = thresh_vals[..., -1:].detach()     # (B, T, 1)
            keep = (s_hat >= sigma_tok)                    # (B, T, P) bool
        else:
            if sigma <= 0.0:
                sigma = 1.0 / float(P)                     # original SPARC heuristic
            keep = (s_hat >= sigma)

        # ensure every VALID text token keeps at least one patch
        valid = t_mask.to(device)
        if valid.dtype != torch.bool:
            valid = valid.bool()
        no_keep = (keep.sum(dim=-1, keepdim=True) == 0) & valid.unsqueeze(-1)  # (B, T, 1)
        if no_keep.any():
            argmax_idx = s_hat.argmax(dim=-1, keepdim=True)                     # (B, T, 1)
            keep = keep.float()
            # zero existing (for those tokens) then set argmax
            keep = keep * (~no_keep).float()
            keep.scatter_(-1, argmax_idx, 1.0)
            keep = keep.bool()

        s_sparse = s_hat * keep

        # 4) alignment weights & language-grouped vision per token (SPARC Eq. 4–5)
        denom = s_sparse.sum(dim=-1, keepdim=True)
        w = s_sparse / denom                                   # (B, T, P)
        c = torch.einsum('btp,bpc->btc', w, v_tok)             # (B, T, C)

        # normalize c and t along channel
        c = F.normalize(c, dim=-1)
        t = F.normalize(t_tok, dim=-1)

        # 5) per-sample token↔token logits: (B, T, T)  (SPARC Eq. 6, sequence-wise)
        sim_tt = logit_scale * (c @ t.transpose(1, 2))

        # mask out padded tokens on rows & cols
        large_neg = torch.finfo(sim_tt.dtype).min
        row_mask = valid.unsqueeze(2)                          # (B, T, 1)
        col_mask = valid.unsqueeze(1)                          # (B, 1, T)
        keep_mat = row_mask & col_mask                         # (B, T, T)
        sim_tt = sim_tt.masked_fill(~keep_mat, large_neg)

        # CE targets are 0..T-1 (diagonal)
        labels = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # c -> t (rows)
        loss_rows = F.cross_entropy(sim_tt.reshape(B*T, T), labels.reshape(-1), reduction='none')
        loss_rows = (loss_rows.view(B, T) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

        # t -> c (cols)
        sim_tt_T = sim_tt.transpose(1, 2)
        loss_cols = F.cross_entropy(sim_tt_T.reshape(B*T, T), labels.reshape(-1), reduction='none')
        loss_cols = (loss_cols.view(B, T) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

        loss = 0.5 * (loss_rows + loss_cols)

        # simple accuracy proxy + diagnostics
        with torch.no_grad():
            pred = sim_tt.argmax(dim=-1)                                  # (B, T)
            diag = torch.arange(T, device=device).unsqueeze(0)            # (1, T)
            tok_acc = (((pred == diag) & valid).float().sum() / valid.float().sum().clamp_min(1.0))

            # FIX: compute keep_frac only on valid tokens
            keep_frac = keep[valid].float().mean() if valid.any() else torch.tensor(0., device=device)
            patches_per_tok = keep.sum(dim=-1)[valid].float().mean() if valid.any() else torch.tensor(0., device=device)

            weights_entropy = (-(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1))
            weights_entropy = weights_entropy[valid].mean() if valid.any() else torch.tensor(0., device=device)

        return loss, tok_acc, keep_frac.detach(), patches_per_tok.detach(), weights_entropy.detach()


    def _filip_logits(self, v_tokens, t_tokens, t_mask, logit_scale):
        """
        v_tokens: (N, P, C)   3D patch tokens (L2-normalized)
        t_tokens: (N, T, C)   text tokens (L2-normalized)
        t_mask:   (N, T) bool valid text tokens
        returns:  (N, N) logits using FILIP late interaction with top-K selection
        """
        device = v_tokens.device
        N, P, C = v_tokens.shape
        _, T, _ = t_tokens.shape

        # keep ratios (like FILIP Table 5: 25%)  --> Kp, Kt vary per sample
        keep_p = float(getattr(self.config.training, "filip_keep_ratio_p", 0.25))
        keep_t = float(getattr(self.config.training, "filip_keep_ratio_t", 0.25))
        Kp = max(1, int(P * keep_p))

        # precompute per-row (i) top-K patch indices using its *positive* text (i)
        # and per-text (j) top-K token indices using its *positive* 3D tokens (j)
        topP_idx = [None] * N
        topT_idx = [None] * N

        for i in range(N):
            Vi = v_tokens[i]                         # (P, C)
            Ti = t_tokens[i]                         # (T, C)
            Mi = t_mask[i].to(device)
            if Mi.dtype != torch.bool:
                Mi = Mi.bool()

            # positive-pair sims (valid text rows only): (T_valid, P)
            if Mi.any():
                sim_pos = Ti[Mi] @ Vi.t()
            else:
                sim_pos = Ti @ Vi.t()

            # --- select top-K patches for row i (used against all texts j) ---
            score_p = sim_pos.max(dim=0).values      # (P,)
            topP_idx[i] = torch.topk(score_p, Kp, largest=True).indices

            # --- select top-K text tokens for column i (used by text j=i against all rows) ---
            # keep a *ratio* of valid text tokens; fall back to at least 1
            Tv = sim_pos.shape[0]
            Kt_i = max(1, int(Tv * keep_t))
            score_t = sim_pos.max(dim=1).values      # (T_valid,)
            top_local = torch.topk(score_t, Kt_i, largest=True).indices  # indices within valid slice

            if Mi.any():
                valid_idx = Mi.nonzero(as_tuple=False).squeeze(1)
                topT_idx[i] = valid_idx[top_local]   # map back to [0..T)
            else:
                topT_idx[i] = top_local              # no mask case

        # build NxN logits using reduced tokens
        logits = v_tokens.new_zeros(N, N)
        for i in range(N):
            Vi = v_tokens[i][topP_idx[i]]            # (Kp, C)
            for j in range(N):
                Tj_idx = topT_idx[j]
                if Tj_idx.numel() == 0:
                    # degenerate: if a text has no valid tokens, just take all
                    Tj_idx = torch.arange(T, device=device)
                Tj = t_tokens[j][Tj_idx]             # (Kt_j, C)

                # token-wise sims: (Kt_j, Kp)
                sim = Tj @ Vi.t()

                # FILIP symmetric averaging (Eq. 4 & 5): mean of row/col-wise maxima
                s_I = sim.max(dim=1).values.mean()   # avg over text tokens of max over patches
                s_T = sim.max(dim=0).values.mean()   # avg over patches of max over text tokens
                logits[i, j] = 0.5 * (s_I + s_T)

        return logit_scale * logits


    def contras_loss(self, feat1, feat2, logit_scale=1, mask = None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        pbar = None
        iterator = self.train_loader
        if self.rank == 0:
            try:
                pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"train|epoch {self.epoch}", dynamic_ncols=True)
                iterator = pbar
            except Exception:
                pbar = None
        for data in iterator:
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            if not self.config.model.get("use_dense", False):
                out = self.model(data['xyz'], data['features'], \
                                        device = self.config.device, \
                                        quantization_size = self.config.model.voxel_size)
            else:
                out = self.model(data['xyz_dense'], data['features_dense'])
            if isinstance(out, (tuple, list)):
                pred_feat, pred_tokens = out           # (B, C), (B, P, C)
            else:
                pred_feat, pred_tokens = out, None
            logit_scale = self.logit_scale(None)
            idx = data['has_text_idx']

            text_feat = torch.vstack(data['text_feat']).to(self.config.device, non_blocking=True)

            # if isinstance(data['img_feat'], (list, tuple)):
            #     dims = [t.shape[-1] for t in data['img_feat']]
            #     if len(set(dims)) > 1:
            #         names    = list(data.get('name',    ['?'] * len(dims)))
            #         datasets = list(data.get('dataset', ['?'] * len(dims)))
            #         groups   = list(data.get('group',   ['?'] * len(dims)))  # <- category number if your dataset sets it

            #         offenders = [i for i, d in enumerate(dims) if d != dims[0]]
            #         print("[DEBUG] img_feat dim mismatch",
            #             f"\n  dims: {dims}",
            #             f"\n  offenders idx: {offenders}",
            #             f"\n  offender names: {[names[i] for i in offenders]}",
            #             f"\n  offender datasets: {[datasets[i] for i in offenders]}",
            #             f"\n  offender category_numbers: {[groups[i] for i in offenders]}",  # <- prints the number before the id
            #             file=sys.stderr, flush=True)
            #         raise RuntimeError(f"img_feat dim mismatch: {dims}")

            img_feat = torch.vstack(data['img_feat']).to(self.config.device, non_blocking=True) 

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            if len(idx) > 0:
                if self.config.training.use_text_proj:
                    text_feat = self.text_proj(text_feat)
                text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat, logit_scale=logit_scale, mask=mask)
                loss += text_contras_loss * self.config.training.lambda_text_contras 
                text_contras_acc_list.append(text_contras_acc.item())

            if self.config.training.use_image_proj:
                img_feat = self.image_proj(img_feat)
            img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale, mask=mask)
            loss += img_contras_loss * self.config.training.lambda_img_contras

            # --- NEW: 3D<->Text token late interaction (FILIP-style) ---
            # tok_loss = None; tok_acc = None
            # if (pred_tokens is not None) and self.config.training.get("use_text_token_interaction", False):
            #     idx = data['has_text_idx']          # rows in B that have text
            #     # require token lists aligned with text list
            #     if len(idx) > 0 and len(data.get("text_token_feat", [])) == len(data.get("text_feat", [])) and len(data["text_token_feat"]) > 0:
            #         # gather (N_txt, P, C)
            #         v_tok = pred_tokens[idx].to(self.config.device)
            #         # text tokens: pad and (optionally) project to C
            #         t_tok, t_m = self._pad_tokens(
            #             data["text_token_feat"], data["text_token_mask"],
            #             max_T=int(getattr(self.config.training, "max_text_tokens", 0)), device=self.config.device
            #         )
            #         if self.config.training.use_text_proj:
            #             B, T, D = t_tok.shape
            #             t_tok = self.text_proj(t_tok.reshape(B*T, D)).reshape(B, T, -1)
                        
            #         # l2-normalize
            #         v_tok = F.normalize(v_tok, dim=-1)
            #         t_tok = F.normalize(t_tok, dim=-1)
            #         # FILIP logits (N_txt x N_txt)
            #         filip_logits = self._filip_logits(v_tok, t_tok, t_m, logit_scale)
            #         tok_loss, tok_acc = self._contrastive_from_logits(filip_logits)
            #         loss += tok_loss * float(getattr(self.config.training, "lambda_text_token_interaction", 0.1))
            # Prepare tokens once for any local head (SPARC / FILIP), only for rows that have text
            v_tok = t_tok = t_m = None
            if (pred_tokens is not None) and (len(idx) > 0) and \
               ("text_token_feat" in data) and ("text_token_mask" in data) and \
               (len(data["text_token_feat"]) == len(data["text_feat"])):
                v_tok = pred_tokens[idx].to(self.config.device)  # (N_txt, P, C)
                t_tok, t_m = self._pad_tokens(
                    data["text_token_feat"], data["text_token_mask"],
                    max_T=int(getattr(self.config.training, "max_text_tokens", 0)),
                    device=self.config.device
                )                                             # (N_txt, T, D_txt), (N_txt, T)
                if self.config.training.use_text_proj:
                    Bn, Tn, Dn = t_tok.shape
                    t_tok = self.text_proj(t_tok.reshape(Bn*Tn, Dn)).reshape(Bn, Tn, -1)
                # L2-normalize token embeddings
                v_tok = F.normalize(v_tok, dim=-1)
                t_tok = F.normalize(t_tok, dim=-1)

            if self.config.training.use_sparc_local:
                sigma = getattr(self.config.training, "sparc_sigma", 1.0 / v_tok.shape[1])
                eps = getattr(self.config.training, "sparc_eps", 1e-6)
                sparc_loss, sparc_tok_acc, keep_frac, patches_per_tok, weights_entropy = self._sparc_local_loss(v_tok, t_tok, t_m, logit_scale, sigma, eps)
                loss = self.config.training.global_lambda *loss + self.config.training.sparc_lambda * sparc_loss

            loss.backward()
            self.optimizer.step()
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()
            
            img_contras_acc_list.append(img_contras_acc.item())

            if self.rank == 0 and self.step % self.config.training.log_freq == 0:
                # update progress bar postfix if running with tqdm
                if 'pbar' in locals() and pbar is not None:
                    try:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            img_acc=f"{img_contras_acc.item():.3f}",
                            txt_acc=(f"{text_contras_acc.item():.3f}" if len(idx) > 0 else "0.000"),
                            tok_txt_acc=(f"{sparc_tok_acc.item():.3f}" if sparc_tok_acc is not None else "n/a"),
                            lr=self.optimizer.param_groups[0]['lr']
                        )
                    except Exception:
                        pass
                try:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/text_contras_loss": text_contras_loss.item() if len(idx) > 0 else 0,
                        "train/img_contras_loss": img_contras_loss.item(),
                        "train/text_contras_acc": text_contras_acc.item() if len(idx) > 0 else 0,
                        "train/img_contras_acc": img_contras_acc.item(),
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/epoch": self.epoch,
                        "train/step": self.step,
                        "train/logit_scale": logit_scale,
                        "train/has_text": len(idx),
                        "train/filtered_pair": (mask == False).sum() if mask is not None else 0,
                        "train/sparc_loss": sparc_loss.item() if 'sparc_loss' in locals() else 0,
                        "train/sparc_tok_acc": sparc_tok_acc.item() if 'sparc_loss' in locals() else 0,
                        "train/sparc/keep_frac": keep_frac.item() if 'sparc_loss' in locals() else 0,
                        "train/sparc/avg_patches_per_token": patches_per_tok.item() if 'sparc_loss' in locals() else 0,
                        "train/sparc/weights_entropy": weights_entropy.item() if 'sparc_loss' in locals() else 0,
                        })
                except:
                    print("wandb log error", flush=True)
        if self.rank == 0: 
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}'\
                    .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                            np.mean(img_contras_acc_list)))

    def save_model(self, name):
        torch.save({
                    "state_dict": self.model.state_dict(),
                    "logit_scale": self.logit_scale.state_dict(),#module.logit_scale,
                    "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
                    "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
                    "epoch": self.epoch,
                    "step": self.step,
                    "best_img_contras_acc": self.best_img_contras_acc,
                    "best_text_contras_acc": self.best_text_contras_acc,
                    "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
                    "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
                    "best_lvis_acc": self.best_lvis_acc,
                }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct
    
    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if self.rank == 0:
                self.save_model('latest')
                self.test_modelnet40()
                self.test_objaverse_lvis()
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))
                        
    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).to(self.config.device)
        per_cat_count = torch.zeros(40).to(self.config.device)
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in (tqdm(self.modelnet40_loader, desc='eval|modelnet40', dynamic_ncols=True) if self.rank==0 else self.modelnet40_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat, _ = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat, _ = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

                for i in range(40):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count
        #for i in range(40):
        #    print(idx2category[i], per_cat_acc[i])

        if overall_acc > self.best_modelnet40_overall_acc:
            self.best_modelnet40_overall_acc = overall_acc
            self.save_model('best_modelnet40_overall')
        if per_cat_acc.mean() > self.best_modelnet40_class_acc:
            self.best_modelnet40_class_acc = per_cat_acc.mean()
            self.save_model('best_modelnet40_class')

        logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc, self.best_modelnet40_overall_acc, per_cat_acc.mean(), self.best_modelnet40_class_acc))
        logging.info('Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        wandb.log({"test/epoch": self.epoch,
                   "test/step": self.step,
                   "test/ModelNet40_overall_acc": overall_acc,
                   "test/ModelNet40_class_acc": per_cat_acc.mean(),
                   "test/top3_acc": topk_acc[1],
                   "test/top5_acc": topk_acc[2],})

    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).to(self.config.device)
        per_cat_count = torch.zeros(1156).to(self.config.device)
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat, _ = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat, _ = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in torch.unique(labels):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        if overall_acc > self.best_lvis_acc:
            self.best_lvis_acc = overall_acc
            self.save_model('best_lvis')

        logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        wandb.log({"test_lvis/epoch": self.epoch,
                   "test_lvis/step": self.step,
                   "test_lvis/overall_acc": overall_acc,
                   "test_lvis/class_acc": per_cat_acc.mean(),
                   "test_lvis/top3_acc": topk_acc[1],
                   "test_lvis/top5_acc": topk_acc[2],})
        
        
    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}
        
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in (tqdm(self.scanobjectnn_loader, desc='eval|scanobjectnn', dynamic_ncols=True) if self.rank==0 else self.scanobjectnn_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat, _ = self.model(data['xyz'], data['features'], \
                                            device = self.config.device, \
                                            quantization_size = self.config.model.voxel_size)
                else:
                    pred_feat, _ = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(15):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()

        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
        wandb.log({"test_scanobjectnn/epoch": self.epoch,
                   "test_scanobjectnn/step": self.step,
                   "test_scanobjectnn/overall_acc": overall_acc,
                   "test_scanobjectnn/class_acc": per_cat_acc.mean(),
                   "test_scanobjectnn/top3_acc": topk_acc[1],
                   "test_scanobjectnn/top5_acc": topk_acc[2],})
