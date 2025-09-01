import torch
import json
import numpy as np
import random
from utils.data import random_rotate_z, normalize_pc, augment_pc
from torch.utils.data import Dataset, DataLoader
import logging
import copy
from torch.utils.data.distributed import DistributedSampler
import os

def _tokfeat_path_from_entry(meta: dict, tok_root: str):
    ds  = meta.get("dataset", "").strip()
    grp = meta.get("group", "").strip()
    uid = meta.get("id") or os.path.splitext(os.path.basename(meta["data_path"]))[0]
    base = os.path.join(tok_root, ds)
    if grp:
        base = os.path.join(base, grp)
    cand1 = os.path.join(base, f"{uid}.tokfeat.npz")
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(base, f"{uid}.npy.tokfeat.npz")
    if os.path.exists(cand2):
        return cand2
    return None

def _resolve_npy_path(meta, pc_root):
    """
    If meta['data_path'] exists, use it.
    Else fall back to: <pc_root>/<dataset>/<group>/<id>.npy
    """
    p = meta.get("data_path", "")
    if p and os.path.isabs(p) and os.path.exists(p):
        return p

    ds  = meta.get("dataset", "").strip()
    grp = meta.get("group", "").strip()
    uid = meta.get("id",    "").strip()
    cand = os.path.join(pc_root, ds, grp, f"{uid}.npy")
    return cand  # let the original loader raise if it truly doesn't exis



class Four(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.pc_root = '/mnt/disk1/souhail_data/objaverse-processed/merged_for_training_final'
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_objaverse(self, meta):
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        uid = meta["id"]
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]
        
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                texts.append(data["text"][0])
                text_feat.append(data["text_feat"][0][self.text_embed_version])
        
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
                    text_feat.append(data["blip_caption_feat"][self.text_embed_version])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])
                    text_feat.append(data["msft_caption_feat"][self.text_embed_version])
        
        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])
                text_feat.append(data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text

        if len(text_feat) > 0:
            assert len(text_feat) == len(texts)
            text_idx = np.random.randint(len(texts))
            text_feat = text_feat[text_idx]
            texts = texts[text_idx]
            text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
        else:
            text_feat = None
            texts = None

        if np.random.rand() < 0.5:
            img_feat = data['thumbnail_feat']
        else:
            idx = np.random.randint(data['image_feat'].shape[0])
            img_feat = data["image_feat"][idx]

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "text_feat": text_feat,
            "dataset": "Objaverse",
            "group": meta["group"],
            "name":  uid,
            "texts": texts,
            "has_text": text_feat is not None,
        }
    
    def get_others(self, meta):
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            idx = np.random.randint(len(data["text"]))
            texts.append(data["text"][idx])
            text_feat.append(data["text_feat"][idx][self.text_embed_version])
        
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
                    text_feat.append(data["blip_caption_feat"][self.text_embed_version])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])
                    text_feat.append(data["msft_caption_feat"][self.text_embed_version])
        
        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])
                text_feat.append(data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text

        if len(text_feat) > 0:
            assert len(text_feat) == len(texts)
            text_idx = np.random.randint(len(texts))
            text_feat = text_feat[text_idx]
            texts = texts[text_idx]
            text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
        else:
            text_feat = None
            texts = None

        idx = np.random.randint(data['image_feat'].shape[0])
        img_feat = data["image_feat"][idx]
        
        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "text_feat": text_feat,
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name":  meta["id"],
            "texts": texts,
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            if self.split[index]['dataset'] == "Objaverse":
                return self.get_objaverse(self.split[index])
            else:
                return self.get_others(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist() 
            for i in knn_idx:
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                if self.split[idx]['dataset'] == "Objaverse":
                    data_list.append(self.get_objaverse(self.split[idx]))
                else:
                    data_list.append(self.get_others(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num

class FourTextNPZ(Four):
    """
    Minimal variant of Four that reads text features from your new <id>.tokfeat_text.npz:
      - text_global: (N, D) float16/32, unit-norm
      - text_type:   (N,)    object strings among {"original","blip","msft","retrieval"} (optional but preferred)
      - text_raw:    (N,)    object raw strings, same order (optional)

    Expected layout on disk (like embeddings.py):
      <text_npz_root>/<dataset>/<group>/<id>.tokfeat_text.npz
    For datasets without a group (e.g. 3D-FUTURE), it tries without group too.

    Config:
      set `config.dataset.text_tok_root = "/path/to/text_npz_root"`
    """

    def __init__(self, config, phase):
        super().__init__(config, phase)
        self.text_tok_root = getattr(config.dataset, "text_tok_root", None)
        assert self.text_tok_root, "Please set config.dataset.text_tok_root to your text NPZ root."

    # ---------- path + loading ----------

    def _textnpz_candidates(self, meta: dict):
        ds  = (meta.get("dataset") or "").strip()
        grp = (meta.get("group")   or "").strip()
        uid =  meta.get("id") or meta.get("uid") or ""
        if not (ds and uid):
            return []

        c = []
        if grp:
            c.append(os.path.join(self.text_tok_root, ds, grp, f"{uid}.tokfeat_text.npz"))
        c.append(os.path.join(self.text_tok_root, ds, f"{uid}.tokfeat_text.npz"))
        return c

    def _load_text_npz(self, meta, data):
        """
        Returns: (rows: np.ndarray or None, types: List[str] or None)
        rows has shape (N, D). If 'text_type' is missing, we fall back to fixed-order inference:
          original (0), optional blip (1), optional msft (2), then retrieval* (remaining)
        """
        rows, types = None, None
        for p in self._textnpz_candidates(meta):
            if not os.path.exists(p):
                continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    tg = z.get("text_global", None)
                    if tg is None:
                        continue
                    rows = np.asarray(tg, dtype=np.float32)
                    # Normalize shapes
                    if rows.ndim == 1:
                        rows = rows[None, :]
                    # Preferred: explicit types from file
                    tt = z.get("text_type", None)
                    if tt is not None:
                        types = [str(x) for x in np.asarray(tt, dtype=object).tolist()]
                    else:
                        # Fallback: infer types from original .npy pack presence (same order embeddings.py used)
                        types = []
                        # original
                        t = data.get("text")
                        if isinstance(t, (list, tuple)) and t and isinstance(t[0], str) and t[0]:
                            types.append("original")
                        # blip
                        t = data.get("blip_caption")
                        if isinstance(t, str) and t:
                            types.append("blip")
                        # msft
                        t = data.get("msft_caption")
                        if isinstance(t, str) and t:
                            types.append("msft")
                        # retrieval (one per entry)
                        r = data.get("retrieval_text")
                        if isinstance(r, (list, tuple)):
                            for _ in r:
                                types.append("retrieval")
                    return rows, types
            except Exception:
                pass  # try next candidate
        return None, None

    def _pick_row(self, rows, types, wanted_tag, retrieval_idx=None):
        """
        Pick a row from rows/types by tag. For 'retrieval', use retrieval_idx if provided.
        Returns np.ndarray 1D or None.
        """
        if rows is None or types is None:
            return None
        if wanted_tag == "retrieval":
            ridxs = [i for i, t in enumerate(types) if t == "retrieval"]
            if not ridxs:
                return None
            # map retrieval_idx into ridxs
            if retrieval_idx is None:
                k = np.random.randint(len(ridxs))
                return rows[ridxs[k]].reshape(-1)
            if retrieval_idx < len(ridxs):
                return rows[ridxs[retrieval_idx]].reshape(-1)
            # if retrieval_idx is out of range (shouldn't happen), clamp/random
            k = min(retrieval_idx, len(ridxs) - 1)
            return rows[ridxs[k]].reshape(-1)
        else:
            # first occurrence of wanted_tag
            try:
                i = types.index(wanted_tag)
                return rows[i].reshape(-1)
            except ValueError:
                return None

    # ---------- overrides: only the text feature selection changes ----------

    def get_objaverse(self, meta):
        # ----- same prelude as Four.get_objaverse -----
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        uid = meta["id"]
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        # ----- NEW: load NPZ rows/types -----
        rows, types = self._load_text_npz(meta, data)

        text_feat_candidates, text_strings = [], []

        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                v = self._pick_row(rows, types, "original")
                if v is not None and isinstance(data.get("text"), (list, tuple)) and len(data["text"]) > 0:
                    text_feat_candidates.append(v)
                    text_strings.append(data["text"][0])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data.get("blip_caption", "")) > 0:
                    v = self._pick_row(rows, types, "blip")
                    if v is not None:
                        text_feat_candidates.append(v)
                        text_strings.append(data["blip_caption"])
            else:
                if len(data.get("msft_caption", "")) > 0:
                    v = self._pick_row(rows, types, "msft")
                    if v is not None:
                        text_feat_candidates.append(v)
                        text_strings.append(data["msft_caption"])

        if 'retrieval_text' in self.text_source:
            if len(data.get("retrieval_text", [])) > 0:
                ridx = np.random.randint(len(data["retrieval_text"]))
                v = self._pick_row(rows, types, "retrieval", retrieval_idx=ridx)
                if v is not None:
                    text_feat_candidates.append(v)
                    text_strings.append(data["retrieval_text"][ridx])

        if text_feat_candidates:
            k = np.random.randint(len(text_feat_candidates))
            text_feat = torch.from_numpy(np.asarray(text_feat_candidates[k], dtype=np.float32)).float().reshape(-1)
            text_str  = text_strings[k]
        else:
            text_feat, text_str = None, None

        # image feature stays identical to Four
        if np.random.rand() < 0.5:
            img_feat = data['thumbnail_feat']
        else:
            idx = np.random.randint(data['image_feat'].shape[0])
            img_feat = data["image_feat"][idx]

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "text_feat": text_feat,
            "dataset": "Objaverse",
            "group": meta["group"],
            "name":  uid,
            "texts": text_str,
            "has_text": text_feat is not None,
        }

    def get_others(self, meta):
        # ----- same prelude as Four.get_others -----
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        # ----- NEW: load NPZ rows/types -----
        rows, types = self._load_text_npz(meta, data)

        text_feat_candidates, text_strings = [], []

        if 'text' in self.text_source:
            v = self._pick_row(rows, types, "original")
            if v is not None and isinstance(data.get("text"), (list, tuple)) and len(data["text"]) > 0:
                # for non-Objaverse, Four randomly picks an index in data["text"]; but your NPZ has
                # a single 'original' pooled text (first). We keep behavior: use that single row.
                text_feat_candidates.append(v)
                text_strings.append(data["text"][0])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data.get("blip_caption", "")) > 0:
                    v = self._pick_row(rows, types, "blip")
                    if v is not None:
                        text_feat_candidates.append(v)
                        text_strings.append(data["blip_caption"])
            else:
                if len(data.get("msft_caption", "")) > 0:
                    v = self._pick_row(rows, types, "msft")
                    if v is not None:
                        text_feat_candidates.append(v)
                        text_strings.append(data["msft_caption"])

        if 'retrieval_text' in self.text_source:
            if len(data.get("retrieval_text", [])) > 0:
                ridx = np.random.randint(len(data["retrieval_text"]))
                v = self._pick_row(rows, types, "retrieval", retrieval_idx=ridx)
                if v is not None:
                    text_feat_candidates.append(v)
                    text_strings.append(data["retrieval_text"][ridx])

        if text_feat_candidates:
            k = np.random.randint(len(text_feat_candidates))
            text_feat = torch.from_numpy(np.asarray(text_feat_candidates[k], dtype=np.float32)).float().reshape(-1)
            text_str  = text_strings[k]
        else:
            text_feat, text_str = None, None

        # image feature identical to Four
        idx = np.random.randint(data['image_feat'].shape[0])
        img_feat = data["image_feat"][idx]
        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "text_feat": text_feat,
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name":  meta["id"],
            "texts": text_str,
            "has_text": text_feat is not None,
        }




class FourNPZ(Four):
    """
    Minimal-change Four that loads TEXT and IMAGE features from your NPZ exports.

    Expected NPZ layouts (as written by embeddings.py):
      Text:  <text_tok_root>/<dataset>/<group>/<id>.tokfeat_text.npz
             - text_global: [N,D], text_type: [N] (optional), text_raw: [N] (optional)

      Image: <image_tok_root>/<dataset>/<group>/<id>.tokfeat_image.npz
             - image_global: [V,D], view_files: [V] (object)

    Config fields required:
      config.dataset.text_tok_root  = "/path/to/text_npz_root"
      config.dataset.image_tok_root = "/path/to/image_npz_root"
    """

    # --------------------- init ---------------------
    def __init__(self, config, phase):
        super().__init__(config, phase)
        self.text_tok_root  = getattr(config.dataset, "text_tok_root", None)
        self.image_tok_root = getattr(config.dataset, "image_tok_root", None)
        assert self.text_tok_root,  "Set config.dataset.text_tok_root to your text NPZ root."
        assert self.image_tok_root, "Set config.dataset.image_tok_root to your image NPZ root."

    # --------------------- helpers: path candidates ---------------------
    def _npz_candidates(self, root: str, meta: dict, kind: str):
        """kind in {'text','image'}; returns ordered candidate npz paths (with and without group)."""
        ds  = (meta.get("dataset") or "").strip()
        grp = (meta.get("group")   or "").strip()
        uid =  meta.get("id") or meta.get("uid") or ""
        if not (ds and uid): return []
        name = f"{uid}.tokfeat_{kind}.npz"
        c = []
        if grp:
            c.append(os.path.join(root, ds, grp, name))
        c.append(os.path.join(root, ds, name))
        return c

    # --------------------- TEXT NPZ ---------------------
    def _load_text_npz(self, meta, data):
        """
        Returns: rows (np.float32 [N,D]) and types (List[str]) or (None, None) if not found.
        """
        for p in self._npz_candidates(self.text_tok_root, meta, "text"):
            if not os.path.exists(p): continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    tg = z.get("text_global", None)
                    if tg is None: continue
                    rows = np.asarray(tg, dtype=np.float32)
                    rows = rows if rows.ndim == 2 else rows.reshape(1, -1)
                    tt = z.get("text_type", None)
                    if tt is not None:
                        types = [str(x) for x in np.asarray(tt, dtype=object).tolist()]
                    else:
                        # Infer order exactly like we saved: original -> blip -> msft -> retrieval[*]
                        types = []
                        t = data.get("text")
                        if isinstance(t, (list, tuple)) and t and isinstance(t[0], str) and t[0]:
                            types.append("original")
                        t = data.get("blip_caption")
                        if isinstance(t, str) and t:
                            types.append("blip")
                        t = data.get("msft_caption")
                        if isinstance(t, str) and t:
                            types.append("msft")
                        r = data.get("retrieval_text")
                        if isinstance(r, (list, tuple)):
                            for _ in r: types.append("retrieval")
                    return rows, types
            except Exception:
                continue
        return None, None

    def _pick_text_row(self, rows, types, wanted_tag, retrieval_idx=None):
        if rows is None or types is None: return None
        if wanted_tag == "retrieval":
            ridxs = [i for i, t in enumerate(types) if t == "retrieval"]
            if not ridxs: return None
            if retrieval_idx is None:
                return rows[random.choice(ridxs)].reshape(-1)
            if retrieval_idx < len(ridxs):
                return rows[ridxs[retrieval_idx]].reshape(-1)
            return rows[ridxs[-1]].reshape(-1)
        else:
            try:
                i = types.index(wanted_tag)
                return rows[i].reshape(-1)
            except ValueError:
                return None

    # --------------------- IMAGE NPZ ---------------------
    def _load_image_npz(self, meta):
        """
        Returns: (globals [V,D] float32, view_files [V] list[str]) or (None, None) if not found.
        """
        for p in self._npz_candidates(self.image_tok_root, meta, "image"):
            if not os.path.exists(p): continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    ig = z.get("image_global", None)
                    if ig is None: continue
                    globals_ = np.asarray(ig, dtype=np.float32)
                    globals_ = globals_ if globals_.ndim == 2 else globals_.reshape(1, -1)
                    vf = z.get("view_files", None)
                    view_files = [str(x) for x in (vf.tolist() if vf is not None else [f"colors_{i}.png" for i in range(globals_.shape[0])])]
                    return globals_, view_files
            except Exception:
                continue
        return None, None

    def _pick_image_global(self, meta, globals_, view_files, dataset_name: str):
        """
        Mimic Four's behavior:
          - Objaverse: 50% 'thumbnail' (prefer colors_0.*) else random view.
          - Others:    random view.
        Returns 1D np.array or None.
        """
        if globals_ is None or globals_.size == 0: return None

        def find_colors0(vf):
            for i, p in enumerate(vf):
                name = os.path.basename(str(p))
                # match colors_0.{png|jpg|...}
                if name.startswith("colors_0."):
                    return i
            return 0  # fallback

        if dataset_name == "Objaverse":
            if np.random.rand() < 0.5:
                i = find_colors0(view_files)
            else:
                i = np.random.randint(globals_.shape[0])
        else:
            i = np.random.randint(globals_.shape[0])

        return globals_[i].reshape(-1)

    # --------------------- overrides ---------------------
    def get_objaverse(self, meta):
        # ------- same point cloud prelude as Four -------
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        uid = meta["id"]
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        # ------- TEXT from NPZ -------
        rows, types = self._load_text_npz(meta, data)
        text_feat_candidates, text_strings = [], []

        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                v = self._pick_text_row(rows, types, "original")
                if v is not None and isinstance(data.get("text"), (list, tuple)) and len(data["text"]) > 0:
                    text_feat_candidates.append(v); text_strings.append(data["text"][0])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data.get("blip_caption", "")) > 0:
                    v = self._pick_text_row(rows, types, "blip")
                    if v is not None:
                        text_feat_candidates.append(v); text_strings.append(data["blip_caption"])
            else:
                if len(data.get("msft_caption", "")) > 0:
                    v = self._pick_text_row(rows, types, "msft")
                    if v is not None:
                        text_feat_candidates.append(v); text_strings.append(data["msft_caption"])

        if 'retrieval_text' in self.text_source:
            if len(data.get("retrieval_text", [])) > 0:
                ridx = np.random.randint(len(data["retrieval_text"]))
                v = self._pick_text_row(rows, types, "retrieval", retrieval_idx=ridx)
                if v is not None:
                    text_feat_candidates.append(v); text_strings.append(data["retrieval_text"][ridx])

        if text_feat_candidates:
            k = np.random.randint(len(text_feat_candidates))
            text_feat = torch.from_numpy(np.asarray(text_feat_candidates[k], dtype=np.float32)).float().reshape(-1)
            text_str  = text_strings[k]
        else:
            text_feat, text_str = None, None

        # ------- IMAGE from NPZ (with fallback to pack) -------
        ig, vf = self._load_image_npz(meta)
        img_vec = self._pick_image_global(meta, ig, vf, "Objaverse")
        if img_vec is None:
            # fallback (exact Four behavior)
            if np.random.rand() < 0.5:
                img_vec = data['thumbnail_feat']
            else:
                ridx = np.random.randint(data['image_feat'].shape[0])
                img_vec = data['image_feat'][ridx]
        img_feat = torch.from_numpy(np.asarray(img_vec, dtype=np.float32)).float().reshape(-1)

        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": img_feat,
            "text_feat": text_feat,
            "dataset": "Objaverse",
            "group": meta["group"],
            "name":  uid,
            "texts": text_str,
            "has_text": text_feat is not None,
        }

    def get_others(self, meta):
        # ------- same point cloud prelude as Four -------
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        data = np.load(file_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        # ------- TEXT from NPZ -------
        rows, types = self._load_text_npz(meta, data)
        text_feat_candidates, text_strings = [], []

        if 'text' in self.text_source:
            v = self._pick_text_row(rows, types, "original")
            if v is not None and isinstance(data.get("text"), (list, tuple)) and len(data["text"]) > 0:
                text_feat_candidates.append(v); text_strings.append(data["text"][0])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data.get("blip_caption", "")) > 0:
                    v = self._pick_text_row(rows, types, "blip")
                    if v is not None:
                        text_feat_candidates.append(v); text_strings.append(data["blip_caption"])
            else:
                if len(data.get("msft_caption", "")) > 0:
                    v = self._pick_text_row(rows, types, "msft")
                    if v is not None:
                        text_feat_candidates.append(v); text_strings.append(data["msft_caption"])

        if 'retrieval_text' in self.text_source:
            if len(data.get("retrieval_text", [])) > 0:
                ridx = np.random.randint(len(data["retrieval_text"]))
                v = self._pick_text_row(rows, types, "retrieval", retrieval_idx=ridx)
                if v is not None:
                    text_feat_candidates.append(v); text_strings.append(data["retrieval_text"][ridx])

        if text_feat_candidates:
            k = np.random.randint(len(text_feat_candidates))
            text_feat = torch.from_numpy(np.asarray(text_feat_candidates[k], dtype=np.float32)).float().reshape(-1)
            text_str  = text_strings[k]
        else:
            text_feat, text_str = None, None

        # ------- IMAGE from NPZ (random), fallback to pack random ------
        ig, vf = self._load_image_npz(meta)
        img_vec = None
        if ig is not None and ig.size > 0:
            i = np.random.randint(ig.shape[0])
            img_vec = ig[i].reshape(-1)
        if img_vec is None:
            ridx = np.random.randint(data['image_feat'].shape[0])
            img_vec = data['image_feat'][ridx]
        img_feat = torch.from_numpy(np.asarray(img_vec, dtype=np.float32)).float().reshape(-1)

        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": img_feat,
            "text_feat": text_feat,
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name":  meta["id"],
            "texts": text_str,
            "has_text": text_feat is not None,
        }
    


# --- UPDATED FourNPZ: now also returns text_token_feat & text_token_mask ---

class FourNPZ_text_tokens(Four):
    """
    Loads TEXT and IMAGE features from your NPZ exports.
    Adds per-token text features when available: text_token_feat [Tmax,D], text_token_mask [Tmax].
    """

    def __init__(self, config, phase):
        super().__init__(config, phase)
        self.text_tok_root  = getattr(config.dataset, "text_tok_root", None)
        self.image_tok_root = getattr(config.dataset, "image_tok_root", None)
        assert self.text_tok_root,  "Set config.dataset.text_tok_root to your text NPZ root."
        assert self.image_tok_root, "Set config.dataset.image_tok_root to your image NPZ root."

    # ---------- helpers ----------
    def _npz_candidates(self, root: str, meta: dict, kind: str):
        ds  = (meta.get("dataset") or "").strip()
        grp = (meta.get("group")   or "").strip()
        uid =  meta.get("id") or meta.get("uid") or ""
        if not (ds and uid): return []
        name = f"{uid}.tokfeat_{kind}.npz"
        c = []
        if grp: c.append(os.path.join(root, ds, grp, name))
        c.append(os.path.join(root, ds, name))
        return c

    # ---------- TEXT NPZ (now returns globals + types + tokens + mask) ----------
    def _load_text_npz(self, meta, data):
        """
        Returns:
          rows  : (N,D) float32 text_global
          types : list[str] labels for rows
          toks  : (N,Tmax,D) float32 or None
          mask  : (N,Tmax)   bool or None
        or (None, None, None, None) if no npz found.
        """
        for p in self._npz_candidates(self.text_tok_root, meta, "text"):
            if not os.path.exists(p): continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    tg = z.get("text_global", None)
                    if tg is None: continue
                    rows = np.asarray(tg, dtype=np.float32)
                    rows = rows if rows.ndim == 2 else rows.reshape(1, -1)

                    # optional tokens/mask
                    toks = z.get("text_tokens", None)
                    toks = (np.asarray(toks, dtype=np.float32)
                            if toks is not None else None)
                    mask = z.get("text_mask", None)
                    mask = (np.asarray(mask, dtype=bool)
                            if mask is not None else None)

                    # labels
                    tt = z.get("text_type", None)
                    if tt is not None:
                        types = [str(x) for x in np.asarray(tt, dtype=object).tolist()]
                    else:
                        # infer the order we used when saving
                        types = []
                        t = data.get("text")
                        if isinstance(t, (list, tuple)) and t and isinstance(t[0], str) and t[0]:
                            types.append("original")
                        t = data.get("blip_caption")
                        if isinstance(t, str) and t:
                            types.append("blip")
                        t = data.get("msft_caption")
                        if isinstance(t, str) and t:
                            types.append("msft")
                        r = data.get("retrieval_text")
                        if isinstance(r, (list, tuple)):
                            for _ in r: types.append("retrieval")

                    return rows, types, toks, mask
            except Exception:
                continue
        return None, None, None, None

    def _choose_idx(self, types, wanted_tag, retrieval_idx=None):
        """Return index into types for the wanted tag, handling retrieval sets."""
        if types is None: return None
        if wanted_tag == "retrieval":
            ridxs = [i for i, t in enumerate(types) if t == "retrieval"]
            if not ridxs: return None
            if retrieval_idx is None:
                return random.choice(ridxs)
            if retrieval_idx < len(ridxs):
                return ridxs[retrieval_idx]
            return ridxs[-1]
        else:
            try:
                return types.index(wanted_tag)
            except ValueError:
                return None

    # ---------- IMAGE NPZ (unchanged) ----------
    def _load_image_npz(self, meta):
        for p in self._npz_candidates(self.image_tok_root, meta, "image"):
            if not os.path.exists(p): continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    ig = z.get("image_global", None)
                    if ig is None: continue
                    globals_ = np.asarray(ig, dtype=np.float32)
                    globals_ = globals_ if globals_.ndim == 2 else globals_.reshape(1, -1)
                    vf = z.get("view_files", None)
                    view_files = [str(x) for x in (vf.tolist() if vf is not None else [f"colors_{i}.png" for i in range(globals_.shape[0])])]
                    return globals_, view_files
            except Exception:
                continue
        return None, None

    def _pick_image_global(self, meta, globals_, view_files, dataset_name: str):
        if globals_ is None or globals_.size == 0: return None
        def find_colors0(vf):
            for i, p in enumerate(vf):
                if os.path.basename(str(p)).startswith("colors_0."):
                    return i
            return 0
        if dataset_name == "Objaverse":
            i = find_colors0(view_files) if (np.random.rand() < 0.5) else np.random.randint(globals_.shape[0])
        else:
            i = np.random.randint(globals_.shape[0])
        return globals_[i].reshape(-1)

    # ---------- overrides ----------
    def _build_return(self, meta, data, dataset_name: str):
        # point cloud prelude (same as Four)
        file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        ndata = np.load(file_path, allow_pickle=True).item()
        n = ndata['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = ndata['xyz'][idx]
        rgb = ndata['rgb'][idx]

        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        # TEXT globals + tokens from NPZ
        rows, types, toks, mask = self._load_text_npz(meta, ndata)
        text_feat_candidates, text_tok_candidates, text_msk_candidates, text_strings = [], [], [], []

        uid = meta.get("id", "")
        # 'text'
        if 'text' in self.text_source:
            if dataset_name != "Objaverse" or not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                j = self._choose_idx(types, "original")
                if j is not None and isinstance(ndata.get("text"), (list, tuple)) and len(ndata["text"]) > 0:
                    text_feat_candidates.append(rows[j])
                    text_tok_candidates.append(None if toks is None else toks[j])
                    text_msk_candidates.append(None if mask is None else mask[j])
                    text_strings.append(ndata["text"][0])

        # 'caption'
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(ndata.get("blip_caption", "")) > 0:
                    j = self._choose_idx(types, "blip")
                    if j is not None:
                        text_feat_candidates.append(rows[j])
                        text_tok_candidates.append(None if toks is None else toks[j])
                        text_msk_candidates.append(None if mask is None else mask[j])
                        text_strings.append(ndata["blip_caption"])
            else:
                if len(ndata.get("msft_caption", "")) > 0:
                    j = self._choose_idx(types, "msft")
                    if j is not None:
                        text_feat_candidates.append(rows[j])
                        text_tok_candidates.append(None if toks is None else toks[j])
                        text_msk_candidates.append(None if mask is None else mask[j])
                        text_strings.append(ndata["msft_caption"])

        # 'retrieval_text'
        if 'retrieval_text' in self.text_source:
            if len(ndata.get("retrieval_text", [])) > 0:
                ridx = np.random.randint(len(ndata["retrieval_text"]))
                j = self._choose_idx(types, "retrieval", retrieval_idx=ridx)
                if j is not None:
                    text_feat_candidates.append(rows[j])
                    text_tok_candidates.append(None if toks is None else toks[j])
                    text_msk_candidates.append(None if mask is None else mask[j])
                    text_strings.append(ndata["retrieval_text"][ridx])

        if text_feat_candidates:
            k = np.random.randint(len(text_feat_candidates))
            text_feat = torch.from_numpy(np.asarray(text_feat_candidates[k], dtype=np.float32)).float().reshape(-1)
            tok_feat  = (None if text_tok_candidates[k] is None
                         else torch.from_numpy(np.asarray(text_tok_candidates[k], dtype=np.float32)).float())
            tok_mask  = (None if text_msk_candidates[k] is None
                         else torch.from_numpy(np.asarray(text_msk_candidates[k], dtype=bool)))
            text_str  = text_strings[k]
        else:
            text_feat, tok_feat, tok_mask, text_str = None, None, None, None

        # IMAGE global from NPZ (with pack fallback)
        ig, vf = self._load_image_npz(meta)
        img_vec = self._pick_image_global(meta, ig, vf, dataset_name)
        if img_vec is None:
            # fallback to original pack behavior
            if dataset_name == "Objaverse":
                if np.random.rand() < 0.5:
                    img_vec = ndata['thumbnail_feat']
                else:
                    ridx = np.random.randint(ndata['image_feat'].shape[0])
                    img_vec = ndata['image_feat'][ridx]
            else:
                ridx = np.random.randint(ndata['image_feat'].shape[0])
                img_vec = ndata['image_feat'][ridx]
        img_feat = torch.from_numpy(np.asarray(img_vec, dtype=np.float32)).float().reshape(-1)

        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": img_feat,
            "text_feat": text_feat,                    # (D,)
            "text_token_feat": tok_feat,               # (Tmax,D) or None
            "text_token_mask": tok_mask,               # (Tmax,)  or None
            "dataset": dataset_name if dataset_name=="Objaverse" else meta["dataset"],
            "group": meta["group"],
            "name":  meta["id"],
            "texts": text_str,
            "has_text": text_feat is not None,
        }

    def get_objaverse(self, meta):
        return self._build_return(meta, None, "Objaverse")

    def get_others(self, meta):
        return self._build_return(meta, None, "Others")



# class Four(Dataset):
#     def __init__(self, config, phase):
#         self.phase = phase
#         if phase == "train":
#             self.split = json.load(open(config.dataset.train_split, "r"))
#             if config.dataset.train_partial > 0:
#                 self.split = self.split[:config.dataset.train_partial]
#         else:
#             raise NotImplementedError("Phase %s not supported." % phase)
#         self.pc_root = '/mnt/disk1/souhail_data/objaverse-processed/merged_for_training_final'
#         self.phase = phase
#         self.y_up = config.dataset.y_up
#         self.random_z_rotate = config.dataset.random_z_rotate
#         self.num_points = config.dataset.num_points
#         self.use_color = config.dataset.use_color
#         self.normalize = config.dataset.normalize
#         self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
#         self.text_source = config.dataset.text_source
#         self.augment = config.dataset.augment
#         self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
#         self.use_text_filtering = config.dataset.use_text_filtering
#         self.use_prompt_engineering = config.dataset.use_prompt_engineering
#         self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
#         if self.use_knn_negative_sample:
#             self.negative_sample_num = config.dataset.negative_sample_num
#             self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
#             self.uid_to_index = {}
#             for i, item in enumerate(self.split):
#                 self.uid_to_index[item['id']] = i
#         if self.use_text_filtering:
#             self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
#         logging.info("Phase %s: %d samples" % (phase, len(self.split)))


#     def _npz_caption_vec(self, meta, data, tag: str):
#         """
#         Return the NPZ text_global vector for a caption tag ('blip' or 'msft'),
#         mapping the NPZ row based on which captions actually exist in the .npy pack.

#         Mapping:
#         - both blip & msft exist -> BLIP:1, MSFT:2  (need >=3 rows)
#         - only blip exists       -> BLIP:1         (need >=2 rows)
#         - only msft exists       -> MSFT:1         (need >=2 rows)
#         - otherwise              -> None (no caption candidate)
#         """
#         import os
#         tokroot = getattr(self, "tokfeat_root", None)
#         if tokroot is None and hasattr(self, "config") and hasattr(self.config, "dataset"):
#             tokroot = getattr(self.config.dataset, "tokfeat_root", None)
#         if tokroot is None:
#             return None

#         ds  = (meta.get("dataset") or "").strip()
#         grp = (meta.get("group")   or "").strip()
#         uid =  meta.get("id") or meta.get("uid")
#         if not (ds and uid):
#             return None

#         # NPZ candidate paths (with and without group)
#         candidates = []
#         if grp:
#             candidates.append(os.path.join(tokroot, ds, grp, f"{uid}.tokfeat.npz"))
#             candidates.append(os.path.join(tokroot, ds, grp, f"{uid}.npy.tokfeat.npz"))
#         candidates.append(os.path.join(tokroot, ds, f"{uid}.tokfeat.npz"))
#         candidates.append(os.path.join(tokroot, ds, f"{uid}.npy.tokfeat.npz"))

#         tg = None
#         for p in candidates:
#             if os.path.exists(p):
#                 with np.load(p) as z:
#                     if "text_global" in z:
#                         arr = np.asarray(z["text_global"])
#                         if arr.ndim == 2 and arr.shape[1] > 0:
#                             tg = arr
#                             break
#         if tg is None:
#             return None

#         has_blip = ("blip_caption" in data) and (len(data["blip_caption"]) > 0)
#         has_msft = ("msft_caption" in data) and (len(data["msft_caption"]) > 0)

#         idx = None
#         if has_blip and has_msft:
#             if tg.shape[0] >= 3:
#                 idx = 1 if tag == "blip" else 2
#         elif has_blip and not has_msft:
#             if tg.shape[0] >= 2 and tag == "blip":
#                 idx = 1
#         elif has_msft and not has_blip:
#             if tg.shape[0] >= 2 and tag == "msft":
#                 idx = 1
#         else:
#             return None

#         if idx is None:
#             return None
#         return tg[idx].astype(np.float32).reshape(-1)



#     def _npz_text_global(self, meta, idx: int):
#         import os
#         tokroot = None
#         # prefer an existing attribute; otherwise look on config.dataset (keeps variables local to Four)
#         if hasattr(self, "tokfeat_root"):
#             tokroot = self.tokfeat_root
#         elif hasattr(self, "config") and hasattr(self.config, "dataset"):
#             tokroot = getattr(self.config.dataset, "tokfeat_root", None)
#         assert tokroot is not None, "dataset.tokfeat_root must be set"

#         ds  = (meta.get("dataset") or "").strip()
#         grp = (meta.get("group")   or "").strip()
#         uid =  meta.get("id") or meta.get("uid")
#         assert ds and uid, f"Missing dataset/id in meta: {meta}"

#         # try with group, then without (matches how your NPZs are laid out elsewhere)
#         candidates = []
#         if grp:
#             candidates.append(os.path.join(tokroot, ds, grp, f"{uid}.tokfeat.npz"))
#             candidates.append(os.path.join(tokroot, ds, grp, f"{uid}.npy.tokfeat.npz"))
#         candidates.append(os.path.join(tokroot, ds, f"{uid}.tokfeat.npz"))
#         candidates.append(os.path.join(tokroot, ds, f"{uid}.npy.tokfeat.npz"))

#         for p in candidates:
#             if os.path.exists(p):
#                 z = np.load(p)
#                 tg = z["text_global"]             # must exist
#                 tg = np.asarray(tg)
#                 vec = tg[idx] if tg.ndim == 2 else tg
#                 return np.asarray(vec, dtype=np.float32).reshape(-1)

#         # If you ever reach here, your NPZ gate didn’t skip this sample.
#         raise FileNotFoundError(f"NPZ not found for {ds}/{grp}/{uid}")


#     def get_objaverse(self, meta):
#         file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
#         uid = meta["id"]
#         data = np.load(file_path, allow_pickle=True).item()
#         n = data['xyz'].shape[0]
#         idx = random.sample(range(n), self.num_points)
#         xyz = data['xyz'][idx]
#         rgb = data['rgb'][idx]
        
#         if self.y_up:
#             # swap y and z axis
#             xyz[:, [1, 2]] = xyz[:, [2, 1]]
#         if self.normalize:
#             xyz = normalize_pc(xyz)
#         if self.phase == "train" and self.augment:
#             xyz = augment_pc(xyz)
#         if self.phase == "train" and self.random_z_rotate:
#             xyz = random_rotate_z(xyz)
#         if self.use_color:
#             if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
#                 rgb = np.ones_like(rgb) * 0.4
#             features = np.concatenate([xyz, rgb], axis=1)
#         else:
#             features = xyz

#         text_feat = []
#         texts = []
#         if 'text' in self.text_source:
#             if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
#                 texts.append(data["text"][0])
#                 text_feat.append(data["text_feat"][0][self.text_embed_version])
        
#         if 'caption' in self.text_source:
#             if np.random.rand() < 0.5:
#                 # BLIP branch
#                 if "blip_caption" in data and len(data["blip_caption"]) > 0:
#                     vec = self._npz_caption_vec(meta, data, "blip")
#                     if vec is not None:
#                         texts.append(data["blip_caption"])
#                         text_feat.append(vec)
#             else:
#                 # MSFT branch
#                 if "msft_caption" in data and len(data["msft_caption"]) > 0:
#                     vec = self._npz_caption_vec(meta, data, "msft")
#                     if vec is not None:
#                         texts.append(data["msft_caption"])
#                         text_feat.append(vec)

        
#         if 'retrieval_text' in self.text_source:
#             if len(data["retrieval_text"]) > 0:
#                 idx = np.random.randint(len(data["retrieval_text"]))
#                 texts.append(data["retrieval_text"][idx])
#                 text_feat.append(data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text

#         if len(text_feat) > 0:
#             assert len(text_feat) == len(texts)
#             text_idx = np.random.randint(len(texts))
#             text_feat = text_feat[text_idx]
#             texts = texts[text_idx]
#             text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
#         else:
#             text_feat = None
#             texts = None

#         if np.random.rand() < 0.5:
#             img_feat = data['thumbnail_feat']
#         else:
#             idx = np.random.randint(data['image_feat'].shape[0])
#             img_feat = data["image_feat"][idx]

#         assert not np.isnan(xyz).any()

#         return {
#             "xyz": torch.from_numpy(xyz).type(torch.float32),
#             "features": torch.from_numpy(features).type(torch.float32),
#             "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
#             "text_feat": text_feat,
#             "dataset": "Objaverse",
#             "group": meta["group"],
#             "name":  uid,
#             "texts": texts,
#             "has_text": text_feat is not None,
#         }
    
#     def get_others(self, meta):
#         file_path = '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
#         data = np.load(file_path, allow_pickle=True).item()
#         n = data['xyz'].shape[0]
#         idx = random.sample(range(n), self.num_points)
#         xyz = data['xyz'][idx]
#         rgb = data['rgb'][idx]

#         if self.y_up:
#             # swap y and z axis
#             xyz[:, [1, 2]] = xyz[:, [2, 1]]
#         if self.normalize:
#             xyz = normalize_pc(xyz)
#         if self.phase == "train" and self.augment:
#             xyz = augment_pc(xyz)
#         if self.phase == "train" and self.random_z_rotate:
#             xyz = random_rotate_z(xyz)
#         if self.use_color:
#             if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
#                 rgb = np.ones_like(rgb) * 0.4
#             features = np.concatenate([xyz, rgb], axis=1)
#         else:
#             features = xyz

#         text_feat = []
#         texts = []
#         if 'text' in self.text_source:
#             idx = np.random.randint(len(data["text"]))
#             texts.append(data["text"][idx])
#             text_feat.append(data["text_feat"][idx][self.text_embed_version])
        
#         if 'caption' in self.text_source:
#             if np.random.rand() < 0.5:
#                 # BLIP branch
#                 if "blip_caption" in data and len(data["blip_caption"]) > 0:
#                     vec = self._npz_caption_vec(meta, data, "blip")
#                     if vec is not None:
#                         texts.append(data["blip_caption"])
#                         text_feat.append(vec)
#             else:
#                 # MSFT branch
#                 if "msft_caption" in data and len(data["msft_caption"]) > 0:
#                     vec = self._npz_caption_vec(meta, data, "msft")
#                     if vec is not None:
#                         texts.append(data["msft_caption"])
#                         text_feat.append(vec)

        
#         if 'retrieval_text' in self.text_source:
#             if len(data["retrieval_text"]) > 0:
#                 idx = np.random.randint(len(data["retrieval_text"]))
#                 texts.append(data["retrieval_text"][idx])
#                 text_feat.append(data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text

#         if len(text_feat) > 0:
#             assert len(text_feat) == len(texts)
#             text_idx = np.random.randint(len(texts))
#             text_feat = text_feat[text_idx]
#             texts = texts[text_idx]
#             text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
#         else:
#             text_feat = None
#             texts = None

#         idx = np.random.randint(data['image_feat'].shape[0])
#         img_feat = data["image_feat"][idx]
        
#         assert not np.isnan(xyz).any()

#         return {
#             "xyz": torch.from_numpy(xyz).type(torch.float32),
#             "features": torch.from_numpy(features).type(torch.float32),
#             "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
#             "text_feat": text_feat,
#             "dataset": meta["dataset"],
#             "group": meta["group"],
#             "name":  meta["id"],
#             "texts": texts,
#             "has_text": text_feat is not None,
#         }

#     def __getitem__(self, index: int):
#         if self.use_knn_negative_sample == False:
#             if self.split[index]['dataset'] == "Objaverse":
#                 return self.get_objaverse(self.split[index])
#             else:
#                 return self.get_others(self.split[index])
#         else:
#             data_list = []
#             # random select a seed shape from split
#             index = random.randint(0, len(self.split) - 1)
#             uid = self.split[index]['id']
#             # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
#             knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist() 
#             for i in knn_idx:
#                 idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
#                 if self.split[idx]['dataset'] == "Objaverse":
#                     data_list.append(self.get_objaverse(self.split[idx]))
#                 else:
#                     data_list.append(self.get_others(self.split[idx]))
#             return data_list

#     def __len__(self):
#         if self.use_knn_negative_sample == False:
#             return len(self.split)
#         else:
#             return len(self.split) // self.negative_sample_num

class FourWithNPZGate(Four):
    """
    Identical to Four, except: before delegating to Four.__getitem__, we ensure the NPZ
    exists, loads cleanly, and has enough rows in text_global based on which captions
    actually exist for that sample in the .npy pack.
    """

    def __init__(self, config, phase: str):
        super().__init__(config, phase)
        self.tokfeat_root       = getattr(config.dataset, "tokfeat_root", None)
        self.max_resample_tries = int(getattr(config.dataset, "max_resample_tries", 32))

    # --- utilities copied from earlier version so we don't touch Four ---
    def _meta_at(self, index: int):
        for attr in ("split", "train_split", "data_list", "data", "items"):
            if hasattr(self, attr):
                seq = getattr(self, attr)
                try:
                    return seq[index]
                except Exception:
                    pass
        return None

    def _npz_candidates(self, meta):
        if not self.tokfeat_root:
            return []
        ds  = (meta.get("dataset") or "").strip()
        grp = (meta.get("group")   or "").strip()
        uid =  meta.get("id") or meta.get("uid") or ""
        if not ds or not uid:
            return []
        c = []
        if grp:
            c.append(os.path.join(self.tokfeat_root, ds, grp, f"{uid}.tokfeat.npz"))
            c.append(os.path.join(self.tokfeat_root, ds, grp, f"{uid}.npy.tokfeat.npz"))
        c.append(os.path.join(self.tokfeat_root, ds, f"{uid}.tokfeat.npz"))
        c.append(os.path.join(self.tokfeat_root, ds, f"{uid}.npy.tokfeat.npz"))
        return c

    # --- NEW helpers for dynamic gating based on which captions exist in the .npy pack ---
    def _resolve_pack_path(self, meta):
        # match Four's rewrite logic for the .npy pack path
        p = meta.get('data_path', '')
        try:
            return '/mnt/disk1/souhail_data/' + p.split('data/')[1]
        except Exception:
            return p

    def _required_text_rows(self, meta) -> int:
        """
        Decide how many NPZ text_global rows are required:
          - both BLIP & MSFT present in .npy -> 3
          - only one present                -> 2
          - neither present                 -> 1
        If .npy pack can't be read, force 3 so we skip this sample.
        """
        pc_path = self._resolve_pack_path(meta)
        try:
            d = np.load(pc_path, allow_pickle=True).item()
            has_blip = ("blip_caption" in d) and (len(d["blip_caption"]) > 0)
            has_msft = ("msft_caption" in d) and (len(d["msft_caption"]) > 0)
        except Exception:
            return 3

        if has_blip and has_msft:
            return 3
        elif has_blip or has_msft:
            return 2
        else:
            return 1

    def _npz_has_rows(self, meta, min_rows: int) -> bool:
        """
        NPZ must (1) exist, (2) load cleanly, (3) have text_global with >= min_rows.
        We try all candidate paths; only log once if none are usable.
        """
        candidates = self._npz_candidates(meta)
        any_existing = False
        first_reason = None
        existing_count = 0

        for p in candidates:
            if not os.path.exists(p):
                continue
            any_existing = True
            existing_count += 1
            try:
                with np.load(p) as z:
                    tg = z.get("text_global", None)
                    if tg is None:
                        if first_reason is None:
                            first_reason = f"{p}: no 'text_global'"
                        continue
                    tg = np.asarray(tg)
                    _ = tg.shape  # force materialization to trip CRC/magic errors

                    if min_rows <= 1:
                        ok = (tg.ndim == 1) or (tg.ndim == 2 and tg.shape[0] >= 1)
                    else:
                        ok = (tg.ndim == 2 and tg.shape[0] >= min_rows)

                    if ok:
                        return True  # ✅ usable NPZ found; no log
                    if first_reason is None:
                        first_reason = f"{p}: insufficient rows {tg.shape}"
            except Exception as e:
                if first_reason is None:
                    first_reason = f"{p}: {type(e).__name__}: {e}"

        if any_existing and first_reason:
            logging.warning(
                f"[NPZSkip] No usable NPZ among {existing_count} candidate(s) "
                f"for ds={meta.get('dataset','?')} group={meta.get('group','?')} id={meta.get('id', meta.get('uid','?'))}. "
                f"Reason: {first_reason}"
            )
        return False

    # --- gate then delegate to Four ---
    def __getitem__(self, index: int):
        if not self.tokfeat_root:
            return super().__getitem__(index)

        tries = 0
        for attr in ("split", "train_split", "data_list", "data", "items"):
            if hasattr(self, attr):
                n = len(getattr(self, attr))
                break
        else:
            n = None

        while True:
            if n is None or tries >= self.max_resample_tries:
                break

            meta = self._meta_at(index)
            if meta is None:
                break

            # compute required rows from the .npy pack for this sample
            min_rows = self._required_text_rows(meta)

            if self._npz_has_rows(meta, min_rows):
                return super().__getitem__(index)  # exact Four behavior after gating

            tries += 1
            index = (index + 1) % n

        raise FileNotFoundError(
            f"NPZ missing/invalid for {self.max_resample_tries} successive items "
            f"(starting index {index})."
        )


def minkowski_collate_fn(list_data):
    if isinstance(list_data[0], list):
        merged_list = []
        for data in list_data:
            merged_list += data
        list_data = merged_list
    return {
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "img_feat": [data["img_feat"] for data in list_data],
        "text_feat": [data["text_feat"] for data in list_data if data["text_feat"] is not None],
        "dataset": [data["dataset"] for data in list_data],
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "texts": [data["texts"] for data in list_data],
        "has_text_idx": [i for i, data in enumerate(list_data) if data["text_feat"] is not None],
    }


class FourTokGlobal(torch.utils.data.Dataset):
    """
    Drop-in variant of Four that reads *your generated global features* (NPZ)
    while keeping the rest of OpenShape logic (same outputs & collate).

    Config (under `config.dataset`):
      - train_split: path to JSON split list (same as Four)
      - tokfeat_root: ROOT folder containing <dataset>/<group>/<id>.tokfeat.npz
      - num_points, y_up, use_color, normalize, augment, random_z_rotate,
        rgb_random_drop_prob, text_source, use_text_filtering, gpt4_filtering_path
      - (optional) pc_rewrite_from / pc_rewrite_to: rewrite prefix for data_path
    """
    def __init__(self, config, phase: str):
        self.use_prompt_avg_for_simple = bool(getattr(config.dataset, "use_prompt_avg_for_simple", True))
        self.retrieval_text_prob = float(getattr(config.dataset, "retrieval_text_prob", 0.25))
        assert phase == "train", f"Phase {phase} not supported in FourTokGlobal"
        self.skip_missing_tokfeat = bool(getattr(config.dataset, "skip_missing_tokfeat", True))
        self.max_miss_resamples  = int(getattr(config.dataset, "max_miss_resamples", 20))
        self.clip_dim = int(getattr(config.dataset, "clip_embed_dim", 1280))  # already present in your setup
        self.phase = phase
        self.split = json.load(open(config.dataset.train_split, "r"))
        if getattr(config.dataset, "train_partial", 0) > 0:
            self.split = self.split[:config.dataset.train_partial]

        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_text_filtering = config.dataset.use_text_filtering
        self.text_embed_version = "prompt_avg" if config.dataset.use_prompt_engineering else "original"
        self.tok_root = getattr(config.dataset, "tokfeat_root")
        assert self.tok_root, "config.dataset.tokfeat_root must be set for FourTokGlobal"

        # optional text filtering
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))

        # optional path rewrite
        self.pc_rewrite_from = getattr(config.dataset, "pc_rewrite_from", None)
        self.pc_rewrite_to   = getattr(config.dataset, "pc_rewrite_to", None)

        logging.info("Phase %s: %d samples (FourTokGlobal)" % (phase, len(self.split)))

    def _resolve_pc_path(self, meta):
        p = meta.get('data_path', '')
        if self.pc_rewrite_from and self.pc_rewrite_to and p.startswith(self.pc_rewrite_from):
            return self.pc_rewrite_to + p[len(self.pc_rewrite_from):]
        # fallback used in your Four: '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]
        if 'data/' in p:
            return '/mnt/disk1/souhail_data/' + p.split('data/')[1]
        return p

    def _load_xyz_rgb(self, meta):
        file_path = self._resolve_pc_path(meta)
        d = np.load(file_path, allow_pickle=True).item()
        n = d['xyz'].shape[0]
        idx = np.random.choice(n, self.num_points, replace=(n < self.num_points))
        xyz = d['xyz'][idx]
        rgb = d['rgb'][idx]
        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        return xyz, features

    # def _load_global_feats(self, meta):
    #     npz_path = _tokfeat_path_from_entry(meta, self.tok_root)
    #     text_feat = None
    #     img_feat = None
    #     if npz_path is not None and os.path.exists(npz_path):
    #         with np.load(npz_path) as pack:
    #             # image_global: (V, D) or (D,)
    #             img_g = pack.get('image_global')
    #             if img_g is not None:
    #                 img_g = np.asarray(img_g)
    #                 if img_g.ndim == 2 and img_g.shape[0] > 0:
    #                     v = np.random.randint(img_g.shape[0])
    #                     img_feat = img_g[v]
    #                 else:
    #                     img_feat = img_g.reshape(-1)
    #             # text_global: (T, D) or (D,)
    #             txt_g = pack.get('text_global')
    #             if txt_g is not None:
    #                 txt_g = np.asarray(txt_g)
    #                 if txt_g.ndim == 2 and txt_g.shape[0] > 0:
    #                     t = np.random.randint(txt_g.shape[0])
    #                     text_feat = txt_g[t]
    #                 elif txt_g.ndim == 1 and txt_g.shape[0] > 0:
    #                     text_feat = txt_g
    #     return img_feat, text_feat
    def _load_global_feats(self, meta):
        npz_path = _tokfeat_path_from_entry(meta, self.tok_root)
        text_feat = None
        img_feat = None

        # --- image_global (unchanged) ---
        if npz_path is not None and os.path.exists(npz_path):
            with np.load(npz_path) as pack:
                img_g = pack.get('image_global')
                if img_g is not None:
                    img_g = np.asarray(img_g)
                    if img_g.ndim == 2 and img_g.shape[0] > 0:
                        v = np.random.randint(img_g.shape[0])
                        img_feat = img_g[v]
                    else:
                        img_feat = img_g.reshape(-1)

        # --- NEW: sometimes use retrieval text from OpenShape .npy ---
        if np.random.rand() < self.retrieval_text_prob and text_feat is None:
            try:
                npy_path = self._resolve_pc_path(meta)  # same resolver you use for xyz/rgb
                d = np.load(npy_path, allow_pickle=True).item()
                if 'retrieval_text' in d and len(d['retrieval_text']) > 0 and 'retrieval_text_feat' in d:
                    i = np.random.randint(len(d['retrieval_text']))
                    tf = d['retrieval_text_feat'][i].get('original', None)  # retrieval stores only "original"
                    if tf is not None:
                        text_feat = np.asarray(tf, dtype=np.float32).reshape(-1)
            except Exception:
                pass  # fall back to NPZ text_global

        # --- NPZ text_global (unchanged + keep your slot-0 prompt-avg override) ---
        if text_feat is None and npz_path is not None and os.path.exists(npz_path):
            with np.load(npz_path) as pack:
                txt_g = pack.get('text_global')
                if txt_g is not None:
                    txt_g = np.asarray(txt_g)
                    if txt_g.ndim == 2 and txt_g.shape[0] > 0:
                        t = np.random.randint(txt_g.shape[0])
                        if t == 0 and getattr(self, "use_prompt_avg_for_simple", True):
                            try:
                                npy_path = self._resolve_pc_path(meta)
                                d = np.load(npy_path, allow_pickle=True).item()
                                if 'text' in d and 'text_feat' in d and len(d['text']) > 0:
                                    i = np.random.randint(len(d['text']))
                                    tf = d['text_feat'][i].get('prompt_avg', None) or d['text_feat'][i].get('original', None)
                                    if tf is not None:
                                        text_feat = np.asarray(tf, dtype=np.float32).reshape(-1)
                            except Exception:
                                text_feat = None
                        if text_feat is None:
                            text_feat = txt_g[t].reshape(-1)
                    elif txt_g.ndim == 1 and txt_g.shape[0] > 0:
                        text_feat = txt_g.reshape(-1)

        return img_feat, text_feat
    def __getitem__(self, index: int):
        tries = 0
        while True:
            meta = self.split[index]
            npz_path = _tokfeat_path_from_entry(meta, self.tok_root)
            if (not self.skip_missing_tokfeat) or (npz_path and os.path.exists(npz_path)):
                break
            if tries == 0:
                logging.warning(f"[TokfeatMissing] id={meta.get('id','?')} ds={meta.get('dataset','?')} npz={npz_path} — skipping")

            tries += 1
            if tries >= self.max_miss_resamples:
                # last resort: either raise or fall back to zeros in the right dim
                # raise FileNotFoundError(f"Missing tokfeat after {tries} tries …")
                return self.__getitem__((index + 1) % len(self.split))  # still moves on

            index = (index + 1) % len(self.split)


        xyz, features = self._load_xyz_rgb(meta)
        img_feat, text_feat = self._load_global_feats(meta)

        text_tensor = torch.from_numpy(text_feat).float().reshape(-1)
        # # If no text_global, we will mark as no-text; trainer will skip via has_text_idx
        # text_tensor = None if text_feat is None else torch.from_numpy(text_feat).float().reshape(-1)
        # # If no image_global (rare), fallback to zeros of text dim if available, else leave None -> will error;
        # # so ensure tensor present
        # # if img_feat is None:
        # #     if text_feat is not None:
        # #         img_feat = np.zeros_like(text_feat)
        # #     else:
        # #         # last resort: 1024-d zero (CLIP ViT-B/32 default dim). Adjust if needed.
        # #         img_feat = np.zeros((1024,), dtype=np.float32)
        img_tensor = torch.from_numpy(np.asarray(img_feat, dtype=np.float32)).reshape(-1)

        uid = meta.get("id") or meta.get("uid") or ""
        group = meta.get("group", "")
        dataset = meta.get("dataset", "")

        return {
            "xyz": torch.from_numpy(xyz).float(),
            "features": torch.from_numpy(features).float(),
            "img_feat": img_tensor,
            "text_feat": text_tensor,
            "dataset": dataset,
            "group": group,
            "name": uid,
            "texts": None,
            "has_text": text_tensor is not None,
        }

    def __len__(self):
        return len(self.split)


class FourSkipIfNPZMissing(Dataset):
    """
    Exactly like OpenShape 'Four' for data loading, EXCEPT:
    - Before returning a sample, we require that the corresponding NPZ tokfeat exists.
      If not found, we resample the next item (up to max_resample_tries), i.e. skip it.
    - Image/text features are loaded from the .npy pack using the same choices as OpenShape.

    Use this to isolate whether lower performance comes from NPZ coverage rather than feature values.
    """

    def __init__(self, config, phase: str):
        assert phase == "train", "This class is intended for training."
        self.phase = phase

        with open(config.dataset.train_split, "r") as f:
            self.split = json.load(f)
        if getattr(config.dataset, "train_partial", 0) > 0:
            self.split = self.split[:config.dataset.train_partial]

        # OpenShape-like knobs
        self.num_points         = int(config.dataset.num_points)
        self.y_up               = bool(config.dataset.y_up)
        self.normalize          = bool(config.dataset.normalize)
        self.augment            = bool(config.dataset.augment)
        self.random_z_rotate    = bool(config.dataset.random_z_rotate)
        self.use_color          = bool(config.dataset.use_color)
        self.rgb_random_drop    = float(getattr(config.dataset, "rgb_random_drop_prob", 0.5))
        self.text_sources       = list(getattr(config.dataset, "text_source", ["text","caption","retrieval_text"]))
        self.use_text_filtering = bool(getattr(config.dataset, "use_text_filtering", False))
        self.text_embed_version = "prompt_avg" if bool(getattr(config.dataset, "use_prompt_engineering", False)) else "original"

        # NPZ presence gate
        self.tokfeat_root       = getattr(config.dataset, "tokfeat_root", None)
        self.max_resample_tries = int(getattr(config.dataset, "max_resample_tries", 32))

        # Optional GPT-4 caption filter map (dataset/id -> 'Y'/'N')
        if self.use_text_filtering:
            with open(config.dataset.gpt4_filtering_path, "r") as f:
                self.gpt4_filtering = json.load(f)

        # Optional path rewrite for .npy PC pack
        self.pc_rewrite_from = getattr(config.dataset, "pc_rewrite_from", None)
        self.pc_rewrite_to   = getattr(config.dataset, "pc_rewrite_to",   None)

        # Expected CLIP dim (used only for sanity checks; not strict)
        self.clip_dim = int(getattr(config.dataset, "clip_feat_dim", 1280))

        logging.info(f"Phase {phase}: {len(self.split)} samples (FourSkipIfNPZMissing)")

    # ---------- path helpers ----------

    def _resolve_pc_path(self, meta):
        # identical to Four.get_objaverse/get_others behavior
        return '/mnt/disk1/souhail_data/' + meta['data_path'].split('data/')[1]

    def _tokfeat_path_from_entry(self, meta):
        """
        Resolve the expected NPZ path. We check both with/without 'group' to cover datasets
        that don't use a category folder (e.g., 3D-FUTURE).
        """
        if not self.tokfeat_root:
            return None

        ds  = (meta.get("dataset") or "").strip()
        grp = (meta.get("group")   or "").strip()
        uid =  meta.get("id") or meta.get("uid")
        if not ds or not uid:
            return None

        candidates = []
        if grp:
            candidates.append(os.path.join(self.tokfeat_root, ds, grp, f"{uid}.tokfeat.npz"))
            candidates.append(os.path.join(self.tokfeat_root, ds, grp, f"{uid}.npy.tokfeat.npz"))
        candidates.append(os.path.join(self.tokfeat_root, ds, f"{uid}.tokfeat.npz"))
        candidates.append(os.path.join(self.tokfeat_root, ds, f"{uid}.npy.tokfeat.npz"))

        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ---------- core helpers (OpenShape-like) ----------

    def _gpt4_allows(self, meta):
        # Match Four’s behavior: if there’s no entry, allow.
        if not self.use_text_filtering:
            return True

        uid = meta.get('id') or meta.get('uid') or ''
        ds  = meta.get('dataset', '') or ''
        # Support either schema: keyed by uid OR by "dataset/uid"
        filt = getattr(self, 'gpt4_filtering', None) or getattr(self, 'gpt4_filter', None)
        if not isinstance(filt, dict):
            return True

        entry = filt.get(uid) or filt.get(f"{ds}/{uid}")
        if not isinstance(entry, dict):
            # missing or malformed -> allow
            return True

        flag = entry.get('flag', 'Y')
        return str(flag).upper() != 'N'

    def _load_xyz_rgb(self, pc_path, dataset_hint=None):
        d = np.load(pc_path, allow_pickle=True).item()
        xyz = d["xyz"].astype(np.float32)
        rgb = d["rgb"].astype(np.float32) / 255.0

        n = xyz.shape[0]
        idx = random.sample(range(n), self.num_points)  # match Four
        xyz = xyz[idx]; rgb = rgb[idx]

        if self.y_up:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)

        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop:
                rgb = np.ones_like(rgb, dtype=np.float32) * 0.4
            feats = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            feats = xyz.astype(np.float32)

        return xyz.astype(np.float32), feats, d

    def _pick_image_feat(self, meta, d):
        """OpenShape behavior: Objaverse uses thumbnail 50% of the time, else a random view; others use random view."""
        ds = meta.get("dataset", "")

        if ds == "Objaverse" and "thumbnail_feat" in d and d["thumbnail_feat"] is not None:
            if np.random.rand() < 0.5:
                img = np.asarray(d["thumbnail_feat"], dtype=np.float32).reshape(-1)
            else:
                arr = np.asarray(d["image_feat"], dtype=np.float32)
                v = np.random.randint(arr.shape[0]) if arr.ndim == 2 else 0
                img = arr[v].reshape(-1)
        else:
            arr = np.asarray(d["image_feat"], dtype=np.float32)
            v = np.random.randint(arr.shape[0]) if arr.ndim == 2 else 0
            img = arr[v].reshape(-1)

        return torch.from_numpy(img).float()

    def _pick_text_feat(self, meta, d):
        """
        OpenShape-like: try a few random sources based on config.text_source.
        - 'text'           -> .npy text_feat[i][text_embed_version] (with optional GPT-4 filter)
        - 'caption'        -> prefer BLIP, else MSFT, both from .npy ..._caption_feat[text_embed_version]
        - 'retrieval_text' -> .npy retrieval_text_feat[i]['original']
        If nothing available, return None (trainer/collate handles no-text cases).
        """
        for _ in range(3):  # a few attempts to find an available source
            src = np.random.choice(self.text_sources)

            if src == "text":
                if not self._gpt4_allows(meta):
                    continue
                if ("text" in d) and ("text_feat" in d) and len(d["text"]) > 0:
                    i = 0 if meta.get("dataset") == "Objaverse" else np.random.randint(len(d["text"]))
                    tf = d["text_feat"][i].get(self.text_embed_version, None)
                    if tf is not None:
                        return torch.from_numpy(np.asarray(tf, dtype=np.float32).reshape(-1)).float()

            elif src == "caption":
                # try BLIP then MSFT (as in Four)
                if ("blip_caption" in d) and (d["blip_caption"] is not None) and ("blip_caption_feat" in d):
                    tf = d["blip_caption_feat"].get(self.text_embed_version, None)
                    if tf is not None:
                        return torch.from_numpy(np.asarray(tf, dtype=np.float32).reshape(-1)).float()
                if ("msft_caption" in d) and (d["msft_caption"] is not None) and ("msft_caption_feat" in d):
                    tf = d["msft_caption_feat"].get(self.text_embed_version, None)
                    if tf is not None:
                        return torch.from_numpy(np.asarray(tf, dtype=np.float32).reshape(-1)).float()

            elif src == "retrieval_text":
                if ("retrieval_text" in d) and ("retrieval_text_feat" in d) and len(d["retrieval_text"]) > 0:
                    i  = np.random.randint(len(d["retrieval_text"]))
                    tf = d["retrieval_text_feat"][i].get("original", None)
                    if tf is not None:
                        return torch.from_numpy(np.asarray(tf, dtype=np.float32).reshape(-1)).float()

        # no text candidate selected
        return None

    # ---------- Dataset API ----------

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index: int):
        # NPZ presence gate: skip items that don't have their NPZ, regardless of whether we'll use it.
        # tries = 0
        # while True:
        meta = self.split[index]

        pc_path = self._resolve_pc_path(meta)
        xyz, feats, d = self._load_xyz_rgb(pc_path, dataset_hint=meta.get("dataset", ""))

        img_feat  = self._pick_image_feat(meta, d)
        text_feat = self._pick_text_feat(meta, d)  # may be None (handled by your collate/Trainer)

        return {
            "xyz":       torch.from_numpy(xyz).float(),
            "features":  torch.from_numpy(feats).float(),
            "img_feat":  img_feat,
            "text_feat": text_feat,
            "dataset":   meta.get("dataset", ""),
            "group":     meta.get("group", ""),
            "name":      meta.get("id", meta.get("uid", "")),
            "texts":     None,
        }


def make(config, phase, rank, world_size):
    if config.dataset.name == "Four":
        dataset = Four(config, phase,)
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory = True,
            drop_last=True,
            sampler=sampler
        )
    elif config.dataset.name == "FourTokGlobal":
        dataset = FourTokGlobal(config, phase)
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
            data_loader = DataLoader(
                dataset,
                num_workers=config.dataset.num_workers,
                collate_fn=minkowski_collate_fn,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=True,
                sampler=sampler
            )
    elif config.dataset.name == "FourNPZ":
        dataset = FourNPZ(config, phase)
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
            data_loader = DataLoader(
                dataset,
                num_workers=config.dataset.num_workers,
                collate_fn=minkowski_collate_fn,
                batch_size=batch_size,
                pin_memory=True,
                drop_last=True,
                sampler=sampler
            )
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
    else:
        raise NotImplementedError("Dataset %s not supported." % config.dataset.name)
    return data_loader

class ModelNet40Test(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.modelnet40.test_split, "r"))
        self.pcs = np.load(config.modelnet40.test_pc, allow_pickle=True)
        self.num_points = config.modelnet40.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.modelnet40.y_up
        clip_feat = np.load(config.modelnet40.clip_feat_path, allow_pickle=True).item()
        self.categories = list(clip_feat.keys())
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

        logging.info("ModelNet40Test: %d samples" % len(self.split))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        pc = copy.deepcopy(self.pcs[index])
        n = pc['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = pc['xyz'][idx]
        rgb = pc['rgb'][idx]
        rgb = rgb / 255.0 # 100, scale to 0.4 to make it consistent with the training data
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        
        xyz = normalize_pc(xyz)

        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": self.split[index]["name"],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)

def minkowski_modelnet40_collate_fn(list_data):
    return {
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int64),
    }
def make_modelnet40test(config):
    dataset = ModelNet40Test(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.modelnet40.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.modelnet40.test_batch_size, \
        pin_memory = True, \
        shuffle=False
    )
    return data_loader
class Objaverse_no_LVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.dataset.train_split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        # self.categories = sorted(np.unique([data['category'] for data in self.split]))
        # self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        
        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        self.split[index]['data_path'] = '/mnt/disk1/souhail_data/objaverse-processed' + self.split[index]['data_path'].split('objaverse-processed')[1]
        data = np.load(self.split[index]['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
        }

    def __len__(self):
        return len(self.split)

class Objaverse_no_LVIS_id(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.dataset.train_split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        # self.categories = sorted(np.unique([data['category'] for data in self.split]))
        # self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        
        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        self.split[index]['data_path'] = '/mnt/disk1/souhail_data/objaverse-processed' + self.split[index]['data_path'].split('objaverse-processed')[1]
        data = np.load(self.split[index]['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "id": self.split[index]['id'],
        }

    def __len__(self):
        return len(self.split)

class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        
        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        self.split[index]['data_path'] = '/mnt/disk1/souhail_data/objaverse-processed' + self.split[index]['data_path'].split('objaverse-processed')[1]
        data = np.load(self.split[index]['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "group": self.split[index]['group'],
            "name":  self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)
    
def minkowski_objaverse_lvis_collate_fn(list_data):
    return {
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype = torch.int32),
    }
def minkowski_objaverse_no_lvis_collate_fn(list_data):
    return {
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data])
    }

def make_objaverse_no_lvis(config):
    return DataLoader(
        Objaverse_no_LVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=minkowski_objaverse_no_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory = True, \
        shuffle=False 
    )
def make_objaverse_no_lvis_no_collate(config):
    return DataLoader(
        Objaverse_no_LVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory = True, \
        shuffle=True 
    )
def make_objaverse_lvis(config):
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=minkowski_objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory = True, \
        shuffle=False 
    )

class ScanObjectNNTest(Dataset):
    def __init__(self, config):
        self.data = np.load(config.scanobjectnn.data_path, allow_pickle=True).item()
        self.num_points = config.scanobjectnn.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.scanobjectnn.y_up
        clip_feat = np.load(config.scanobjectnn.clip_feat_path, allow_pickle=True).item()
        self.categories = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"]
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
        logging.info("ScanObjectNNTest: %d samples" % self.__len__())
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        xyz = copy.deepcopy(self.data['xyz'][index])
        if 'rgb' not in self.data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.data['rgb'][index]
        label = self.data['label'][index]
        n = xyz.shape[0]
        if n != self.num_points:
            idx = np.random.choice(n, self.num_points) #random.sample(range(n), self.num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        
        xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": str(index),
            "category": label,
        }

    def __len__(self):
        return len(self.data['xyz'])
    
def make_scanobjectnntest(config):
    dataset = ScanObjectNNTest(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.scanobjectnn.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.scanobjectnn.test_batch_size, \
        pin_memory = True, \
        shuffle=False
    )
    return data_loader
