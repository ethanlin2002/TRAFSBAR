import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor


"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)

        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.dataset
        self.seq_len = args.seq_len
        self.split = "train"
        self.tensor_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                                         std=[0.225, 0.225, 0.225])])
        self.img_size = args.img_size

        # ===== traj settings =====
        self.use_traj = getattr(args, "use_traj", False)
        self.traj_root = getattr(args, "traj_root", None)
        self.traj_ext = getattr(args, "traj_ext", ".npz")
        self.traj_key = getattr(args, "traj_key", None)
        self.traj_dim = getattr(args, "traj_dim", None)
        self.traj_reduce = getattr(args, "traj_reduce", "flatten")  # flatten / mean
        self.traj_missing = getattr(args, "traj_missing", "zeros")  # zeros / error

        if self.use_traj:
            assert self.traj_root is not None, "args.traj_root is required when use_traj=True"
            assert self.traj_dim is not None, "args.traj_dim is required when use_traj=True"


        # --- annotation (split list) path ---
        self.annotation_path = getattr(args, "traintestlist", None)

        if self.annotation_path is None:
            repo_root = os.path.dirname(__file__)
            splits_root = os.path.join(repo_root, "splits")

            ds_name = os.path.basename(str(self.data_dir)).lower()  # e.g. "multisports_fsar"
            cand = None
            if os.path.isdir(splits_root):
                # 先做：完全相等 / 包含關係 的匹配（不分大小寫）
                for sub in os.listdir(splits_root):
                    sub_l = sub.lower()
                    if sub_l == ds_name or sub_l in ds_name or ds_name in sub_l:
                        cand = os.path.join(splits_root, sub)
                        break

            if cand is None:
                raise ValueError(f"Cannot find matching splits under {splits_root} for dataset={self.data_dir}")
            self.annotation_path = cand

        self.way=args.way
        self.eval_way=args.eval_way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.val_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []

        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))
        video_transform_list.append(ColorJitter(brightness=0.5,
                                                contrast=0.5,
                                                saturation=0.5,
                                                hue=0.25))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)

    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            self.mem = open(zip_fn, 'rb').read()
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))
        else:
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))

            class_folders = list(set([x.split(os.sep)[-3] for x in dir_list if len(x.split(os.sep)) > 2]))
            class_folders.sort()
            self.class_folders = class_folders
            video_folders = list(set([x.split(os.sep)[-2] for x in dir_list if len(x.split(os.sep)) > 3]))
            video_folders.sort()
            self.video_folders = video_folders

            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}

            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()

            c = self.get_train_val_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:

                class_folder, video_folder, jpg = img_path.split(os.sep)[-3:]

                if video_folder != last_video_folder:
                    if len(insert_frames) >= self.seq_len:
                    #if len(insert_frames) > 0:
                        c = self.get_train_val_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]

                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_val_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
            #if c != None and len(insert_frames) > 0:
                c.add_vid(insert_frames, last_video_class)
        else:
            split_folders = os.listdir(self.data_dir)
            for split in split_folders:
                class_folders = os.listdir(os.path.join(self.data_dir, split))
                class_folders.sort()
                self.class_folders = class_folders
                for class_folder in class_folders:
                    video_folders = os.listdir(os.path.join(self.data_dir, split, class_folder))
                    video_folders.sort()
                    for video_folder in video_folders:
                        c = self.get_train_val_or_test_db(video_folder)
                        if c == None:
                            continue
                        imgs = os.listdir(os.path.join(self.data_dir, split, class_folder, video_folder))
                        if len(imgs) < self.seq_len:
                        #if len(imgs) == 0:
                            continue
                        imgs.sort()
                        paths = [os.path.join(self.data_dir, split, class_folder, video_folder, img) for img in imgs]
                        paths.sort()
                        class_id = class_folders.index(class_folder)
                        c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, val: {}, test: {}".format(len(self.train_split), len(self.val_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_val_or_test_db(self, split=None):
        if split is None:
            if self.split == 'train':
                return self.train_split
            elif self.split == 'val':
                return self.val_split
            elif self.split == 'test':
                return self.test_split
            else:
                return None
        else:
            if split in self.train_val_test_lists["train"]:
                return self.train_split
            elif split in self.train_val_test_lists["val"]:
                return self.val_split
            elif split in self.train_val_test_lists["test"]:
                return self.test_split
            else:
                return None

    """ load the paths of all videos in the train and test splits. """
    def _select_fold(self):
        lists = {}
        for name in ["train", "val", "test"]:
            fname = "{}list.txt".format(name)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()

                if "kinetics" in self.args.dataset:
                    data = [x.strip('\n') for x in data]
                    data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                elif "ssv2_small_V2" in self.args.dataset:
                    data = [x.strip('\n') for x in data]
                    data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                else:
                    data = [x.strip().split(" ")[0] for x in data]
                    data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]

                selected_files.extend(data)
            lists[name] = selected_files
        self.train_val_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_val_or_test_db()
        return 1000000
        return len(c)

    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_val_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes

    """Loads a single image from a specified path """
    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i

    def _parse_video_id_from_paths(self, paths):
        """
        從某支影片的 frame path 推出 video_id。
        - 非 zip: .../<split>/<class>/<video_folder>/<img>.jpg  -> video_folder
        - zip:    .../<class>/<video_folder>/<img>.jpg          -> video_folder
        """
        p0 = paths[0]
        parts = p0.split(os.sep)
        # 最後是 img，倒數第二個通常是 video_folder
        if len(parts) < 2:
            return os.path.splitext(os.path.basename(p0))[0]
        return parts[-2]


    def _reduce_pred_tracks_to_seq(pred_tracks, pred_visibility=None, point_queries=None, group_by_query=False):
        """
        Convert pred_tracks [T, P, 2] (+visibility [T,P]) into per-frame stats [T, D_small].

        Default (group_by_query=False):
        D_small = 11 = mean_xy(2) + std_xy(2) + min_xy(2) + max_xy(2) + vis_ratio(1) + mean_vel_xy(2)

        If group_by_query=True and point_queries is provided (values 0..Q-1):
        compute the above stats per query-group and concatenate (D_small = 11 * Q)
        """

        def _safe_to_numpy(x):
            """Ensure x is a numpy array (no-op if already)."""
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            try:
                return np.asarray(x)
            except Exception:
                return None
        pred_tracks = _safe_to_numpy(pred_tracks)
        if pred_tracks is None or pred_tracks.ndim != 3 or pred_tracks.shape[-1] != 2:
            return None

        T, P, _ = pred_tracks.shape
        xy = pred_tracks.astype(np.float32, copy=False)

        if pred_visibility is None:
            vis = np.ones((T, P), dtype=bool)
        else:
            vis = _safe_to_numpy(pred_visibility)
            if vis is None or vis.shape[:2] != (T, P):
                vis = np.ones((T, P), dtype=bool)
            else:
                vis = vis.astype(bool, copy=False)

        def stats_for_mask(xy_t, vis_t):
            # xy_t: [P,2], vis_t: [P]
            if vis_t.any():
                vxy = xy_t[vis_t]
                mean_xy = vxy.mean(axis=0)
                std_xy  = vxy.std(axis=0)
                min_xy  = vxy.min(axis=0)
                max_xy  = vxy.max(axis=0)
                vis_ratio = np.array([vis_t.mean()], dtype=np.float32)
            else:
                mean_xy = np.zeros((2,), dtype=np.float32)
                std_xy  = np.zeros((2,), dtype=np.float32)
                min_xy  = np.zeros((2,), dtype=np.float32)
                max_xy  = np.zeros((2,), dtype=np.float32)
                vis_ratio = np.array([0.0], dtype=np.float32)
            return mean_xy, std_xy, min_xy, max_xy, vis_ratio

        # velocity on mean positions (use visible points mean)
        mean_pos = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            if vis[t].any():
                mean_pos[t] = xy[t][vis[t]].mean(axis=0)
            else:
                mean_pos[t] = mean_pos[t-1] if t > 0 else 0.0

        vel = np.zeros((T, 2), dtype=np.float32)
        vel[1:] = mean_pos[1:] - mean_pos[:-1]  # simple diff

        if group_by_query and point_queries is not None:
            pq = _safe_to_numpy(point_queries)
            if pq is None or pq.shape[0] != P:
                group_by_query = False
            else:
                pq = pq.astype(np.int64, copy=False)
                Q = int(pq.max()) + 1 if pq.size > 0 else 1
                feats = []
                for t in range(T):
                    per_q = []
                    for q in range(Q):
                        mask_q = (pq == q)
                        vis_q = vis[t] & mask_q
                        mean_xy, std_xy, min_xy, max_xy, vis_ratio = stats_for_mask(xy[t], vis_q)
                        f = np.concatenate([mean_xy, std_xy, min_xy, max_xy, vis_ratio], axis=0)  # 9 dims
                        per_q.append(f)
                    f_all = np.concatenate(per_q, axis=0)  # 9*Q
                    f_all = np.concatenate([f_all, vel[t]], axis=0)  # +2 => (9*Q + 2)
                    feats.append(f_all)
                seq = np.stack(feats, axis=0).astype(np.float32)
                return seq

        # default non-grouped
        feats = []
        for t in range(T):
            mean_xy, std_xy, min_xy, max_xy, vis_ratio = stats_for_mask(xy[t], vis[t])
            f = np.concatenate([mean_xy, std_xy, min_xy, max_xy, vis_ratio, vel[t]], axis=0)  # 11 dims
            feats.append(f)
        seq = np.stack(feats, axis=0).astype(np.float32)
        return seq


    # --------------------------
    # Paste these into your VideoReader class
    # --------------------------

    def _load_traj_npz(self, video_id):
        """
        Default path: traj_root/<video_id>.npz (or f"{video_id}{self.traj_ext}")
        Returns a dict with available fields:
        - traj_vec: [36]
        - traj_seq: [T,6]
        - pred_tracks: [T,P,2]
        - pred_visibility: [T,P]
        - point_queries: [P]
        """
        npz_path = os.path.join(self.traj_root, f"{video_id}{getattr(self, 'traj_ext', '.npz')}")
        if not os.path.exists(npz_path):
            if getattr(self, "traj_missing", "zero") == "error":
                raise FileNotFoundError(f"traj not found: {npz_path}")
            return None

        data = np.load(npz_path, allow_pickle=True)
        keys = set(list(data.keys()))

        out = {}
        # Prefer the keys you actually have
        if "traj_vec" in keys:
            out["traj_vec"] = data["traj_vec"]
        if "traj_seq" in keys:
            out["traj_seq"] = data["traj_seq"]
        if "pred_tracks" in keys:
            out["pred_tracks"] = data["pred_tracks"]
        if "pred_visibility" in keys:
            out["pred_visibility"] = data["pred_visibility"]
        if "point_queries" in keys:
            out["point_queries"] = data["point_queries"]

        # Fallback: if user forced a key
        forced_key = getattr(self, "traj_key", None)
        if forced_key is not None and forced_key in keys and forced_key not in out:
            out[forced_key] = data[forced_key]

        return out if len(out) > 0 else None


    def _sample_traj(self, traj_full, idxs=None):
        """
        Output: torch.FloatTensor [seq_len, traj_dim]

        Supports traj_full as:
        - dict returned by _load_traj_npz()
        - raw np.ndarray: [T,D] or [T,P,2] or [D] (will be handled)
        """

        def _reduce_pred_tracks_to_seq(pred_tracks, pred_visibility=None, point_queries=None, group_by_query=False):
            """
            Convert pred_tracks [T, P, 2] (+visibility [T,P]) into per-frame stats [T, D_small].

            Default (group_by_query=False):
            D_small = 11 = mean_xy(2) + std_xy(2) + min_xy(2) + max_xy(2) + vis_ratio(1) + mean_vel_xy(2)

            If group_by_query=True and point_queries is provided (values 0..Q-1):
            compute the above stats per query-group and concatenate (D_small = 11 * Q)
            """
        def _safe_to_numpy(x):
            """Ensure x is a numpy array (no-op if already)."""
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            try:
                return np.asarray(x)
            except Exception:
                return None
        
        def _sample_by_interp(arr_TD, seq_len):
            """
            Linear interpolation resampling for [T, D] -> [seq_len, D].
            Works even if seq_len > T; avoids repeating last frame.
            """
            
            def _safe_to_numpy(x):
                """Ensure x is a numpy array (no-op if already)."""
                if x is None:
                    return None
                if isinstance(x, np.ndarray):
                    return x
                try:
                    return np.asarray(x)
                except Exception:
                    return None

            def _linspace_indices(T, seq_len):
                """Return float indices for linspace sampling: shape [seq_len]."""
                if T <= 1:
                    return np.zeros((seq_len,), dtype=np.float32)
                return np.linspace(0, T - 1, seq_len, dtype=np.float32)

                
            if arr_TD is None:
                return None
            arr_TD = _safe_to_numpy(arr_TD)
            if arr_TD is None or arr_TD.ndim != 2:
                return None

            T, D = arr_TD.shape
            if T == seq_len:
                return arr_TD.astype(np.float32, copy=False)

            if T <= 1:
                out = np.repeat(arr_TD[:1], seq_len, axis=0)
                return out.astype(np.float32, copy=False)

            # interpolate each dim independently
            x_old = np.arange(T, dtype=np.float32)
            x_new = _linspace_indices(T, seq_len)
            out = np.zeros((seq_len, D), dtype=np.float32)
            for d in range(D):
                out[:, d] = np.interp(x_new, x_old, arr_TD[:, d].astype(np.float32))
            return out

        seq_len = int(getattr(self.args, "seq_len", 8))
        traj_dim = int(getattr(self, "traj_dim", 64))

        # missing -> zeros
        if traj_full is None:
            return torch.zeros(seq_len, traj_dim, dtype=torch.float32)

        # Choose mode: 'auto'|'traj_vec'|'traj_seq'|'pred_tracks'
        mode = getattr(self, "traj_mode", "auto")
        group_by_query = bool(getattr(self, "traj_group_by_query", False))

        # -------- pick the source array (as [T,D] or [D]) ----------
        src = None

        if isinstance(traj_full, dict):
            if mode == "traj_vec" and "traj_vec" in traj_full:
                src = traj_full["traj_vec"]
            elif mode == "traj_seq" and "traj_seq" in traj_full:
                src = traj_full["traj_seq"]
            elif mode in ("pred_tracks", "tracks") and "pred_tracks" in traj_full:
                src = ("pred_tracks", traj_full.get("pred_tracks"), traj_full.get("pred_visibility"), traj_full.get("point_queries"))
            else:
                # auto priority
                if "traj_vec" in traj_full:
                    src = traj_full["traj_vec"]
                elif "traj_seq" in traj_full:
                    src = traj_full["traj_seq"]
                elif "pred_tracks" in traj_full:
                    src = ("pred_tracks", traj_full.get("pred_tracks"), traj_full.get("pred_visibility"), traj_full.get("point_queries"))
                else:
                    # fallback: take any first key
                    k0 = next(iter(traj_full.keys()))
                    src = traj_full[k0]
        else:
            src = traj_full

        # -------- convert src into [T,D] --------
        arr_TD = None


        if isinstance(src, tuple) and len(src) == 4 and src[0] == "pred_tracks":
            _, pred_tracks, pred_vis, point_queries = src
            arr_TD = _reduce_pred_tracks_to_seq(
                pred_tracks,
                pred_visibility=pred_vis,
                point_queries=point_queries,
                group_by_query=group_by_query,
            )  # [T, D_small]
        else:
            src = _safe_to_numpy(src)
            if src is None:
                return torch.zeros(seq_len, traj_dim, dtype=torch.float32)

            if src.ndim == 1:
                # traj_vec: [D] -> tile to [seq_len, D]
                vec = src.astype(np.float32, copy=False).reshape(1, -1)
                arr_TD = np.repeat(vec, seq_len, axis=0)  # [seq_len, D]
            elif src.ndim == 2:
                # traj_seq: [T,D]
                arr_TD = src.astype(np.float32, copy=False)
            elif src.ndim == 3 and src.shape[-1] == 2:
                # if user passed raw tracks without dict, reduce by mean across points
                if getattr(self, "traj_reduce", "mean") == "mean":
                    arr_TD = src.mean(axis=1).astype(np.float32, copy=False)  # [T,2]
                else:
                    arr_TD = src.reshape(src.shape[0], -1).astype(np.float32, copy=False)  # [T, 2P]
            else:
                # fallback flatten to [T,D] if possible
                if src.shape[0] > 0:
                    arr_TD = src.reshape(src.shape[0], -1).astype(np.float32, copy=False)
                else:
                    return torch.zeros(seq_len, traj_dim, dtype=torch.float32)

        if arr_TD is None:
            return torch.zeros(seq_len, traj_dim, dtype=torch.float32)

        # -------- resample to seq_len robustly --------
        if arr_TD.shape[0] != seq_len:
            arr_TD = _sample_by_interp(arr_TD, seq_len)  # [seq_len, D]

        # -------- pad/truncate to traj_dim --------
        D = arr_TD.shape[1]
        if D < traj_dim:
            pad = np.zeros((seq_len, traj_dim - D), dtype=np.float32)
            arr_TD = np.concatenate([arr_TD, pad], axis=1)
        elif D > traj_dim:
            arr_TD = arr_TD[:, :traj_dim]

        # final torch
        out = torch.from_numpy(arr_TD).float()
        return out



    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        def _sample_frame_indices(n_frames, seq_len, split):
            if n_frames <= 0:
                return [0] * seq_len

            if n_frames < seq_len:
                idx_f = np.linspace(0, n_frames - 1, num=seq_len)
                idxs = [int(round(f)) for f in idx_f]
                idxs = [min(max(i, 0), n_frames - 1) for i in idxs]
                return idxs

            interval = n_frames // seq_len  # >=1
            if split == "train":
                idxs = []
                for ind in range(seq_len):
                    lo = ind * interval
                    hi = (ind + 1) * interval - 1
                    lo = min(lo, n_frames - 1)
                    hi = min(hi, n_frames - 1)
                    if hi < lo:
                        hi = lo
                    idxs.append(random.randint(lo, hi))
                return idxs
            else:
                idxs = []
                for ind in range(seq_len):
                    lo = ind * interval
                    hi = (ind + 1) * interval - 1
                    lo = min(lo, n_frames - 1)
                    hi = min(hi, n_frames - 1)
                    mid = (lo + hi) // 2
                    idxs.append(mid)
                return idxs

        c = self.get_train_val_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)

        idxs = _sample_frame_indices(n_frames, self.args.seq_len, self.split)
        imgs = [self.read_single_image(paths[i]) for i in idxs]

        if self.transform is not None:
            transform = self.transform["train"] if self.split == "train" else self.transform["test"]
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)  # [seq_len, 3, H, W]

        # ---------- traj branch ----------
        tr = None
        if getattr(self, "use_traj", False):
            # 注意：vid_id 是 split 內的 index，不是 video_folder
            video_id = self._parse_video_id_from_paths(paths)

            traj_full = self._load_traj_npz(video_id)      # dict or None
            tr = self._sample_traj(traj_full, idxs=idxs)   # torch [seq_len, traj_dim]

            # 保險：確保 dtype/shape
            if not isinstance(tr, torch.Tensor):
                tr = torch.zeros(self.args.seq_len, self.traj_dim, dtype=torch.float32)
            else:
                tr = tr.float()
                if tr.ndim != 2 or tr.shape[0] != self.args.seq_len or tr.shape[1] != self.traj_dim:
                    # 強制修正到 [seq_len, traj_dim]
                    tr = tr.reshape(tr.shape[0], -1)
                    # resample len
                    if tr.shape[0] != self.args.seq_len:
                        tr = torch.from_numpy(
                            np.interp(
                                np.linspace(0, tr.shape[0] - 1, self.args.seq_len).astype(np.float32),
                                np.arange(tr.shape[0], dtype=np.float32),
                                tr.numpy(),
                            )
                        ).float()
                    # pad/trunc dim
                    if tr.shape[1] < self.traj_dim:
                        pad = torch.zeros(self.args.seq_len, self.traj_dim - tr.shape[1])
                        tr = torch.cat([tr, pad], dim=1)
                    elif tr.shape[1] > self.traj_dim:
                        tr = tr[:, :self.traj_dim]

        return imgs, vid_id, tr


    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        if self.split == "train":
            c = self.train_split
        elif self.split == "val":
            c = self.val_split
        elif self.split == "test":
            c = self.test_split
        classes = c.get_unique_classes()

        need = self.args.shot + (self.args.query_per_class if self.split=="train" else self.args.query_per_class_test)
        eligible_classes = [cl for cl in classes if c.get_num_videos_for_class(cl) >= need]

        if len(eligible_classes) < (self.way if self.split=="train" else self.eval_way):
            raise ValueError(
                f"Not enough eligible classes for split={self.split}. "
                f"need_per_class={need}, eligible={len(eligible_classes)}, "
                f"required_way={(self.way if self.split=='train' else self.eval_way)}"
            )

        if self.split == "train":
            batch_classes = random.sample(eligible_classes, self.way)
        else:
            batch_classes = random.sample(eligible_classes, self.eval_way)

        if self.split == "train":
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        support_traj = []
        target_traj = []

        for bl, bc in enumerate(batch_classes):

            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid, vid_id, tr = self.get_seq(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
                real_support_labels.append(bc)
                if self.use_traj:
                    support_traj.append(tr)
            for idx in idxs[self.args.shot:]:
                vid, vid_id, tr = self.get_seq(bc, idx)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
                if self.use_traj:
                    target_traj.append(tr)

        s = list(zip(support_set, support_labels, real_support_labels))
        # random.shuffle(s)
        support_set, support_labels, real_support_labels = zip(*s)

        t = list(zip(target_set, target_labels, real_target_labels))
        # random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)

        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes)

        if self.use_traj:
            # 每個 tr 是 [seq_len, traj_dim]，cat 後變 [num_vid*seq_len, traj_dim]
            support_traj = torch.cat(support_traj, dim=0)
            target_traj = torch.cat(target_traj, dim=0)


        out = {"support_set": support_set,
               "support_labels": support_labels,
               "real_support_labels": real_support_labels,
               "target_set": target_set,
               "target_labels": target_labels,
               "real_target_labels": real_target_labels,
               "batch_class_list": batch_classes}

        if self.use_traj:
            out["support_traj"] = support_traj
            out["target_traj"] = target_traj

        return out