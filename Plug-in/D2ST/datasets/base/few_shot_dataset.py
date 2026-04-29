#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

import os
import random
import torch
import torch.utils.data
import utils.logging as logging
import numpy as np
import traceback
import utils.bucket as bu

from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo,
    AutoResizedCropVideo,
    KineticsResizedCrop,
    KineticsResizedCropFewshot
)

from datasets.base.base_dataset import BaseVideoDataset
from datasets.utils.random_erasing import RandomErasing

logger = logging.get_logger(__name__)


class Split_few_shot():
    """Contains video frame paths and ground truth labels for a single split (e.g. train videos). """

    def __init__(self, folder, split_dataset, dataset):
        # self.args = args

        self.gt_a_list = []
        self.videos = []
        self.split_dataset = split_dataset

        for class_folder in folder:
            paths = class_folder.strip().split('//')[-1]
            class_id = int(class_folder.strip().split('//')[0][len(split_dataset):])
            self.add_vid(paths, class_id)

        logger.info("loaded {} videos from {} dataset: {} !".format(len(self.gt_a_list), split_dataset, dataset))

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

    def get_single_video(self, index):
        return self.videos[index], self.gt_a_list[index]

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def __len__(self):
        return len(self.gt_a_list)


class Few_shot(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Few_shot, self).__init__(cfg, split)
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        self.split_dataset = split

        # ---- traj cfg: same as MOLO ----
        self.use_traj = getattr(self.cfg.TRAIN, "USE_TRAJ", False)

        self.traj_root = getattr(self.cfg.TRAJ, "ROOT", "")
        self.traj_suffix = getattr(self.cfg.TRAJ, "SUFFIX", ".npz")
        self.traj_key = getattr(self.cfg.TRAJ, "KEY", "traj_seq")
        self.traj_missing = getattr(self.cfg.TRAJ, "MISSING", "zeros")  # "zeros" or "error"

        self.traj_seq_len = int(getattr(self.cfg.TRAJ, "SEQ_LEN", self.cfg.DATA.NUM_INPUT_FRAMES))
        self.traj_dim = int(getattr(self.cfg.TRAJ, "DIM", 6))

        # False: traj_root/00035_0.npz
        # True : traj_root/LeftPR/00035_0.npz
        self.traj_keep_subdir = getattr(self.cfg.TRAJ, "KEEP_SUBDIR", False)

        self._traj_debug_count = 0
        self._traj_debug_max = 0

    def _get_ssl_label(self, frames):
        pass

    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """

        name = "{}_few_shot.txt".format(
            "train" if self.split == "train" else "test",
        )
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        class_ = self._samples[index]["label_idx"]
        video_path = os.path.join(self.data_root_dir, self._samples[index]["id"] + ".mp4")
        sample_info = {
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info

    def _construct_dataset(self, cfg):
        self.dataset_name = self.cfg.TRAIN.DATASET
        self._num_clips = 1
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        for retry in range(5):
            try:
                logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
                local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
                local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
                if local_file[-4:] == ".csv":
                    import pandas
                    lines = pandas.read_csv(local_file)
                    for line in lines.values.tolist():
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                elif local_file[-4:] == "json":
                    import json
                    with open(local_file, "r") as f:
                        lines = json.load(f)
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                else:
                    with open(local_file) as f:
                        lines = f.readlines()
                        for line in lines:
                            for idx in range(self._num_clips):
                                self._samples.append(line.strip())
                                self._spatial_temporal_index.append(idx)
                self.split_few_shot = Split_few_shot(lines, self.split, dataset=self.dataset_name)
                logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
                break
            except:
                if retry < 4:
                    continue
                else:
                    raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))

        if hasattr(self.cfg.TRAIN, "FEW_SHOT") and self.cfg.TRAIN.FEW_SHOT and self.split == "train":
            """ Sample number setting for training in few-shot settings: 
                During few shot training, the batch size could be larger than the size of the training samples.
                Therefore, the number of samples in the same sample is multiplied by 10 times, and the training schedule is reduced by 10 times. 
            """
            self._samples = self._samples * 10
            print("10 FOLD FEW SHOT SAMPLES")

        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))

    def _video_to_traj_path(self, video_path):
        """
        Convert video path to trajectory npz path.

        Example:
        video_path:
            /home/.../FineSports_video/LeftPR/00035_0.avi

        if traj_keep_subdir=False:
            traj_root/00035_0.npz

        if traj_keep_subdir=True:
            traj_root/LeftPR/00035_0.npz
        """
        video_path = str(video_path).replace("\\", "/")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        class_name = os.path.basename(os.path.dirname(video_path))

        if self.traj_keep_subdir:
            rel_path = os.path.join(class_name, video_name + self.traj_suffix)
        else:
            rel_path = video_name + self.traj_suffix

        return os.path.join(self.traj_root, rel_path)

    def _fix_traj_shape(self, arr):
        """
        Normalize npz trajectory array into [traj_seq_len, traj_dim].

        Supported:
        - traj_vec: [D] -> repeat to [T, D]
        - traj_seq: [T, D]
        - pred_tracks: [T, P, 2] -> flatten to [T, 2P]
        """
        seq_len = self.traj_seq_len
        traj_dim = self.traj_dim

        arr = np.asarray(arr)

        # case 1: [D] -> [seq_len, D]
        if arr.ndim == 1:
            arr = arr.astype(np.float32).reshape(1, -1)
            arr = np.repeat(arr, seq_len, axis=0)

        # case 2: [T, D]
        elif arr.ndim == 2:
            arr = arr.astype(np.float32)

        # case 3: [T, P, 2] -> [T, 2P]
        elif arr.ndim == 3 and arr.shape[-1] == 2:
            T = arr.shape[0]
            arr = arr.reshape(T, -1).astype(np.float32)

        else:
            if self.traj_missing == "error":
                raise ValueError(f"[traj] unexpected shape: {arr.shape}")
            return np.zeros((seq_len, traj_dim), dtype=np.float32)

        # ---- align temporal length T -> traj_seq_len ----
        T, D = arr.shape

        if T == seq_len:
            out = arr

        elif T <= 1:
            out = np.repeat(arr[:1], seq_len, axis=0)

        else:
            x_old = np.arange(T, dtype=np.float32)
            x_new = np.linspace(0, T - 1, seq_len, dtype=np.float32)

            out = np.zeros((seq_len, D), dtype=np.float32)
            for d in range(D):
                out[:, d] = np.interp(x_new, x_old, arr[:, d].astype(np.float32))

        # ---- align feature dim D -> traj_dim ----
        D = out.shape[1]

        if D < traj_dim:
            pad = np.zeros((seq_len, traj_dim - D), dtype=np.float32)
            out = np.concatenate([out, pad], axis=1)

        elif D > traj_dim:
            out = out[:, :traj_dim]

        return out.astype(np.float32)


    def _load_traj_npz(self, video_path):
        traj_path = self._video_to_traj_path(video_path)

        if self._traj_debug_count < self._traj_debug_max:
            print(f"[TRAJ DEBUG] video_path = {video_path}")
            print(f"[TRAJ DEBUG] traj_path = {traj_path}")
            print(f"[TRAJ DEBUG] exists = {os.path.exists(traj_path)}")

        if not os.path.exists(traj_path):
            if self.traj_missing == "error":
                raise FileNotFoundError(f"[traj] missing: {traj_path}")

            if self._traj_debug_count < self._traj_debug_max:
                print("[TRAJ DEBUG] missing traj, return zeros")

            self._traj_debug_count += 1
            return torch.zeros(self.traj_seq_len, self.traj_dim, dtype=torch.float32)

        data = np.load(traj_path, allow_pickle=True)

        if self._traj_debug_count < self._traj_debug_max:
            print(f"[TRAJ DEBUG] npz keys = {list(data.keys())}")

        if self.traj_key not in data:
            if self.traj_missing == "error":
                raise KeyError(
                    f"[traj] key '{self.traj_key}' not found in {traj_path}. "
                    f"Available keys: {list(data.keys())}"
                )

            if self._traj_debug_count < self._traj_debug_max:
                print(f"[TRAJ DEBUG] key {self.traj_key} not found, return zeros")

            self._traj_debug_count += 1
            return torch.zeros(self.traj_seq_len, self.traj_dim, dtype=torch.float32)

        arr = data[self.traj_key]

        if self._traj_debug_count < self._traj_debug_max:
            print(f"[TRAJ DEBUG] raw traj shape = {arr.shape}")

        arr = self._fix_traj_shape(arr)

        if self._traj_debug_count < self._traj_debug_max:
            print(f"[TRAJ DEBUG] fixed traj shape = {arr.shape}")
            print(f"[TRAJ DEBUG] traj mean = {arr.mean():.6f}, std = {arr.std():.6f}")
            print("-" * 60)

        self._traj_debug_count += 1

        return torch.from_numpy(arr).to(torch.float32).contiguous().clone()
    
    def __getitem__(self, index):
        """
            Returns:
                frames (dict): {
                    "video": (tensor), 
                    "text_embedding" (optional): (tensor)
                }
                labels (dict): {
                    "supervised": (tensor),
                    "self-supervised" (optional): (...)
                }
        """
        if self.cfg.TRAIN.META_BATCH:
            """returns dict of support and target images and labels for a meta training task"""
            # select classes to use for this task
            c = self.split_few_shot
            classes = c.get_unique_classes()
            batch_classes = random.sample(classes, self.cfg.TRAIN.WAY)
            if self.split != "train" and hasattr(self.cfg.TRAIN, "WAT_TEST"):
                batch_classes = random.sample(classes, self.cfg.TRAIN.WAT_TEST)

            if self.split == "train":
                n_queries = self.cfg.TRAIN.QUERY_PER_CLASS
            else:
                n_queries = self.cfg.TRAIN.QUERY_PER_CLASS_TEST

            retries = 5
            for retry in range(retries):
                try:

                    support_set = []
                    support_labels = []
                    target_set = []
                    target_labels = []
                    real_support_labels = []
                    real_target_labels = []

                    support_traj = []
                    target_traj = []

                    for bl, bc in enumerate(batch_classes):
                        n_total = c.get_num_videos_for_class(bc)
                        # retries = 5
                        # for retry in range(retries):
                        #     try:
                        idxs = random.sample([i for i in range(n_total)], self.cfg.SUPPORT_SHOT + n_queries)

                        for idx in idxs[0:self.cfg.SUPPORT_SHOT]:
                            if hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF_SUPPORT") and self.cfg.AUGMENTATION.SUPPORT_QUERY_DIFF_SUPPORT and self.split_dataset == "train":
                                if self.use_traj:
                                    vid, vid_id, traj = self.get_seq_query(bc, idx)
                                else:
                                    vid, vid_id = self.get_seq_query(bc, idx)
                                    traj = None                            
                            else:
                                if self.use_traj:
                                    vid, vid_id, traj = self.get_seq(bc, idx)
                                else:
                                    vid, vid_id = self.get_seq(bc, idx)
                                    traj = None

                            support_set.append(vid)
                            support_labels.append(bl)
                            real_support_labels.append(bc)

                            if self.use_traj:
                                support_traj.append(traj)         
                        # try:
                        for idx in idxs[self.cfg.SUPPORT_SHOT:]:
                            if hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF") and self.cfg.AUGMENTATION.SUPPORT_QUERY_DIFF and self.split_dataset == "train":
                                if self.use_traj:
                                    vid, vid_id, traj = self.get_seq_query(bc, idx)
                                else:
                                    vid, vid_id = self.get_seq_query(bc, idx)
                                    traj = None
                            else:
                                if self.use_traj:
                                    vid, vid_id, traj = self.get_seq(bc, idx)
                                else:
                                    vid, vid_id = self.get_seq(bc, idx)
                                    traj = None                            
                            target_set.append(vid)
                            target_labels.append(bl)
                            real_target_labels.append(bc)

                            if self.use_traj:
                                target_traj.append(traj)
                    break

                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry + 1, retries, idxs, bc
                    ))

            if self.use_traj:
                s = list(zip(support_set, support_labels, real_support_labels, support_traj))
                if len(s) == 0:
                    raise RuntimeError("support list is empty before zip/unzip.")

                random.shuffle(s)
                support_set, support_labels, real_support_labels, support_traj = zip(*s)

                t = list(zip(target_set, target_labels, real_target_labels, target_traj))
                if len(t) == 0:
                    raise RuntimeError(
                        "target list is empty before zip/unzip. "
                        f"len(target_set)={len(target_set)}, "
                        f"len(target_labels)={len(target_labels)}, "
                        f"len(real_target_labels)={len(real_target_labels)}, "
                        f"len(target_traj)={len(target_traj)}"
                    )

                random.shuffle(t)
                target_set, target_labels, real_target_labels, target_traj = zip(*t)

            else:
                s = list(zip(support_set, support_labels, real_support_labels))
                if len(s) == 0:
                    raise RuntimeError("support list is empty before zip/unzip.")

                random.shuffle(s)
                support_set, support_labels, real_support_labels = zip(*s)

                t = list(zip(target_set, target_labels, real_target_labels))
                if len(t) == 0:
                    raise RuntimeError("target list is empty before zip/unzip.")

                random.shuffle(t)
                target_set, target_labels, real_target_labels = zip(*t)

            support_set = torch.cat(support_set)  # [200, 3, 224, 224]

            target_set = torch.cat(target_set)  # [200, 3, 224, 224]

            if self.use_traj:
                support_traj = torch.stack(support_traj, dim=0)
                target_traj = torch.stack(target_traj, dim=0)

            support_labels = torch.FloatTensor(support_labels)
            target_labels = torch.FloatTensor(target_labels)
            real_target_labels = torch.FloatTensor(real_target_labels)  # shape: [25]
            real_support_labels = torch.FloatTensor(real_support_labels)
            # [45., 59., 45., 11., 39., 39., 39., 11., 11., 25., 25., 25., 59., 45., 11., 25., 59., 25., 45., 39., 45., 59., 39., 59., 11.]
            batch_classes = torch.FloatTensor(batch_classes)  # [45., 11., 59., 25., 39.]

            out = {
                "support_set": support_set,
                "support_labels": support_labels,
                "target_set": target_set,
                "target_labels": target_labels,
                "real_target_labels": real_target_labels,
                "batch_class_list": batch_classes,
                "real_support_labels": real_support_labels,
            }

            if self.use_traj:
                out["support_traj"] = support_traj
                out["target_traj"] = target_traj

                if self._traj_debug_count < self._traj_debug_max:
                    print("[BATCH TRAJ DEBUG]")
                    print("[DBG] support_set shape:", support_set.shape)
                    print("[DBG] target_set shape:", target_set.shape)
                    print("[DBG] support_traj shape:", support_traj.shape)
                    print("[DBG] target_traj shape:", target_traj.shape)
                    print("[DBG] support_traj mean/std:", support_traj.mean().item(), support_traj.std().item())
                    print("[DBG] target_traj mean/std:", target_traj.mean().item(), target_traj.std().item())

            return out


        else:

            sample_info = self._get_sample_info(index)

            retries = 1 if self.split == "train" else 10
            for retry in range(retries):
                try:
                    data, file_to_remove, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry + 1, retries, index, sample_info["path"]
                    ))

            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
                if not hasattr(self, "label_embd"):
                    self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
                data["text_embedding"] = self.label_embd

            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()

            # self.visualize_frames(data["video"], index)

            labels = {}
            labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
            if self.cfg.PRETRAIN.ENABLE:
                try:
                    data, labels["self-supervised"] = self.ssl_generator(data, index)
                except Exception as e:
                    traceback.print_exc()
                    print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                        index, sample_info["path"], data["video"].shape
                    ))
                    return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            else:
                labels["self-supervised"] = {}
                if "flow" in data.keys() and "video" in data.keys():
                    data = self.transform(data)
                elif "video" in data.keys():
                    data["video"] = self.transform(data["video"])  # C, T, H, W = 3, 16, 240, 320, RGB

            if "Slowfast" in self.cfg.VIDEO.BACKBONE.META_ARCH and self.split not in ['extract_feat']:
                slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1] // self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA + 1).long()[:-1]
                fast_frames = data["video"].clone()
                slow_frames = data["video"][:, slow_idx, :, :].clone()
                data["video"] = [slow_frames, fast_frames]
            bu.clear_tmp_file(file_to_remove)

            if self.split in ['extract_feat']:
                meta = {'video_name': sample_info['video_name'],
                        'subset': sample_info['subset']}
            else:
                meta = {}

            # self.reversed_visualize_frames(data["video"], index)
            return data, labels, index, meta

    def get_seq(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.split_few_shot
        if self.cfg.TRAIN.META_BATCH:
            paths, vid_id = c.get_rand_vid(label, idx)
            # imgs = self.load_and_transform_paths(paths)
            if 'SSv2' in self.dataset_name:
                video_path = os.path.join(self.data_root_dir, paths + ".mp4")
            else:
                video_path = os.path.join(self.data_root_dir, paths)
            sample_info = {
                "path": video_path,
                # "supervised_label": class_,
            }
            # sample_info = self._get_sample_info(index)
            index = vid_id
            retries = 5 if self.split == "train" else 10  # 1
            for retry in range(retries):
                try:
                    data, file_to_remove, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry + 1, retries, index, sample_info["path"]
                    ))

            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            # if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
            #     if not hasattr(self, "label_embd"):
            #         self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
            #     data["text_embedding"] = self.label_embd

            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()

            # self.visualize_frames(data["video"], index)

            labels = {}
            labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
            if self.cfg.PRETRAIN.ENABLE:
                try:
                    data, labels["self-supervised"] = self.ssl_generator(data, index)
                except Exception as e:
                    traceback.print_exc()
                    print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                        index, sample_info["path"], data["video"].shape
                    ))
                    return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            else:
                labels["self-supervised"] = {}
                if "flow" in data.keys() and "video" in data.keys():
                    data = self.transform(data)
                elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                    data["video"] = self.transform(data["video"])  # C, T, H, W = 3, 16, 240, 320, RGB

            if "Slowfast" in self.cfg.VIDEO.BACKBONE.META_ARCH and self.split not in ['extract_feat']:
                slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1] // self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA + 1).long()[:-1]
                fast_frames = data["video"].clone()
                slow_frames = data["video"][:, slow_idx, :, :].clone()
                data["video"] = [slow_frames, fast_frames]
            bu.clear_tmp_file(file_to_remove)

            if self.use_traj:
                traj = self._load_traj_npz(video_path)
                return data["video"].permute(1, 0, 2, 3), vid_id, traj

            return data["video"].permute(1, 0, 2, 3), vid_id

    def get_seq_query(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.split_few_shot
        if self.cfg.TRAIN.META_BATCH:
            paths, vid_id = c.get_rand_vid(label, idx)
            # imgs = self.load_and_transform_paths(paths)
            if 'SSv2' in self.dataset_name:
                video_path = os.path.join(self.data_root_dir, paths + ".mp4")
            else:
                video_path = os.path.join(self.data_root_dir, paths)
            sample_info = {
                "path": video_path,
                # "supervised_label": class_,
            }
            # sample_info = self._get_sample_info(index)
            index = vid_id
            retries = 5 if self.split == "train" else 10  # 1
            for retry in range(retries):
                try:
                    data, file_to_remove, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry + 1, retries, index, sample_info["path"]
                    ))

            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            # if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
            #     if not hasattr(self, "label_embd"):
            #         self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
            #     data["text_embedding"] = self.label_embd

            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()

            # self.visualize_frames(data["video"], index)

            labels = {}
            labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
            if self.cfg.PRETRAIN.ENABLE:
                try:
                    data, labels["self-supervised"] = self.ssl_generator(data, index)
                except Exception as e:
                    traceback.print_exc()
                    print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                        index, sample_info["path"], data["video"].shape
                    ))
                    return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            else:
                labels["self-supervised"] = {}
                if "flow" in data.keys() and "video" in data.keys():
                    data = self.transform(data)
                elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                    data["video"] = self.transform_query(data["video"])  # C, T, H, W = 3, 16, 240, 320, RGB

            if "Slowfast" in self.cfg.VIDEO.BACKBONE.META_ARCH and self.split not in ['extract_feat']:
                slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1] // self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA + 1).long()[:-1]
                fast_frames = data["video"].clone()
                slow_frames = data["video"][:, slow_idx, :, :].clone()
                data["video"] = [slow_frames, fast_frames]

            bu.clear_tmp_file(file_to_remove)

            if self.use_traj:
                traj = self._load_traj_npz(video_path)
                return data["video"].permute(1, 0, 2, 3), vid_id, traj

            return data["video"].permute(1, 0, 2, 3), vid_id
        
    def __len__(self):
        if self.split == 'train' and hasattr(self.cfg.TRAIN, "NUM_TRAIN_TASKS") and self.cfg.TRAIN.NUM_TRAIN_TASKS:
            return self.cfg.TRAIN.NUM_TRAIN_TASKS
        elif hasattr(self.cfg.TRAIN, "NUM_TEST_TASKS") and self.cfg.TRAIN.NUM_TEST_TASKS:
            return self.cfg.TRAIN.NUM_TEST_TASKS
        else:
            return len(self.split_few_shot)  # len(self._samples)

    def _config_transform(self):
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list_query = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
                ), ]
            if hasattr(self.cfg.AUGMENTATION, "RANDOM_FLIP") and self.cfg.AUGMENTATION.RANDOM_FLIP:
                std_transform_list = [
                    transforms.ToTensorVideo(),
                    transforms.RandomHorizontalFlipVideo(),
                    KineticsResizedCropFewshot(
                        short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                        crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    ), ]  # KineticsResizedCrop
            else:
                std_transform_list = [
                    transforms.ToTensorVideo(),
                    KineticsResizedCropFewshot(
                        short_side_range=[self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                        crop_size=self.cfg.DATA.TRAIN_CROP_SIZE,
                    ),
                    # transforms.RandomHorizontalFlipVideo()
                ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query.append(
                ColorJitter(
                    brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                    contrast=self.cfg.AUGMENTATION.CONTRAST,
                    saturation=self.cfg.AUGMENTATION.SATURATION,
                    hue=self.cfg.AUGMENTATION.HUE,
                    grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                    consistent=self.cfg.AUGMENTATION.CONSISTENT,
                    shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                    gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                    is_split=self.cfg.AUGMENTATION.IS_SPLIT
                ),
            )
            std_transform_list_query += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
            ]

            if hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_ERASE") and self.cfg.AUGMENTATION.NO_RANDOM_ERASE:
                std_transform_list += [
                    transforms.NormalizeVideo(
                        mean=self.cfg.DATA.MEAN,
                        std=self.cfg.DATA.STD,
                        inplace=True
                    ),
                    # RandomErasing(self.cfg)
                ]
            else:
                std_transform_list += [
                    transforms.NormalizeVideo(
                        mean=self.cfg.DATA.MEAN,
                        std=self.cfg.DATA.STD,
                        inplace=True
                    ),
                    RandomErasing(self.cfg)
                ]
            self.transform = Compose(std_transform_list)
            self.transform_query = Compose(std_transform_list_query)
        elif self.split == 'val' or self.split == 'test':
            idx = -1
            if hasattr(self.cfg.DATA, "TEST_CENTER_CROP"):
                idx = self.cfg.DATA.TEST_CENTER_CROP

            if isinstance(self.cfg.DATA.TEST_SCALE, list):
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TEST_SCALE[0], self.cfg.DATA.TEST_SCALE[1]],
                    crop_size=self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops=self.cfg.TEST.NUM_SPATIAL_CROPS,
                    idx=idx
                )  # KineticsResizedCrop
            else:
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range=[self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                    crop_size=self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops=self.cfg.TEST.NUM_SPATIAL_CROPS,
                    idx=idx
                )  # KineticsResizedCrop
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)

    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, clip_idx, num_clips, num_frames, interval)
