# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
identity = lambda x: x
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from . import video_transforms, volume_transforms
from .loader import get_image_loader, get_video_loader
from .random_erasing import RandomErasing

class Arg2():
    def __init__(self):
        self.aa='rand-m7-n4-mstd0.5-inc1'
        self.train_interpolation = 'bicubic'
        self.num_sample = 2
        # self.input_size = 112  #112*112
        self.input_size = 224  #224*224
        self.data_set = 'k400'
        self.reprob = 0.25
        self.remode = 'pixel'
        self.recount = 1

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift else video_transforms.random_resized_crop)
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size,
                                                  spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

class VideoClsDataset():
    """Load your own video classification dataset."""

    def __init__(self,
                 image_size,
                 samples,
                 anno_path='',
                 data_root='',
                 mode='train',
                 clip_len=1,
                 frame_sample_rate=2,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 #num_segment=16,
                 num_segment=8,
                 num_crop=1,
                 test_num_segment=10,
                 test_num_crop=3,
                 sparse_sample=False):
        self.image_size = image_size
        self.anno_path = anno_path
        self.samples = samples
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.sparse_sample = sparse_sample
        self.aug = False
        self.rand_erase = False

        if self.mode in ['train']:
            self.aug = True
        self.rand_erase = True

        self.video_loader = get_video_loader()

    def getitem(self):

        args =  Arg2()
        args.input_size = self.image_size
        scale_t = 1
        sample =self.samples

        buffer = self.load_video(sample, sample_rate_scale=scale_t)
        while len(buffer) == 0:
            buffer = self.load_video(sample, sample_rate_scale=scale_t)

        new_frames = self._aug_frame(buffer, args)
        return new_frames

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            # crop_size=224,
            crop_size=args.input_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W

        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        fname = sample

        try:
            vr = self.video_loader(fname)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = len(vr)

        if self.mode == 'test':
            if self.sparse_sample:
                tick = length / float(self.num_segment)
                all_index = []
                for t_seg in range(self.test_num_segment):
                    tmp_index = [
                        int(t_seg * tick / self.test_num_segment + tick * x)
                        for x in range(self.num_segment)
                    ]
                    all_index.extend(tmp_index)
                all_index = list(np.sort(np.array(all_index)))
            else:
                all_index = [
                    x for x in range(0, length, self.frame_sample_rate)
                ]
                while len(all_index) < self.clip_len:
                    all_index.append(all_index[-1])

            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = length // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(
                    0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (index,
                     np.ones(self.clip_len - seg_len // self.frame_sample_rate)
                     * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                if self.mode == 'validation':
                    end_idx = (converted_len + seg_len) // 2
                else:
                    end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

class SimpleDataset:
    def __init__(self, data_path, data_file_list, transform, target_transform=identity):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)
                    label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.target_transform = target_transform
        self.checkimgsize(self.data[random.randint(0, len(self.label) - 1 )])
    
    def checkimgsize(self, data):
        data_path = os.path.join(data)
        data = Image.open(data_path).convert('RGB')
        if data.size == (84, 84):
            raise RuntimeError("Please use raw images instead of fixed resolution(84, 84) images !")

    def __getitem__(self, i):
        image_path = os.path.join(self.data[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.label[i] - min(self.label))
        return img, target

    def __len__(self):
        return len(self.label)


class SetDataset:
    def __init__(self, data_path, data_file_list, batch_size, transform):
        label = []
        data = []
        k = 0
        data_dir_list = data_file_list.replace(" ","").split(',')
        for data_file in data_dir_list:
            img_dir = data_path + '/' + data_file
            for i in os.listdir(img_dir):
                file_dir = os.path.join(img_dir, i)
                for j in os.listdir(file_dir):
                    data.append(file_dir + '/' + j)
                    label.append(k)
                k += 1
        self.data = data
        self.label = label
        self.transform = transform
        self.cl_list = np.unique(self.label).tolist()
        self.checkimgsize(self.data[random.randint(0, len(self.label) - 1 )])

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.data, self.label):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def checkimgsize(self, data):
        data_path = os.path.join(data)
        data = Image.open(data_path).convert('RGB')
        if data.size == (84, 84):
            raise RuntimeError("Please use raw images instead of fixed resolution(84, 84) images !")

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class SimpleDataset_JSON:
    def __init__(self, data_path, data_file, transform, target_transform=identity):
        data = data_path + '/' + data_file
        with open(data, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.checkimgsize(self.meta['image_names'][random.randint(0, len(self.meta['image_names']) - 1 )])

    def checkimgsize(self, data):
        data_path = os.path.join(data)
        data = Image.open(data_path).convert('RGB')
        if data.size == (84, 84):
            raise RuntimeError("Please use raw images instead of fixed resolution(84, 84) images !")
        
    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset_JSON:
    def __init__(self, image_size, data_path, data_file, batch_size, transform, args=None):
        self.image_size = image_size
        self.args = args
        data = data_path + '/' + data_file
        self.batch_size = batch_size
        with open(data, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        # 随机取一张图片/视频
        # self.meta中含有label_names image_names image_labels 这三个key
        self.checkimgsize(self.meta['image_names'][random.randint(0, len(self.cl_list) - 1 )])

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # 把同一个image_label的image_name聚合到一起
        # self.sub_meta是个字典 key是image_label value就是image_name的列表

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            # 在 SetDataset_JSON.__init__ 裡，for x,y in zip(...) 內
            if args is not None and getattr(args, "use_traj", False):
                base = os.path.splitext(os.path.basename(x))[0]
                traj_path = os.path.join(args.traj_root, base + getattr(args, "traj_suffix", ".npz"))
                if not os.path.exists(traj_path):
                    print('[DBG] traj missing. SKIP')
                    continue  # 沒有traj就跳過這個樣本
            self.sub_meta[y].append(x)
        
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False
                                      )
        #drop_last=True
        for cl in self.cl_list:
            sub_dataset = SubDataset_JSON(
                self.image_size,
                self.sub_meta[cl],
                cl,
                transform=transform,

                # NEW: traj params
                use_traj=getattr(self.args, "use_traj", False) if self.args is not None else False,
                traj_root=getattr(self.args, "traj_root", "") if self.args is not None else "",
                traj_suffix=getattr(self.args, "traj_suffix", ".npz") if self.args is not None else ".npz",
                traj_key=getattr(self.args, "traj_key", "traj_seq") if self.args is not None else "traj",
                seq_len=getattr(self.args, "seq_len", 8) if self.args is not None else 8,
                traj_dim=getattr(self.args, "traj_dim", 64) if self.args is not None else 64,
                traj_missing="error",   # debug期建議：缺檔直接報錯
            )
        
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))
    
    def checkimgsize(self, data):
        data_path = os.path.join(data)
        # data = Image.open(data_path).convert('RGB')
        # if data.size == (84, 84):
        #     raise RuntimeError("Please use raw images instead of fixed resolution(84, 84) images !")
        
    def __getitem__(self, i):
        # 取该数据集的第i个元素 返回的是第i个image_label对应的dataloader的第一个batch
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        # 该Dataset的长度是image_label的总个数
        return len(self.cl_list)


class SubDataset_JSON:
    def __init__(
        self,
        image_size,
        sub_meta,
        cl,
        transform=transforms.ToTensor(),
        target_transform=identity,
        use_traj=False,
        traj_root="",
        traj_suffix=".npz",
        traj_key="traj",          # npz裡的key
        seq_len=8,
        traj_dim=64,
        traj_missing="zeros",     # "zeros" or "error"
    ):
        self.image_size = image_size
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

        # traj cfg
        self.use_traj = use_traj
        self.traj_root = traj_root
        self.traj_suffix = traj_suffix
        self.traj_key = traj_key
        self.seq_len = int(seq_len)
        self.traj_dim = int(traj_dim)
        self.traj_missing = traj_missing
    
    def _video_to_traj_path(self, video_path: str) -> str:
        base = os.path.splitext(os.path.basename(video_path))[0]  # e.g. xxx from xxx.mp4
        return os.path.join(self.traj_root, base + self.traj_suffix)  # /traj_root/xxx.npz
    
    # data/dataset.py  SubDataset_JSON.__getitem__
    def __getitem__(self, i):
        video_path = os.path.join(self.sub_meta[i])
        image_size = self.image_size

        video_load = VideoClsDataset(image_size, samples=video_path)
        video = video_load.getitem()
        target = self.target_transform(self.cl)

        # ---- traj ----
        if not self.use_traj:
            traj = torch.zeros(self.seq_len, self.traj_dim, dtype=torch.float32)
            return (video, traj), target

        traj_path = self._video_to_traj_path(video_path)

        if not os.path.exists(traj_path):
            if self.traj_missing == "error":
                raise FileNotFoundError(f"[traj] missing: {traj_path}")
            traj = torch.zeros(self.seq_len, self.traj_dim, dtype=torch.float32)
            return (video, traj), target

        data = np.load(traj_path, allow_pickle=True)
        if self.traj_key not in data:
            if self.traj_missing == "error":
                raise KeyError(f"[traj] key '{self.traj_key}' not in {traj_path}. keys={list(data.keys())}")
            traj = torch.zeros(self.seq_len, self.traj_dim, dtype=torch.float32)
            return (video, traj), target

        arr = data[self.traj_key]  # 期望是 [T, D] or [D]...
        arr = np.asarray(arr)

        # 對齊成 [seq_len, traj_dim]
        # 情況1: arr = [T, D]
        if arr.ndim == 2:
            T, D = arr.shape
            # D 不符時，先簡單處理：截斷/補0（你也可以改成線性投影，但先別）
            if D > self.traj_dim:
                arr = arr[:, :self.traj_dim]
                D = self.traj_dim
            elif D < self.traj_dim:
                pad = np.zeros((T, self.traj_dim - D), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)

            # T 對齊 seq_len：截斷或補0
            if T >= self.seq_len:
                arr = arr[:self.seq_len]
            else:
                pad = np.zeros((self.seq_len - T, self.traj_dim), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)

        else:
            # 其他形狀先直接 fallback（避免 silent bug）
            if self.traj_missing == "error":
                raise ValueError(f"[traj] unexpected shape {arr.shape} in {traj_path}")
            arr = np.zeros((self.seq_len, self.traj_dim), dtype=np.float32)
            
        # 讓 traj 變成獨立可安全堆疊的 tensor（避免 DataLoader collate resize 問題）
        traj = torch.from_numpy(arr).to(torch.float32).contiguous().clone()

        # 保險起見，video 也做 contiguous（很多 video loader 會 permute）
        if isinstance(video, torch.Tensor) and not video.is_contiguous():
            video = video.contiguous().clone()
        
        #print('[DBG] type_video:',type(video))
        #print('[DBG] video_shape:',video.shape)
        #print('[DBG] video.is_contiguous:',video.is_contiguous())

        #print('[DBG] type_traj:',type(traj))
        #print('[DBG] traj_shape:',traj.shape)
        #print('[DBG] traj.is_contiguous:',traj.is_contiguous())

        return (video, traj), target
            
    def __len__(self):
        # 该Dataset的长度就是在该image_label下的image_name数量
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        # 返回的是一个列表 列表长self.n_episodes
        # 每个元素都是 先对0~self.n_classes随机排列 取前self.n_way个
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


