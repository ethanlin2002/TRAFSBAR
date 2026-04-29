#!/usr/bin/env python3
# -----------------------------------------------
# Written by Qizhong Tan
# -----------------------------------------------

import torch
import clip
import math
import utils.logging as logging
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from collections import OrderedDict
import torchvision

from utils.registry import Registry

HEAD_REGISTRY = Registry("Head")

logger = logging.get_logger(__name__)


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x


def OTAM_dist(dists, lbda=0.5):
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(- cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(- cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class ResNet_DeformAttention(nn.Module):
    def __init__(self, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_k = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_v = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_out = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3

        return ref

    def forward(self, x):
        B, C, T, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # B * g 3 Tp Hp Wp
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp
        # logger.info('{}x{}x{}={}'.format(Tp, Hp, Wp, n_sample))

        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_sampled = F.grid_sample(input=x.reshape(B * self.groups, self.group_channels, T, H, W),
                                  grid=pos[..., (2, 1, 0)],  # z, y, x -> x, y, z
                                  mode='bilinear', align_corners=True)  # B * g, Cg, Tp, Hp, Wp

        x_sampled = x_sampled.reshape(B, C, 1, 1, n_sample)
        q = q.reshape(B * self.heads, self.head_channels, T * H * W)
        k = self.proj_k(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)

        attn = einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        attn = F.softmax(attn, dim=-1)

        out = einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, T, H, W)
        out = self.proj_out(out)

        return out


class ResNet_Vanilla_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = self.relu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_ST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))
        self.conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=self.adapter_channels)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = self.conv(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_DST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))

        self.s_conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=self.adapter_channels, bias=False)
        self.s_bn = nn.BatchNorm3d(num_features=self.adapter_channels)

        self.t_conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=self.adapter_channels, bias=False)
        self.t_bn = nn.BatchNorm3d(num_features=self.adapter_channels)

        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        # Spatial Pathway
        xs = self.s_bn(self.s_conv(x))

        # Temporal Pathway
        xt = self.t_bn(self.t_conv(x))

        x = (xs + xt) / 2
        x = self.relu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_D2ST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)
        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))

        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        self.s_ln = LayerNormProxy(dim=self.adapter_channels)
        self.t_ln = LayerNormProxy(dim=self.adapter_channels)
        if dim == self.args.ADAPTER.WIDTH // 8:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=1, groups=1, kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=1, groups=1, kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 4:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=2, groups=2, kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=2, groups=2, kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 2:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=4, groups=4, kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=4, groups=4, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        else:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=8, groups=8, kernel_size=(4, 4, 4), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=8, groups=8, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        self.gelu = nn.GELU()

        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = x + self.pos_embed(x)

        # Spatial Deformable Attention
        xs = x + self.s_attn(self.s_ln(x))

        # Temporal Deformable Attention
        xt = x + self.t_attn(self.t_ln(x))

        x = (xs + xt) / 2
        x = self.gelu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x

def resize_clip_positional_embedding(pos_embed_ckpt, pos_embed_model):
    """
    Resize CLIP ViT positional embedding.
    Example:
      ckpt : [197, 768] = 1 + 14*14
      model: [50, 768]  = 1 + 7*7
    """
    if pos_embed_ckpt.shape == pos_embed_model.shape:
        return pos_embed_ckpt

    cls_pos = pos_embed_ckpt[:1]      # [1, C]
    patch_pos = pos_embed_ckpt[1:]    # [old_H*old_W, C]

    old_num_patches = patch_pos.shape[0]
    new_num_patches = pos_embed_model.shape[0] - 1

    old_grid = int(math.sqrt(old_num_patches))
    new_grid = int(math.sqrt(new_num_patches))

    if old_grid * old_grid != old_num_patches:
        raise ValueError(f"Old pos_embed is not square: {old_num_patches}")

    if new_grid * new_grid != new_num_patches:
        raise ValueError(f"New pos_embed is not square: {new_num_patches}")

    C = patch_pos.shape[-1]

    patch_pos = patch_pos.reshape(1, old_grid, old_grid, C).permute(0, 3, 1, 2)

    patch_pos = F.interpolate(
        patch_pos,
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    )

    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, C)

    return torch.cat([cls_pos, patch_pos], dim=0)

@HEAD_REGISTRY.register()
class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.args = cfg
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        if self.args.ADAPTER.LAYERS == 18:
            backbone = torchvision.models.resnet18(pretrained=True)
        elif self.args.ADAPTER.LAYERS == 34:
            backbone = torchvision.models.resnet34(pretrained=True)
        elif self.args.ADAPTER.LAYERS == 50:
            backbone = torchvision.models.resnet50(pretrained=True)
        self.stage1 = nn.Sequential(*list(backbone.children())[:5])
        self.Adapter_1 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 8)
        self.stage2 = nn.Sequential(*list(backbone.children())[5])
        self.Adapter_2 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 4)
        self.stage3 = nn.Sequential(*list(backbone.children())[6])
        self.Adapter_3 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 2)
        self.stage4 = nn.Sequential(*list(backbone.children())[7])
        self.Adapter_4 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH)
        self.stage5 = nn.Sequential(*list(backbone.children())[8:-1])
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            self.classification_layer = nn.Linear(self.args.ADAPTER.WIDTH, int(self.args.TRAIN.NUM_CLASS))
        self.init_weights()

    def init_weights(self):
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def get_feat(self, x):
        x = self.stage1(x)
        x = self.Adapter_1(x)
        x = self.stage2(x)
        x = self.Adapter_2(x)
        x = self.stage3(x)
        x = self.Adapter_3(x)
        x = self.stage4(x)
        x = self.Adapter_4(x)
        x = self.stage5(x)
        return x.squeeze()

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        support_features = self.get_feat(support_images)
        query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)

        class_logits = None
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            class_logits = self.classification_layer(torch.cat([torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)], 0))

        support_features = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        # OTAM
        # class_dist = OTAM_dist(dist) + OTAM_dist(rearrange(dist, 'q s n m -> q s m n'))

        return_dict = {'logits': - class_dist, 'class_logits': class_logits}
        return return_dict


class ViT_DeformAttention(nn.Module):
    def __init__(self, cfg, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_k = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_v = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_out = nn.Linear(in_features=self.dim, out_features=self.dim)

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3

        return ref

    def forward(self, x):
        # hw+1 bt c
        n, BT, C = x.shape
        T = self.args.DATA.NUM_INPUT_FRAMES
        B = BT // T
        H = round(math.sqrt(n - 1))
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q[1:, :, :], '(h w) (b t) c -> b c t h w', h=H, t=T)
        q_off = rearrange(q_off, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # B * g 3 Tp Hp Wp
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp
        # logger.info('{}x{}x{}={}'.format(Tp, Hp, Wp, n_sample))

        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_sampled = rearrange(x[1:, :, :], '(h w) (b t) c -> b c t h w', h=H, t=T)
        x_sampled = rearrange(x_sampled, 'b (g c) t h w -> (b g) c t h w', g=self.groups)
        x_sampled = F.grid_sample(input=x_sampled, grid=pos[..., (2, 1, 0)], mode='bilinear', align_corners=True)  # B * g, Cg, Tp, Hp, Wp
        x_sampled = rearrange(x_sampled, '(b g) c t h w -> b (g c) t h w', g=self.groups)
        x_sampled = rearrange(x_sampled, 'b c t h w -> b (t h w) c')

        q = rearrange(q, 'n (b t) c -> b c (t n)', b=B)
        q = rearrange(q, 'b (h c) n -> (b h) c n', h=self.heads)

        k = self.proj_k(x_sampled)
        k = rearrange(k, 'b n (h c) -> (b h) c n', h=self.heads)

        v = self.proj_v(x_sampled)
        v = rearrange(v, 'b n (h c) -> (b h) c n', h=self.heads)

        attn = einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = einsum('b m n, b c n -> b c m', attn, v)
        out = rearrange(out, '(b h) c n -> b (h c) n', h=self.heads)
        out = rearrange(out, 'b c (t n) -> n (b t) c', t=T)
        out = self.proj_out(out)

        return out


class ViT_D2ST_Adapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.in_channels = cfg.ADAPTER.WIDTH
        self.out_channels = cfg.ADAPTER.WIDTH
        self.adapter_channels = int(cfg.ADAPTER.WIDTH * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Linear(in_features=self.in_channels, out_features=self.adapter_channels)
        self.gelu1 = nn.GELU()

        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        self.s_ln = nn.LayerNorm(normalized_shape=self.adapter_channels)
        self.s_attn = ViT_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
        self.t_ln = nn.LayerNorm(normalized_shape=self.adapter_channels)
        self.t_attn = ViT_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        self.gelu = nn.GELU()

        self.up = nn.Linear(in_features=self.adapter_channels, out_features=self.out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        # hw+1 bt c
        n, bt, c = x.shape
        H = round(math.sqrt(n - 1))
        x_in = x

        x = self.down(x)
        x = self.gelu1(x)

        cls = x[0, :, :].unsqueeze(0)
        x = x[1:, :, :]

        x = rearrange(x, '(h w) (b t) c -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES, h=H)
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c t h w -> (h w) (b t) c')

        x = torch.cat([cls, x], dim=0)

        # Spatial Deformable Attention
        xs = x + self.s_attn(self.s_ln(x))

        # Temporal Deformable Attention
        xt = x + self.t_attn(self.t_ln(x))

        x = (xs + xt) / 2
        x = self.gelu(x)

        x = self.up(x)
        x = self.gelu2(x)

        x += x_in
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.ADAPTER.WIDTH
        n_head = cfg.ADAPTER.HEADS
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.Adapter = ViT_D2ST_Adapter(cfg)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        # x shape [HW+1, BT, C]
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = self.Adapter(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(cfg) for _ in range(cfg.ADAPTER.LAYERS)])

    def forward(self, x):
        return self.resblocks(x)


class TrajEncoder(nn.Module):
    """
    Input : [N, T, D_in]
    Output: [N, T, D_model]
    Compatible with older PyTorch versions.
    """
    def __init__(self, d_in=6, d_model=768, n_heads=4):
        super().__init__()

        self.proj = nn.Linear(d_in, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=0.0,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=1,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [N, T, D_in]
        x = self.proj(x)  # [N, T, D_model]

        # older TransformerEncoder expects [T, N, D_model]
        x = x.permute(1, 0, 2).contiguous()

        x = self.encoder(x)

        # back to [N, T, D_model]
        x = x.permute(1, 0, 2).contiguous()

        x = self.norm(x)

        return x

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    dists: [N_query, N_way, T_query, T_proto]
    return: [N_query, N_way]
    """
    dists = F.pad(dists, (1, 1), "constant", 0)

    cum_dists = torch.zeros_like(dists)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, 0] / lbda)
            + torch.exp(-cum_dists[:, :, l - 1, 1] / lbda)
            + torch.exp(-cum_dists[:, :, l, 0] / lbda)
        )

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(-cum_dists[:, :, l - 1, m - 1] / lbda)
                + torch.exp(-cum_dists[:, :, l, m - 1] / lbda)
            )

        # last column
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, -2] / lbda)
            + torch.exp(-cum_dists[:, :, l - 1, -1] / lbda)
            + torch.exp(-cum_dists[:, :, l, -2] / lbda)
        )

    return cum_dists[:, :, -1, -1]

@HEAD_REGISTRY.register()
class ViT_CLIP(nn.Module):
    def __init__(self, cfg):
        super(ViT_CLIP, self).__init__()
        self.args = cfg
        self.pretrained = cfg.ADAPTER.PRETRAINED
        self.width = cfg.ADAPTER.WIDTH
        self.patch_size = cfg.ADAPTER.PATCH_SIZE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        scale = self.width ** -0.5
        self.layers = cfg.ADAPTER.LAYERS
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((cfg.DATA.TRAIN_CROP_SIZE // self.patch_size) ** 2 + 1, self.width))
        self.ln_pre = LayerNorm(self.width)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.width))
        self.transformer = Transformer(cfg)
        self.ln_post = LayerNorm(self.width)
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            self.classification_layer = nn.Linear(self.width, int(self.args.TRAIN.NUM_CLASS))
        self.init_weights()

        # ================= TRAJ =================
        self.use_traj = getattr(self.args.TRAIN, "USE_TRAJ", False)
        self.norm_dist = getattr(self.args.TRAJ, "NORM_DIST", False) if hasattr(self.args, "TRAJ") else False
        self.norm_mode = getattr(self.args.TRAJ, "NORM_MODE", "zscore") if hasattr(self.args, "TRAJ") else "zscore"

        if hasattr(self.args, "TRAJ"):
            self.traj_lam = getattr(self.args.TRAJ, "LAM", 0.3)
            self.traj_lbda = getattr(self.args.TRAJ, "LBDA", 0.3)

        if self.use_traj:
            if hasattr(self.args, "TRAJ"):
                self.traj_dim = getattr(self.args.TRAJ, "DIM", 6)
                self.traj_mid_dim = getattr(self.args.TRAJ, "MID_DIM", self.width)
                self.traj_heads = getattr(self.args.TRAJ, "HEADS", 4)

            self.traj_encoder = TrajEncoder(
                d_in=self.traj_dim,
                d_model=self.traj_mid_dim,
                n_heads=self.traj_heads,
            )

    def init_weights(self):
        logger.info(f'load model from: {self.pretrained}')

        # Load OpenAI CLIP pretrained weights
        clip_model, _ = clip.load(self.pretrained, device="cpu")
        pretrain_dict = clip_model.visual.state_dict()
        del clip_model

        if 'proj' in pretrain_dict:
            del pretrain_dict['proj']

        # resize positional embedding for 112 crop
        if "positional_embedding" in pretrain_dict:
            if pretrain_dict["positional_embedding"].shape != self.positional_embedding.shape:
                logger.info(
                    "Resize positional_embedding: "
                    f"ckpt={pretrain_dict['positional_embedding'].shape}, "
                    f"model={self.positional_embedding.shape}"
                )

                pretrain_dict["positional_embedding"] = resize_clip_positional_embedding(
                    pretrain_dict["positional_embedding"],
                    self.positional_embedding,
                )

        msg = self.load_state_dict(pretrain_dict, strict=False)

        logger.info('Missing keys: {}'.format(msg.missing_keys))
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        torch.cuda.empty_cache()

        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def get_feat(self, x):
        x = self.conv1(x)  # b*t c h w
        x = rearrange(x, 'b c h w -> b (h w) c')
        # b*t h*w+1 c
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        # n = h*w+1
        n = x.shape[1]

        x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x[:, 0, :]
        return x

    def _normalize_traj_shape(self, traj):
        """
        Accept:
        - [N, T, D]
        - [1, N, T, D]
        - [N, T * D]  # fallback, only if TRAJ.SEQ_LEN exists
        Return:
        - [N, T, D]
        """
        if traj.dim() == 4 and traj.shape[0] == 1:
            traj = traj.squeeze(0)

        if traj.dim() == 2:
            if not hasattr(self.args, "TRAJ") or not hasattr(self.args.TRAJ, "SEQ_LEN"):
                raise RuntimeError(
                    f"Got flattened traj shape {tuple(traj.shape)}, "
                    "but cfg.TRAJ.SEQ_LEN is not defined."
                )
            seq_len = int(self.args.TRAJ.SEQ_LEN)
            traj = traj.reshape(traj.shape[0], seq_len, -1)

        if traj.dim() != 3:
            raise RuntimeError(
                f"Expect traj shape [N, T, D], got {tuple(traj.shape)}"
            )

        return traj

    def compute_traj_otam_dists(self, support_traj, target_traj, support_labels):
        """
        support_traj : [N_support, T, D_traj]
        target_traj  : [N_query,   T, D_traj]
        support_labels: [N_support]

        return:
            class_dists_traj: [N_query, N_way]
        """
        support_traj = self._normalize_traj_shape(support_traj)
        target_traj = self._normalize_traj_shape(target_traj)

        unique_labels = torch.unique(support_labels)

        support_z = self.traj_encoder(support_traj.float())
        target_z = self.traj_encoder(target_traj.float())

        # mean over K-shot, but keep temporal dimension:
        # [N_support, T, C] -> [N_way, T, C]
        proto_seq = []
        for c in unique_labels:
            idx = self.extract_class_indices(support_labels, c)
            proto_seq.append(
                torch.mean(torch.index_select(support_z, 0, idx), dim=0)
            )

        proto_seq = torch.stack(proto_seq, dim=0)

        n_queries = target_z.shape[0]
        n_way = proto_seq.shape[0]
        Tq = target_z.shape[1]
        Tp = proto_seq.shape[1]

        # cosine distance between every query timestep and every prototype timestep
        q_flat = rearrange(target_z, "q t d -> (q t) d")
        p_flat = rearrange(proto_seq, "c t d -> (c t) d")

        sim = torch.matmul(
            F.normalize(q_flat, dim=1),
            F.normalize(p_flat, dim=1).transpose(0, 1),
        )

        dist = 1.0 - sim

        dists = rearrange(
            dist,
            "(q tq) (c tp) -> q c tq tp",
            q=n_queries,
            c=n_way,
            tq=Tq,
            tp=Tp,
        )

        # bi-directional OTAM
        d_q2p = OTAM_cum_dist_v2(dists, lbda=self.traj_lbda)
        d_p2q = OTAM_cum_dist_v2(dists.transpose(2, 3), lbda=self.traj_lbda)

        class_dists_traj = d_q2p + d_p2q

        return class_dists_traj
    
    def normalize_episode_dists(self, dists, mode="zscore", eps=1e-6):
        """
        dists: [N_query, N_way]
        normalize each query row over class dimension
        """
        if mode == "zscore":
            mean = dists.mean(dim=1, keepdim=True)
            std = dists.std(dim=1, keepdim=True, unbiased=False)
            return (dists - mean) / (std + eps)

        elif mode == "minmax":
            dmin = dists.min(dim=1, keepdim=True)[0]
            dmax = dists.max(dim=1, keepdim=True)[0]
            return (dists - dmin) / (dmax - dmin + eps)

        elif mode == "mean_center":
            mean = dists.mean(dim=1, keepdim=True)
            return dists - mean

        return dists
    
    def _debug_branch_predictions(
        self,
        class_dist_img,
        class_dist_traj,
        class_dist_final,
        support_labels,
        target_labels,
        prefix="[FUSE DBG]",
    ):
        """
        class_dist_img  : [N_query, N_way]
        class_dist_traj : [N_query, N_way]
        class_dist_final: [N_query, N_way]
        support_labels  : [N_support]
        target_labels   : [N_query]
        """

        if target_labels is None:
            return

        # class order is determined by unique support labels used to build prototypes
        unique_labels = torch.unique(support_labels)

        # distance -> pred index (smaller is better)
        pred_img_idx = torch.argmin(class_dist_img, dim=1)      # [N_query]
        pred_traj_idx = torch.argmin(class_dist_traj, dim=1)    # [N_query]
        pred_fuse_idx = torch.argmin(class_dist_final, dim=1)   # [N_query]

        # map prototype index back to true class label
        pred_img_label = unique_labels[pred_img_idx]
        pred_traj_label = unique_labels[pred_traj_idx]
        pred_fuse_label = unique_labels[pred_fuse_idx]

        gt_label = target_labels.view(-1)

        img_correct = (pred_img_label == gt_label)
        traj_correct = (pred_traj_label == gt_label)
        fuse_correct = (pred_fuse_label == gt_label)

        img_acc = img_correct.float().mean().item() * 100.0
        traj_acc = traj_correct.float().mean().item() * 100.0
        fuse_acc = fuse_correct.float().mean().item() * 100.0

        img_traj_same_top1 = (pred_img_label == pred_traj_label).float().mean().item() * 100.0
        img_wrong_traj_right = ((~img_correct) & traj_correct).float().mean().item() * 100.0
        img_right_traj_wrong = (img_correct & (~traj_correct)).float().mean().item() * 100.0
        fuse_beats_img = ((~img_correct) & fuse_correct).float().mean().item() * 100.0
        fuse_hurts_img = (img_correct & (~fuse_correct)).float().mean().item() * 100.0

        '''
        logger.info(
            "%s img=%.2f traj=%.2f fuse=%.2f same_top1=%.2f img_wrong_traj_right=%.2f img_right_traj_wrong=%.2f fuse_beats_img=%.2f fuse_hurts_img=%.2f",
            prefix,
            img_acc,
            traj_acc,
            fuse_acc,
            img_traj_same_top1,
            img_wrong_traj_right,
            img_right_traj_wrong,
            fuse_beats_img,
            fuse_hurts_img,
        )
        '''
        
        # optional: print a few raw examples
        '''
        logger.info(
            "%s gt=%s img=%s traj=%s fuse=%s",
            prefix,
            gt_label[:10].detach().cpu().tolist(),
            pred_img_label[:10].detach().cpu().tolist(),
            pred_traj_label[:10].detach().cpu().tolist(),
            pred_fuse_label[:10].detach().cpu().tolist(),
        )
        '''

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        support_features = self.get_feat(support_images)
        query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)

        class_logits = None
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            class_logits = self.classification_layer(torch.cat([torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)], 0))

        support_features = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        # OTAM
        # class_dist = OTAM_dist(dist) + OTAM_dist(rearrange(dist, 'q s n m -> q s m n'))

        class_dist_img = class_dist

        if self.use_traj:
            support_traj = inputs["support_traj"]
            target_traj = inputs["target_traj"]

            class_dist_traj = self.compute_traj_otam_dists(
                support_traj=support_traj,
                target_traj=target_traj,
                support_labels=support_labels,
            )
            '''
            logger.info(
                "[TRAJ DIST] mean=%.6f std=%.6f min=%.6f max=%.6f",
                class_dist_traj.mean().item(),
                class_dist_traj.std().item(),
                class_dist_traj.min().item(),
                class_dist_traj.max().item(),
            )
            logger.info(
                "[IMG  DIST] mean=%.6f std=%.6f min=%.6f max=%.6f",
                class_dist_img.mean().item(),
                class_dist_img.std().item(),
                class_dist_img.min().item(),
                class_dist_img.max().item(),
            )
            '''

            if self.norm_dist:
                class_dist_img_fuse = self.normalize_episode_dists(class_dist_img, mode=self.norm_mode)
                class_dist_traj_fuse = self.normalize_episode_dists(class_dist_traj, mode=self.norm_mode)
            else:
                class_dist_img_fuse = class_dist_img
                class_dist_traj_fuse = class_dist_traj

            '''
            logger.info(
                "[IMG norm] mean=%.6f std=%.6f min=%.6f max=%.6f",
                class_dist_img_fuse.mean().item(),
                class_dist_img_fuse.std().item(),
                class_dist_img_fuse.min().item(),
                class_dist_img_fuse.max().item(),
            )

            logger.info(
                "[TRAJ norm] mean=%.6f std=%.6f min=%.6f max=%.6f",
                class_dist_traj_fuse.mean().item(),
                class_dist_traj_fuse.std().item(),
                class_dist_traj_fuse.min().item(),
                class_dist_traj_fuse.max().item(),
            )
            '''

            class_dist_final = class_dist_img_fuse + self.traj_lam * class_dist_traj_fuse
            target_labels = inputs.get("target_labels", None)
            if target_labels is not None:
                self._debug_branch_predictions(
                    class_dist_img=class_dist_img,
                    class_dist_traj=class_dist_traj,
                    class_dist_final=class_dist_final,
                    support_labels=support_labels,
                    target_labels=target_labels,
                    prefix="[VAL DBG]",
                )
        else:
            class_dist_final = class_dist_img            

        return_dict = {
            "logits": -class_dist_final,
            "class_logits": class_logits,
        }

        return return_dict