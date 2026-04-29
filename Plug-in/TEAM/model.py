import timm
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import extract_class_indices
from model_util import Discriminative_Pattern_Matching
from model_util import Discriminative_Pattern_Matching_without_sim

class TrajEncoder(nn.Module):
    """
    Input :
      traj: [N, T, D_in]
    Output:
      emb : [N, T, C]
    """
    def __init__(self, d_in=64, d_model=256, n_heads=4, dropout=0.0):
        super().__init__()

        self.proj = nn.Linear(d_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [N, T, D]
        x = self.proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x   # [N, T, C]

def build_traj_proto_by_class(spt_traj_emb, spt_labels):
    """
    spt_traj_emb: [Nspt, T, C]
    spt_labels  : [Nspt]
    return:
      proto: [way, T, C]
    """
    unique_labels = torch.unique(spt_labels)
    proto_list = []

    for c in unique_labels:
        idx = extract_class_indices(spt_labels, c)
        cls_emb = torch.index_select(spt_traj_emb, 0, idx)   # [shot, T, C]
        cls_proto = cls_emb.mean(dim=0)                      # [T, C]
        proto_list.append(cls_proto)

    proto = torch.stack(proto_list, dim=0)  # [way, T, C]
    return proto

def ensure_traj_shape(traj, seq_len):
    """
    Accept:
      - [N, T, D]
      - [N*T, D]  (same style as images flattened)
    Return:
      - [N, seq_len, D]
    """
    if traj is None:
        return None
    if traj.ndim == 3:
        return traj
    if traj.ndim == 2:
        # assume flattened [N*T, D]
        NT, D = traj.shape
        assert NT % seq_len == 0, f"traj first dim {NT} not divisible by seq_len {seq_len}"
        N = NT // seq_len
        return traj.reshape(N, seq_len, D)
    raise ValueError(f"Unsupported traj shape: {traj.shape}")

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    dists: [Nq, Ns, Tq, Ts]
    return: [Nq, Ns]
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    for l in range(1, dists.shape[2]):
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, 0] / lbda) +
            torch.exp(-cum_dists[:, :, l - 1, 1] / lbda) +
            torch.exp(-cum_dists[:, :, l, 0] / lbda)
        )

        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(-cum_dists[:, :, l - 1, m - 1] / lbda) +
                torch.exp(-cum_dists[:, :, l, m - 1] / lbda)
            )

        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(-cum_dists[:, :, l - 1, -2] / lbda) +
            torch.exp(-cum_dists[:, :, l - 1, -1] / lbda) +
            torch.exp(-cum_dists[:, :, l, -2] / lbda)
        )

    return cum_dists[:, :, -1, -1]

def traj_otam_distance(traj_q, traj_s, lbda=0.5):
    """
    traj_q: [Nq, T, C]
    traj_s: [Ns, T, C]
    return: [Nq, Ns]
    """
    traj_q = F.normalize(traj_q, dim=-1)
    traj_s = F.normalize(traj_s, dim=-1)

    sim = torch.einsum("qtc,skc->qstk", traj_q, traj_s)  # [Nq, Ns, Tq, Ts]
    dists = 1.0 - sim

    cum1 = OTAM_cum_dist_v2(dists, lbda=lbda)
    cum2 = OTAM_cum_dist_v2(rearrange(dists, "q s t k -> q s k t"), lbda=lbda)
    return cum1 + cum2

class CNN_FSHead(nn.Module):
    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.train()
        self.args = args

        last_layer_idx = -2

        if args.backbone == "ResNet":
            backbone = models.resnet50(pretrained=True)
            backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])
            self.backbone = backbone
            self.mid_dim = 2048
        elif args.backbone == "ViT":
            backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            self.backbone = backbone
            self.mid_dim = 768

        self.seq_len = self.args.seq_len
        self.agg_num = self.args.agg_num

    def get_feats(self, spt, tar):
        if self.args.backbone == "ResNet":
            spt = self.backbone(spt)
            tar = self.backbone(tar)
        elif self.args.backbone == "ViT":
            spt = self.backbone(spt).unsqueeze(dim=-1).unsqueeze(dim=-1)
            tar = self.backbone(tar).unsqueeze(dim=-1).unsqueeze(dim=-1)

        return spt, tar

    def get_other(self, x):
        cls_num = x.size(1)
        other = []
        for i in range(cls_num):
            other.append(torch.cat((x[:, 0:i], x[:, i+1:cls_num]), dim=1))
        other = torch.stack(other, dim=1)

        return other

    def get_other_weight(self, x):
        agg_num = x.size(2)
        other = []
        for i in range(agg_num):
            other.append(torch.cat((x[:, :, 0:i], x[:, :, i+1:agg_num]), dim=2))
        other = torch.stack(other, dim=2)

        return other

    def pooling(self, spt, tar):
        spt = spt.reshape(-1, self.args.seq_len, *list(spt.shape[-3:]))
        tar = tar.reshape(-1, self.args.seq_len, *list(tar.shape[-3:]))

        spt = spt.mean(dim=-1).mean(dim=-1)
        tar = tar.mean(dim=-1).mean(dim=-1)

        return spt, tar

    def reshape(self, spt, tar, spt_labels):
        unique_labels = torch.unique(spt_labels)
        spt = [torch.index_select(spt, 0, extract_class_indices(spt_labels, c)) for c in unique_labels]
        spt = torch.stack(spt, dim=0)
        spt = spt.unsqueeze(dim=0)
        tar = tar.unsqueeze(dim=1).unsqueeze(dim=1)

        return spt, tar

    def get_spt_sim(self, spt):
        _, cls_num, t, c = spt.shape
        spt_other = self.get_other(spt)
        spt_other = spt_other.reshape(1, cls_num, cls_num - 1, t, c)
        sim = F.cosine_similarity(spt.unsqueeze(dim=2), spt_other, dim=-1)

        return sim

    def forward(self, spt_images, spt_labels, tar_images):
        raise NotImplementedError

    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

    def loss(self, task_dict, model_dict):
        loss = {'L': F.cross_entropy(model_dict['logits'], task_dict["target_labels"].long())}

        return loss


class TEAM_pos(CNN_FSHead):
    def __init__(self, args):
        super(TEAM_pos, self).__init__(args)
        n_head = 4
        self.DPM = Discriminative_Pattern_Matching(self.agg_num, 
                                                   d_model=self.mid_dim, 
                                                   n_head=n_head, 
                                                   d_k=self.mid_dim // n_head, 
                                                   d_v=self.mid_dim // n_head,
                                                   seq_len=self.seq_len)
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()

        # ---- traj late fusion configs ----
        self.use_traj = bool(getattr(args, "use_traj", False))
        self.traj_lam = float(getattr(args, "traj_lam", 0.0))  # 0.0 = disable
        self.traj_dim = int(getattr(args, "traj_dim", 64))
        self.traj_mid_dim = int(getattr(args, "traj_mid_dim", 256))
        self.traj_heads = int(getattr(args, "traj_heads", 4))
        self.traj_lbda = float(getattr(args, "traj_lbda", 0.3))
        self.traj_dropout = float(getattr(args, "traj_dropout", 0.1))

        print('[TEAM_pos] use=',self.use_traj)
        print('[TEAM_pos] dim=',self.traj_dim)
        print('[TEAM_pos] lam=',self.traj_lam)


        if self.use_traj and self.traj_lam > 0:
            self.traj_encoder = TrajEncoder(
                d_in=self.traj_dim,
                d_model=self.traj_mid_dim,
                n_heads=self.traj_heads,
                dropout=self.traj_dropout,
            )
            print('Traj_encoder used.')
        else:
            self.traj_encoder = None

    def get_cum_dists_pos(self, spt, tar, weight=None):
        sim = F.cosine_similarity(spt, tar, dim=-1)
        dists = self.dropout(1 - sim)
        if weight is None:
            cum_dists = dists.sum(dim=-1)
        else:
            cum_dists = (dists * weight).sum(dim=-1)

        return cum_dists

    def get_cum_dists_neg(self, spt, tar, weight=None):
        sim = F.cosine_similarity(spt, tar, dim=-1)
        sim_other = self.get_other(sim)
        dists = self.dropout(1 - sim_other)
        if weight is None:
            cum_dists = dists.sum(dim=-1)
        else:
            cum_dists = (dists * weight.unsqueeze(dim=2)).sum(dim=-1)
        cum_dists = cum_dists.max(dim=-1)[0]

        return cum_dists

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_pos, tar_pos)

        return cum_dists_pos, cum_dists_pos, cum_dists_pos

    def forward(self, spt, spt_labels, tar, tar_labels=None, spt_traj=None, tar_traj=None):
        spt, tar = self.get_feats(spt, tar)
        spt, tar = self.pooling(spt, tar)
        spt, tar = self.reshape(spt, tar, spt_labels)
        tar_num, (_, cls_num, ins_num, t, c), agg_num = tar.size(0), spt.shape, self.agg_num
        spt, tar = spt.mean(dim=2), tar.mean(dim=2)

        spt_pos, spt_neg = self.DPM(spt)
        spt_pos, spt_neg = self.relu(spt_pos), self.relu(spt_neg)

        tar_pos, tar_neg = self.DPM(tar)
        tar_pos, tar_neg = self.relu(tar_pos), self.relu(tar_neg)

        spt_pos_sim, spt_neg_sim = self.get_spt_sim(spt_pos), self.get_spt_sim(spt_neg)
        spt_other = self.get_other(spt)
        spt_disc_pos, spt_disc_neg = self.DPM(spt, spt_other, spt_pos_sim, spt_neg_sim)
        spt_disc_pos, spt_disc_neg = self.relu(spt_disc_pos), self.relu(spt_disc_neg)
        spt_disc_pos, spt_disc_neg = spt_disc_pos.mean(dim=2), spt_disc_neg.mean(dim=2)

        cum_dists, cum_dists_pos, cum_dists_neg = self.get_cum_dists(
            spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg
        )

        # ---- NEW traj late fusion: seq encoder + OTAM ----
        cum_dists_traj = None
        if (self.traj_encoder is not None) and (spt_traj is not None) and (tar_traj is not None):
            # [Nspt, T, D] / [Ntar, T, D]
            spt_traj = ensure_traj_shape(spt_traj, self.seq_len).to(cum_dists.device)
            tar_traj = ensure_traj_shape(tar_traj, self.seq_len).to(cum_dists.device)

            # encode -> [Nspt, T, C], [Ntar, T, C]
            spt_traj_emb = self.traj_encoder(spt_traj)
            tar_traj_emb = self.traj_encoder(tar_traj)

            # class proto in seq space -> [way, T, C]
            spt_traj_proto = build_traj_proto_by_class(spt_traj_emb, spt_labels)

            # OTAM distance -> [Ntar, way]
            cum_dists_traj = traj_otam_distance(
                tar_traj_emb,
                spt_traj_proto,
                lbda=self.traj_lbda
            )

            lam = self.traj_lam
            cum_dists     = cum_dists     + lam * cum_dists_traj
            cum_dists_pos = cum_dists_pos + lam * cum_dists_traj
            cum_dists_neg = cum_dists_neg + lam * cum_dists_traj

        return_dict = {
            'logits': -cum_dists,
            'logits_pos': -cum_dists_pos,
            'logits_neg': -cum_dists_neg
        }

        if cum_dists_traj is not None:
            return_dict['logits_traj'] = -cum_dists_traj

        return return_dict
    
    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long())}

        return loss


class TEAM_neg_with_pos_loss(TEAM_pos):
    def __init__(self, args):
        super(TEAM_neg_with_pos_loss, self).__init__(args)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_pos, tar_neg) + self.get_cum_dists_neg(spt_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        return cum_dists_neg, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


class TEAM_pos_neg(TEAM_pos):
    def __init__(self, args):
        super(TEAM_pos_neg, self).__init__(args)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_pos, tar_neg) + self.get_cum_dists_neg(spt_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        cum_dists = cum_dists_pos + cum_dists_neg

        return cum_dists, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


class TEAM_disc_without_sim(TEAM_pos):
    def __init__(self, args):
        super(TEAM_disc_without_sim, self).__init__(args)
        self.DPM = Discriminative_Pattern_Matching_without_sim(self.agg_num, seq_len=self.seq_len)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_disc_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_disc_pos, tar_neg) + self.get_cum_dists_neg(spt_disc_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        cum_dists = cum_dists_pos + cum_dists_neg

        return cum_dists, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


class TEAM(TEAM_pos):
    def __init__(self, args):
        super(TEAM, self).__init__(args)

    def get_cum_dists(self, spt_disc_pos, spt_disc_neg, spt_pos, spt_neg, tar_pos, tar_neg):
        cum_dists_pos = self.get_cum_dists_pos(spt_disc_pos, tar_pos)

        cum_dists_neg = self.get_cum_dists_neg(spt_disc_pos, tar_neg) + self.get_cum_dists_neg(spt_disc_neg, tar_pos)
        cum_dists_neg = cum_dists_neg / 2

        cum_dists = cum_dists_pos + cum_dists_neg

        return cum_dists, cum_dists_pos, cum_dists_neg

    def loss(self, task_dict, model_dict):
        loss = {'L_pos': F.cross_entropy(model_dict['logits_pos'], task_dict["target_labels"].long()),
                'L_neg': F.cross_entropy(model_dict['logits_neg'], task_dict["target_labels"].long())}

        return loss


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.way = 3
            self.shot = 1
            self.query_per_class = 3
            self.trans_dropout = 0.1
            self.seq_len = 4
            self.img_size = 224
            self.backbone = "ViT"
            self.num_gpus = 1
            self.cls_num = 10
            self.agg_num = 30
            self.fea_num = 30
            self.repeat = 2
            self.coefficient = 0.1

            self.lam = 0.1
            self.alpha = 0.1

    args = ArgsObject()
    torch.manual_seed(0)

    device = 'cpu'
    model = TEAM(args).to(device)

    support_imgs = torch.rand(args.way * args.shot, args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class, args.seq_len, 3, args.img_size, args.img_size).to(device)

    support_imgs = support_imgs.reshape(args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size)
    target_imgs = target_imgs.reshape(args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size)

    support_labels = torch.tensor([n for n in range(args.way)] * args.shot).to(device)
    target_labels = torch.tensor([n for n in range(args.way)] * args.query_per_class).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    task_dict = {}
    task_dict["support_set"] = support_imgs
    task_dict["support_labels"] = support_labels
    task_dict["target_set"] = target_imgs
    task_dict["target_labels"] = target_labels

    model_dict = model(support_imgs, support_labels, target_imgs, target_labels)

    loss = model.loss(task_dict, model_dict)
    print(loss)
