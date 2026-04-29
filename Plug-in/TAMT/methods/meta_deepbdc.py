import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .template import MetaTemplate
from .bdc_module import BDC
from sklearn.linear_model import LogisticRegression

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]

class TrajEncoder(nn.Module):
    """
    Fixed single-layer trajectory encoder.
    Input : [N, T, D_in]
    Output: [N, T, C]
    """
    def __init__(self, d_in=64, d_model=256, n_heads=4):
        super().__init__()

        self.proj = nn.Linear(d_in, d_model)
        #print(self.proj)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x
    
def ensure_traj_shape(traj, seq_len):
    """
    Accepts traj as:
    - [N, T, D]
    - [N*T, D]  (your TEAM loader cat after dim0)
    Returns [N, seq_len, D]
    """
    if traj is None:
        return None
    if traj.ndim == 3:
        # [N,T,D]
        return traj
    if traj.ndim == 2:
        # [N*T, D] -> [N, T, D]
        NT, D = traj.shape
        assert NT % seq_len == 0, f"traj first dim {NT} not divisible by seq_len={seq_len}"
        N = NT // seq_len
        return traj.view(N, seq_len, D)
    raise ValueError(f"Unsupported traj shape: {traj.shape}")


def traj_otam_distance(traj_q, traj_s, lbda=0.5, eps=1e-2):
    """
    traj_q: [Nq, T, C]   (encoded, per-frame)
    traj_s: [Ns, T, C]
    return: [Nq, Ns] cumulative OTAM distance (bidirectional)
    """
    # cosine sim per-frame: [Nq, Ns, Tq, Ts]
    # normalize to stabilize cosine
    traj_q = F.normalize(traj_q, dim=-1)
    traj_s = F.normalize(traj_s, dim=-1)

    # sim: [Nq, Ns, Tq, Ts]
    sim = torch.einsum("qtc,skc->qstk", traj_q, traj_s)  # q:query, s:support, t:time_q, k:time_s

    dists = 1.0 - sim  # cosine distance

    # OTAM expects [nq, ns, tq, ts]
    cum1 = OTAM_cum_dist_v2(dists, lbda=lbda)
    cum2 = OTAM_cum_dist_v2(rearrange(dists, "q s t k -> q s k t"), lbda=lbda)
    return cum1 + cum2

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.class_way='add'
        self.method = params.method

        # ---- traj late fusion config ----
        self.use_traj = bool(getattr(params, "use_traj", False))
        self.traj_lam = float(getattr(params, "traj_lam", 1.0))
        self.seq_len  = int(getattr(params, "seq_len", 8))
        self.traj_dim = int(getattr(params, "traj_dim", 64))
        self.traj_lbda = float(getattr(params, "traj_lbda", 0.5))

        self.traj_encoder = None
        if self.use_traj:
            traj_mid_dim = int(getattr(params, "traj_mid_dim", 256))
            traj_heads   = int(getattr(params, "traj_heads", 4))

            self.traj_encoder = TrajEncoder(
                d_in=self.traj_dim,
                d_model=traj_mid_dim,
                n_heads=traj_heads,
            )
            
    def set_forward(self, x, is_feature=False):
        # ---- unpack input ----
        x_traj = None
        if isinstance(x, (list, tuple)) and len(x) == 2 and torch.is_tensor(x[0]):
            x_video, x_traj = x[0], x[1]
        else:
            x_video = x

        assert torch.is_tensor(x_video), f"x_video must be Tensor, got {type(x_video)}"

        # ---- image branch ----
        z_support, z_query, _, _ = self.parse_feature(x_video, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        scores = self.metric(z_query, z_proto)

        # ---- traj branch ----
        if (self.traj_encoder is not None) and (x_traj is not None):
            assert torch.is_tensor(x_traj) and x_traj.ndim == 4, \
                f"x_traj shape wrong: {getattr(x_traj, 'shape', None)}"

            # x_traj: [way, shot+query, T, D]
            spt_traj = x_traj[:, :self.n_support]   # [way, support, T, D]
            qry_traj = x_traj[:, self.n_support:]   # [way, query,   T, D]

            Tt, Dt = x_traj.size(2), x_traj.size(3)
            Ns = self.n_way * self.n_support
            Nq = self.n_way * self.n_query

            spt_traj = spt_traj.contiguous().view(Ns, Tt, Dt)
            qry_traj = qry_traj.contiguous().view(Nq, Tt, Dt)

            spt_traj = ensure_traj_shape(spt_traj, self.seq_len).to(z_proto.device)
            qry_traj = ensure_traj_shape(qry_traj, self.seq_len).to(z_proto.device)

            # encode
            spt_traj_emb = self.traj_encoder(spt_traj)   # [Ns, T, C]
            qry_traj_emb = self.traj_encoder(qry_traj)   # [Nq, T, C]

            # class prototype in trajectory space
            spt_traj_emb = spt_traj_emb.view(self.n_way, self.n_support, Tt, -1).mean(1)  # [way, T, C]

            # OTAM distance
            D_traj = traj_otam_distance(qry_traj_emb, spt_traj_emb, lbda=self.traj_lbda)  # [Nq, way]
            scores_traj = -D_traj

            # fuse
            scores = scores + self.traj_lam * scores_traj

        return scores
    
    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def forward_meta_val_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.val_n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.val_n_way), self.n_query)
        scores = self.forward_meta_val(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        # print('x',x.shape) #x torch.Size([80, 32896])
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        # print('x',x.shape) #x torch.Size([80, 5, 32896])

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        # print('score',score.shape) #score torch.Size([80, 5])
        return score
    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score
