import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert r.dtype == DTYPE
            assert w.dtype == DTYPE
            assert k.dtype == DTYPE
            assert v.dtype == DTYPE
            assert a.dtype == DTYPE
            assert b.dtype == DTYPE
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert w.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()
            assert b.is_contiguous()
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y
    
    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, w, k, v, a, b = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            ga = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gb = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            torch.ops.wkv7.backward(B, T, C, H, r, k, v, w, ab, gy, gr, gk, gv, gw, ga, gb)
            ga = torch.sum(ga, 0).view(H, C//H)
            gb = torch.sum(gb, 0).view(H, C//H)

class TimeMix_7(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ddd = torch.empty(1, 1, args.n_embd)
            self.time_maa_x = nn.Parameter(ddd)
            self.time_maa_r = nn.Parameter(ddd)
            self.time_maa_w = nn.Parameter(ddd)
            self.time_maa_k = nn.Parameter(ddd)
            self.time_maa_v = nn.Parameter(ddd)
            self.time_maa_a = nn.Parameter(ddd)
            self.time_maa_g = nn.Parameter(ddd)

            decay_speed = torch.empty(args.dim_att)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            self.time_faaaa = nn.Parameter(torch.empty(self.n_head,self.head_size))
            self.time_aaaaa = nn.Parameter(torch.empty(1,1,args.dim_att))

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.empty(args.n_embd, D_MIX_LORA*6))
            self.time_maa_w2 = nn.Parameter(torch.empty(6, D_MIX_LORA, args.n_embd))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.empty(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, args.dim_att))

            D_AAA_LORA = 64
            self.time_aaa_w1 = nn.Parameter(torch.empty(args.n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(torch.empty(D_AAA_LORA, args.dim_att))

            D_KKK_LORA = 64
            self.time_kkk_w1 = nn.Parameter(torch.empty(args.n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(torch.empty(D_KKK_LORA, args.dim_att))

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(torch.empty(args.n_embd, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(torch.empty(D_GATE_LORA, args.dim_att))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
            self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 6, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, -1)
        mr, mw, mk, mv, ma, mg = xxx.unbind(dim=0)

        xr = x + xx * (self.time_maa_r + mr)
        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xa = x + xx * (self.time_maa_a + ma)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        w = -F.softplus(-(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        g = torch.tanh(xg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk, dim=-1, p=2.0)
        a = torch.sigmoid( self.time_aaaaa + (xa @ self.time_aaa_w1) @ self.time_aaa_w2 ) * 2.0 # a is "in-context learning rate"

        k = k * torch.clamp(w*0.5,max=0).exp()
        x = RUN_CUDA_RWKV7(r, w, k, v, -kk, kk*a)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        
        x = self.output(x * g)
        return x

class ChannelMix_6(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ddd = torch.empty(1, 1, args.n_embd)
            self.time_maa_k = nn.Parameter(ddd)
            self.time_maa_r = nn.Parameter(ddd)

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class FeedForward():
    pass

class RWKV_Block(nn.module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix_7(args, layer_id)
        self.ffn = ChannelMix_6(args, layer_id)
        
    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        if RESCALE_LAYER > 0:
            if (self.layer_id+1) % RESCALE_LAYER == 0:
                x = x / 2
        # if self.layer_id == args.n_layer-1:
        #     print(torch.min(x).item(), torch.max(x).item())

        return x

class FinalLayer(nn.module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT_RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        x = self.head(x)

        return x
