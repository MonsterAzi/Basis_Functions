# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def justnorm(x, dim=-1):
        return F.normalize(x, p=2, dim=dim)


class TimestepEmbedder(nn.Module):
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
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


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = 1 / math.sqrt(dim)
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(dim, dtype=torch.float32))

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.n_heads, self.head_dim)
        xq = sqk * justnorm(xq)
        xk = sqk * justnorm(xk)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
            scale=math.sqrt(self.head_dim),
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        self.sqk_init_value = 1.0
        self.sqk_init_scaling = 1 / math.sqrt(dim)
        self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(dim, dtype=torch.float32))

    def forward(self, x, cond_input):
        bsz, seqlen, _ = x.shape
        cond_seqlen, _ = cond_input.shape

        xq, xk, xv = self.wq(x), self.wk(cond_input), self.wv(cond_input)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(1, cond_seqlen, self.n_heads, self.head_dim)
        xv = xv.view(1, cond_seqlen, self.n_heads, self.head_dim)
        
        sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.n_heads, self.head_dim)
        xq = sqk * justnorm(xq)
        xk = sqk * justnorm(xk)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
            scale=math.sqrt(self.head_dim),
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.dim = dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        self.su = torch.nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))
        self.sv = torch.nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))

    def _forward_silu_gating(self, x1, x3):
        return torch.sin(x1 * (self.dim ** 0.5) * self.sv) * x3 * self.su

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.cross_attention = CrossAttention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        
        self.attn_alpha_init_value = 0.05
        self.attn_alpha_init_scaling = 1 / math.sqrt(dim)
        self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(dim, dtype=torch.float32))
        
        self.cross_alpha_init_value = 0.05
        self.cross_alpha_init_scaling = 1 / math.sqrt(dim)
        self.cross_alpha = torch.nn.Parameter(self.cross_alpha_init_scaling*torch.ones(dim, dtype=torch.float32))

        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1 / math.sqrt(dim)
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(dim, dtype=torch.float32))

    def forward(self, x, freqs_cis, cond_input):
        lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
        lr = torch.abs(lr)
        x_att = self.attention(x, freqs_cis)
        A_norm = justnorm(x)
        B_norm = justnorm(x_att)
        res = A_norm + lr * (B_norm - A_norm)
        x = justnorm(res)


        lr = self.cross_alpha * (self.cross_alpha_init_value / self.cross_alpha_init_scaling)
        lr = torch.abs(lr)
        x_cross = self.cross_attention(x, cond_input)
        A_norm = justnorm(x)
        B_norm = justnorm(x_cross)
        res = A_norm + lr * (B_norm - A_norm)
        x = justnorm(res)
        
        lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
        lr = torch.abs(lr)
        x_mlp = self.feed_forward(x)
        A_norm = justnorm(x)
        B_norm = justnorm(x_mlp)
        res = A_norm + lr * (B_norm - A_norm)
        x = justnorm(res)

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=False
        )
        
        self.sz_init_value = 1.00
        self.sz_init_scaling = 1 / math.sqrt(hidden_size)
        self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(patch_size * patch_size * out_channels, dtype=torch.float32))

    def forward(self, x):
        sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
        x = self.linear(x)
        x = x * sz
        return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
    ):
        super().__init__()

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=False)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = self.precompute_freqs_cis(dim // n_heads, 4096)
        
        # Weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(self.dim))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)
        x = self.patchify(x)
        x = self.x_embedder(x)
        x = justnorm(x)

        t = self.t_embedder(t)  # (N, D)
        t = justnorm(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        y = justnorm(y)
        cond_input = t.to(x.dtype) + y.to(x.dtype)
        cond_input = justnorm(cond_input)

        for i, layer in enumerate(self.layers):
            x = layer(x, self.freqs_cis[: x.size(1)], cond_input)

        x = self.final_layer(x)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)


if __name__ == "__main__":
    model = DiT_Llama_600M_patch2()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out = model(x, t, y)
        print(out.shape)
        out = model.forward_with_cfg(x, t, y, 0.5)
        print(out.shape)