# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class ModelState:
    def __init__(self):
        self.seq_pos = 0
        self.input_tokens_cache = torch.tensor([])
        self.k_cache = torch.tensor([])
        self.block_states:list[BlockState] = []

class TimeMixState:
    def __init__(self, wkv_state=torch.tensor([]), shift_state=torch.tensor([])):
        self.wkv_state = wkv_state
        self.shift_state = shift_state

class ChannelMixState:
    def __init__(self, shift_state=torch.tensor([])):
        self.shift_state = shift_state

class BlockState:
    def __init__(self, time_mix_state: TimeMixState, channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class Shared:
    def __init__(self):
        self.angles = torch.tensor([])
        self.bias_mask = torch.tensor([])


def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def generate_binary_rotary_embedding(max_seqlen:int, dim:int, scale:float=1):
    arange = torch.arange(dim // 2)
    angular_velocity = math.pi * (2.0 ** -arange) / scale # fastest velocity will rotate fully in two steps
    angular_velocity[int(math.log2(max_seqlen)):] = 0.0 # don't supply velocities slower than the one that will get a single full rotation across the seqlen
    #angular_velocity[20:] = 0.0 # don't supply velocities slower than the one that will get a single full rotation across 1024k ctxlen
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def apply_rotary_embedding(q, k, angles, seq_dim:int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    if angles.size(0) == 0:
        return q, k
    
    q_dtype, k_dtype = q.dtype, k.dtype
    L = q.size(seq_dim)
    q_angles = angles[-L:].view(1, 1, L, angles.size(1))
    if q.ndim == 3:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), -1, 2)) * q_angles
    else:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), q.size(2), -1, 2)) * q_angles

    L = k.size(seq_dim)
    k_angles = angles[-L:].view(1, 1, L, angles.size(1))
    if k.ndim == 3:
        k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), -1, 2)) * k_angles
    else:
        k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), k.size(2), -1, 2)) * k_angles

    return torch.view_as_real(q).flatten(3).to(q_dtype), torch.view_as_real(k).flatten(3).to(k_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, max_sequence_length:int, dim:int, seq_dim:int = -2, theta:float = 10000):
        super().__init__()
        self.angles = generate_rotary_embedding(max_sequence_length, dim, theta)
        self.seq_dim = seq_dim

    def forward(self, q, k):
        return apply_rotary_embedding(q, k, self.angles, self.seq_dim)


def get_default_state(x:torch.Tensor, requires_grad:bool):
    B, T, C = x.size()
    return TimeMixState(
        torch.zeros([2, B, 0, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad), 
        torch.tensor([]),
    )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class CMix_llama(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_ffn = dim * 4 * 2 // 3 // 32 * 32

        self.w1 = nn.Linear(dim, self.dim_ffn, bias=False)
        self.w2 = nn.Linear(self.dim_ffn, dim, bias=False)
        self.w3 = nn.Linear(dim, self.dim_ffn, bias=False)

    def forward(self, x, last_state:ChannelMixState):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), last_state


class TMix_llama(nn.Module):
    def get_default_state_factory(self): return get_default_state

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = dim // n_heads
        assert dim % self.n_heads == 0

        self.wq = nn.Linear(dim, self.n_heads * self.head_size, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_size, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_size, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_size, dim, bias=False)

    def forward(self, x, xo, kv_cache, last_state:TimeMixState, shared:Shared):
        B, L, D = x.size()
        H = self.n_heads

        q = self.wq(x) 
        k = self.wk(x)
        v = self.wv(x)
        wkv_state = last_state.wkv_state

        # handle recurrent inference via maintaining a kv cache
        if not self.training:
            new_kv_cache = torch.stack([k, v], dim=0)
            wkv_state = torch.cat([wkv_state, new_kv_cache], dim=-2)
            k, v = wkv_state.unbind(0)
            k, v = k.contiguous(), v.contiguous()

        is_causal = q.size(1)==k.size(1)

        q = q.view(B,-1,H,D//H).transpose(1,2)
        k = k.view(B,-1,H,D//H).transpose(1,2)
        v = v.view(B,-1,H,D//H).transpose(1,2)
        q, k = apply_rotary_embedding(q, k, shared.angles)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        y = y.transpose(1,2).reshape(B,L,D)
        y = self.wo(y)
        return y, TimeMixState(wkv_state, last_state.shift_state)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        norm_eps,
        parallel,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.att = TMix_llama(dim, n_heads)
        self.ffn = CMix_llama(dim)
        self.layer_id = layer_id
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        
        self.parallel = parallel

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, x_original_cache, kv_cache, last_model_state:ModelState, shared:Shared):
        last_block_state:BlockState = last_model_state.block_states[self.layer_id]

        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )
            
            if not self.parallel:
                dx, time_mix_state = self.att(modulate(self.ln1(x), shift_msa, scale_msa), x_original_cache, kv_cache, last_block_state.time_mix_state, shared)
                x += gate_msa.unsqueeze(1) * dx
                dx, channel_mix_state = self.ffn(modulate(self.ln2(x), shift_mlp, scale_mlp), last_block_state.channel_mix_state)
                x += gate_mlp.unsqueeze(1) * dx
            else:
                # parallel
                dx_att, time_mix_state = self.att(modulate(self.ln1(x), shift_msa, scale_msa), x_original_cache, kv_cache, last_block_state.time_mix_state, shared)
                dx_ffn, channel_mix_state = self.ffn(modulate(self.ln2(x), shift_mlp, scale_mlp), last_block_state.channel_mix_state)
                x += gate_msa.unsqueeze(1) * dx_att + gate_mlp.unsqueeze(1) * dx_ffn
        else:
            if not self.parallel:
                dx, time_mix_state = self.att(self.ln1(x), x_original_cache, kv_cache, last_block_state.time_mix_state, shared)
                x += dx
                dx, channel_mix_state = self.ffn(self.ln2(x), last_block_state.channel_mix_state)
                x += dx
            else:
                # parallel
                dx_att, time_mix_state = self.att(self.ln1(x), x_original_cache, kv_cache, last_block_state.time_mix_state, shared)
                dx_ffn, channel_mix_state = self.ffn(self.ln2(x), last_block_state.channel_mix_state)
                x += dx_att + dx_ffn

        return x, BlockState(time_mix_state, channel_mix_state)


class FinalLayer(nn.Module):
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


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
        parallel=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        
        self.shared = Shared()

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    norm_eps,
                    parallel,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = self.precompute_freqs_cis(dim // n_heads, 4096)

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

    def forward(self, x, t, y, return_feature=None):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t.to(x.dtype) + y.to(x.dtype)

        for i, layer in enumerate(self.layers):
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)
            if return_feature == i:
                return x
                

        x = self.final_layer(x, adaln_input)
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

    out = model(x, t, y)
    print(out.shape)
    out = model.forward_with_cfg(x, t, y, 0.5)
    print(out.shape)
