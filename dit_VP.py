# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class _1x1_ChebyKANLayer(nn.Module):
  def __init__(self, input_dim, output_dim, degree):
    super(_1x1ChebyKANLayer, self).__init__()
    self.inputdim = input_dim
    self.outdim = output_dim
    self.degree = degree

    self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
    nn.init.xavier_normal_(self.cheby_coeffs, gain=1 / (input_dim * (degree + 1)))
    self.register_buffer("arange", torch.arange(0, degree + 1, 1))

  def forward(self, x):
    # Since Chebyshev polynomial is defined in [-1, 1]
    # We need to normalize x to [-1, 1] using tanh
    x = torch.tanh(x)
    # View and repeat input degree + 1 times
    x = x.view((-1, self.inputdim, 1)).expand(
      -1, -1, self.degree + 1
    )  # shape = (batch_size, inputdim, self.degree + 1)
    # Apply acos
    x = x.acos()
    # Multiply by arange [0 .. degree]
    x *= self.arange
    # Apply cos
    x = x.cos()
    # Compute the Chebyshev interpolation
    y = torch.einsum(
        "bid,iod->bo", x, self.cheby_coeffs
    )  # shape = (batch_size, outdim)
    y = y.view(-1, self.outdim)
    return y

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # Initialize Chebyshev coefficients with shape (input_dim, output_dim, degree + 1)
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.xavier_normal_(self.cheby_coeffs, gain=1 / (input_dim * (degree + 1)))
        
        # Register arange as a buffer to ensure it's moved with the model's device and dtype
        self.register_buffer("arange", torch.arange(0, degree + 1, 1, dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (..., input_dim)
        
        Returns:
            y: Tensor of shape (..., output_dim)
        """
        # Ensure x is within [-1, 1] using tanh
        x = torch.tanh(x)  # Shape: (..., input_dim)
        
        # Apply arccos to normalize input for Chebyshev polynomials
        x = torch.acos(x)  # Shape: (..., input_dim)
        
        # Multiply by arange to get x * [0, 1, ..., degree]
        # This broadcasts arange to the last dimension
        x = x.unsqueeze(-1) * self.arange  # Shape: (..., input_dim, degree + 1)
        
        # Apply cosine to compute Chebyshev polynomials
        x = torch.cos(x)  # Shape: (..., input_dim, degree + 1)
        
        # Perform the einsum operation to compute the weighted sum
        # "...id" corresponds to all leading dimensions, input_dim, degree + 1
        # "iod" corresponds to input_dim, output_dim, degree + 1
        # The resulting shape is (..., output_dim)
        y = torch.einsum("...id,iod->...o", x, self.cheby_coeffs)
        
        return y


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ChebyKANLayer(frequency_embedding_size, hidden_size, 1),
            nn.LayerNorm(hidden_size),
            ChebyKANLayer(hidden_size, hidden_size, 1),
            nn.LayerNorm(hidden_size),
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # GPTAlpha-style attention weights
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # LayerNorm for q, k, v
        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.v_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        
        # DDLoRAdapt parameters
        self.C_q = nn.Parameter(torch.randn(dim, dim))
        self.D_q = nn.Parameter(torch.randn(dim, dim))
        self.C_k = nn.Parameter(torch.randn(dim, dim))
        self.D_k = nn.Parameter(torch.randn(dim, dim))
        self.C_v = nn.Parameter(torch.randn(dim, dim))
        self.D_v = nn.Parameter(torch.randn(dim, dim))

    def ddloradapt(self, x, C, D):
        return x + torch.tanh(x @ C) @ D

    def ddlerp(self, x_t, x_t_minus_1):
        # This is a placeholder for the ddlerp function
        # You may need to implement this based on the specific requirements
        return 0.5 * (x_t + x_t_minus_1)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        
        # Assuming x_t_minus_1 is the previous timestep's input
        # For simplicity, we'll use a shifted version of x
        x_t_minus_1 = torch.roll(x, shifts=1, dims=1)
        x_t_minus_1[:, 0, :] = 0  # Zero out the first token's previous state

        # GPTAlpha Time Mixing
        q_input = self.ddlerp(x, x_t_minus_1)
        k_input = self.ddlerp(x, x_t_minus_1)
        v_input = self.ddlerp(x, x_t_minus_1)

        # Apply DDLoRAdapt
        q_input = self.ddloradapt(q_input, self.C_q, self.D_q)
        k_input = self.ddloradapt(k_input, self.C_k, self.D_k)
        v_input = self.ddloradapt(v_input, self.C_v, self.D_v)

        # Linear projections
        xq, xk, xv = self.wq(q_input), self.wk(k_input), self.wv(v_input)

        # Apply LayerNorm
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # Apply rotary embeddings (unchanged from original code)
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Perform scaled dot-product attention
        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        
        output = output.flatten(-2)

        # Final projection
        return self.wo(output)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        # Rotary embedding application (unchanged from original code)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        # Reshaping for broadcast (unchanged from original code)
        ndim = x.ndim
        assert 0 <= 1 < ndim
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

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
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = ChebyKANLayer(min(hidden_size, 1024), 2 * hidden_size, 2)
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama_VP(nn.Module):
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
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
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
    return DiT_Llama_VP(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama_VP(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)


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
