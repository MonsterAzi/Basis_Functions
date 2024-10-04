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

class TimeMix_6(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

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

class RWKV_Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix_6(args, layer_id)
        self.ffn = ChannelMix_6(args, layer_id)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(args.dim, 1024), 6 * args.dim, bias=True),
        )
        
    def forward(self, x, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )
            
            if self.layer_id == 0:
                x = self.ln0(x)

            x = x + gate_msa.unsqueeze(1) * self.att(
                modulate(self.ln1(x), shift_msa, scale_msa)
            )
            x = x + gate_mlp.unsqueeze(1) * self.ffn(
                modulate(self.ln2(x), shift_mlp, scale_mlp)
            )
        else:
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

class FinalLayer(nn.Module):
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

        self.in_channels = args.in_channels
        self.out_channels = args.in_channels
        self.input_size = args.input_size
        self.patch_size = args.patch_size
        
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0
        
        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(args.in_channels, args.dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, args.dim // 2),
            nn.Conv2d(args.dim // 2, args.dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, args.dim // 2),
        )

        self.x_embedder = nn.Linear(args.patch_size * args.patch_size * args.dim // 2, args.dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(args.dim, 1024))
        self.y_embedder = LabelEmbedder(args.num_classes, min(args.dim, 1024), args.class_dropout_prob)

        self.blocks = nn.ModuleList([RWKV_Block(args, i) for i in range(args.n_layers)])
        
        self.final_layer = FinalLayer(args.dim, args.patch_size, self.out_channels)
        
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
        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t.to(x.dtype) + y.to(x.dtype)

        for block in self.blocks:
            x = block(x, adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        
        return x


def DiT_RWKV_Test(args):
    return DiT_RWKV(args)


if __name__ == "__main__":
    import types
    args = types.SimpleNamespace()
    args.patch_size = 2
    args.dim = 3072
    args.n_layers = 32
    
    args.n_embd = 3072
    args.in_channels=3
    args.input_size=32
    args.class_dropout_prob=0.1
    args.num_classes=10
    
    args.head_size_a = 64
    args.head_size_divisor = 8
    
    model = DiT_RWKV_Test(args)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out = model(x, t, y)
        print(out.shape)
        out = model.forward_with_cfg(x, t, y, 0.5)
        print(out.shape)