import typer
import torch

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln
    
    def mse_loss(self, x):
        return x ** 2

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        sim_compare = x + vtheta
        error = z1 - x - vtheta
        batchwise_mse = self.mse_loss(error).mean(dim=list(range(1, len(x.shape))))
        batchwise_loss = self.mse_loss(error)
        batchwise_loss = batchwise_loss.mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_loss.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_loss.mean(), ttloss, batchwise_mse.mean()

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

# For nGPT stuff
# def justnorm(x, idim=-1):
#     dtype = x.dtype
#     x = x.float()
#     res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype) 
#     return res

# def normalize_matrices(model):
#     with torch.no_grad():
#         model.final_layer.linear.weight.data.copy_(justnorm(model.final_layer.linear.weight.data, 1))  # final, n_embd
#         model.x_embedder.weight.data.copy_(justnorm(model.x_embedder.weight.data, 0))  # n_embd, PxP
#         model.t_embedder.mlp[0].weight.copy_(justnorm(model.t_embedder.mlp[0].weight.data, 0))  # n_embd, freq
#         model.t_embedder.mlp[2].weight.copy_(justnorm(model.t_embedder.mlp[2].weight.data, 1))  # n_embd, n_embd
#         model.y_embedder.embedding_table.weight.data.copy_(justnorm(model.y_embedder.embedding_table.weight.data, 1))  # n_embd, C
        
#         model.final_layer.AdaFM_Proj[-1].weight.data.copy_(justnorm(model.final_layer.AdaFM_Proj[-1].weight.data, 1))   # n_proj, n_embd

#         for block in model.layers:
#             block.AdaFM_Proj[-1].weight.data.copy_(justnorm(block.AdaFM_Proj[-1].weight.data, 1))   # n_proj, n_embd
            
#             block.attention.wq.weight.data.copy_(justnorm(block.attention.wq.weight.data, 1))   # n_proj, n_embd
#             block.attention.wk.weight.data.copy_(justnorm(block.attention.wk.weight.data, 1))   # n_proj, n_embd
#             block.attention.wv.weight.data.copy_(justnorm(block.attention.wv.weight.data, 1))   # n_proj, n_embd
#             block.attention.wo.weight.data.copy_(justnorm(block.attention.wo.weight.data, 0))   # n_embd, n_proj

#             block.feed_forward.w1.weight.data.copy_(justnorm(block.feed_forward.w1.weight.data, 1))   # n_proj, n_embd
#             block.feed_forward.w2.weight.data.copy_(justnorm(block.feed_forward.w2.weight.data, 0))   # n_embd, n_proj
#             block.feed_forward.w3.weight.data.copy_(justnorm(block.feed_forward.w3.weight.data, 1))   # n_proj, n_embd

def main(CIFAR: bool = False, model_type: str = ""):
    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomCrop(32),
                v2.RandomHorizontalFlip(),
                v2.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).cuda()

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Pad(2),
                v2.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        if model_type == "test":
            from dit_test import DiT_Llama
        else:
            from dit import DiT_Llama
        model = DiT_Llama(
                channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
            ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6:2f}M")
    
    hyperparameter_defaults = dict(
        epochs = 5,
        learning_rate = 2**-11,
        batch_size = 256,
        beta_1 = 0.95,
        beta_2 = 0.95,
        shampoo_beta = 0.95,
        weight_decay = 0.0,
        precondition_freq = 4,
        eps = 1e-7,
        model_size = model_size,
        model_type = model_type
    )

    wandb.init(config=hyperparameter_defaults, project=f"New testing")
    config = wandb.config
    total_time = 0
    
    expected_loss = (model_size / 5.4 * 10**4)**(-0.107)

    rf = RF(model)
    optimizer = torch.optim.Adam(rf.model.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)
    # from power_scheduler import PowerScheduler
    # scheduler = PowerScheduler(optimizer, config.batch_size, lr_max=config.learning_rate,
    #                            warmup_percent=0.05, decay_percent=0.35, total_tokens=config.epochs*6e4)

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=config.batch_size, shuffle=True, drop_last=True)

    for epoch in range(config.epochs):
        start_time = time.time()

        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct, loss_log = rf.forward(x, c)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            wandb.log({"loss": loss_log.item(),
                       "score": expected_loss / loss_log.item()})
            
            # Stop when hit cut-off
            if loss_log.item() > 0.1:
                break

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")
        
        epoch_time = time.time() - start_time
        total_time += epoch_time

        wandb.log({"epoch_time": epoch_time,
                   "total_time": total_time})
        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        rf.model.eval() 
        cond = torch.arange(0, 16).cuda() % 10
        uncond = torch.ones_like(cond) * 10

        init_noise = torch.randn(16, channels, 32, 32).cuda()
        images = rf.sample(init_noise, cond, uncond)
        # image sequences to gif
        gif = []
        for image in images:
            # unnormalize
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            x_as_image = make_grid(image.float(), nrow=4)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))

        gif[0].save(
            f"contents/sample_{epoch}.gif",
            save_all=True,
            append_images=gif[1:],
            duration=100,
            loop=0,
        )

        last_img = gif[-1]
        last_img.save(f"contents/sample_{epoch}_last.png")

        rf.model.train()
    
if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets
    from torchvision.transforms import v2
    from torchvision.utils import make_grid
    from tqdm import tqdm
    import torch.nn.functional as F
    import time

    import wandb
    
    typer.run(main)