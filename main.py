import typer
import torch
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use a smaller network for efficiency, given the simpler grayscale input
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return (x_features - y_features) ** 2

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln
        self.delta_0 = torch.tensor(1.35)
        
    def MAE_loss(self, x):
        return abs(x)
    
    def MSE_loss(self, x):
        return x ** 2
    
    def log_cosh_loss(self, x):
        return torch.log(torch.cosh(x))
    
    def pseudo_huber_loss(self, x, t):
        delta = self.delta_0 * torch.exp(torch.log(self.delta_0) * t)
        delta = delta.view(-1, 1, 1, 1)
        return delta**2 * (torch.sqrt(1 + (x / delta)**2) - 1)

    def huber_loss(self, x, delta=1.0):
        abs_x = torch.abs(x)
        quadratic = torch.where(abs_x <= delta, 0.5 * x ** 2, torch.zeros_like(x))
        linear = torch.where(abs_x > delta, delta * (abs_x - 0.5 * delta), torch.zeros_like(x))
        loss = quadratic + linear
        return loss

    def smooth_l1_loss(self, x, beta=1.0):
        abs_x = torch.abs(x)
        loss = torch.where(abs_x < beta, 0.5 * x ** 2 / beta, abs_x - 0.5 * beta)
        return loss

    def charbonnier_loss(self, x, epsilon=1e-6):
        loss = torch.sqrt(x ** 2 + epsilon ** 2)
        return loss
    
    def hfen_loss(self, a, b, kern: str = "Lap"):
        if kern == "Lap":
            # Define the Laplacian of Gaussian (LoG) filter
            kernel = torch.tensor([
                [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]
            ], dtype=torch.float32).to(a.device)
        else:
            kernel = torch.tensor([
                [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
            ], dtype=torch.float32).to(a.device)
        
        # Apply the LoG filter to both the true image and the prediction
        a_log = F.conv2d(a, kernel, padding=1)
        b_log = F.conv2d(b, kernel, padding=1)
        
        # Compute the HFEN loss
        hfen_loss = torch.abs(a_log - b_log)
        return hfen_loss
    
    def perceptual_loss(self, a, b):
        return PerceptualLoss().cuda()(a, b)

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
        batchwise_mse = (error ** 2).mean(dim=list(range(1, len(x.shape))))
        batchwise_loss = (error ** 2)
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
    
def create_mse(CIFAR: bool = False):
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
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()

    rf = RF(model)
    optimizer = SOAP(model.parameters(), lr=1e-2)
    
    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True)

    for epoch in range(15):
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct, loss_log = rf.forward(x, c)
            loss.backward()
            optimizer.step()

        rf.model.train()
    
    return rf

def main(CIFAR: bool = False):
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
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6:2f}M")
    
    hyperparameter_defaults = dict(
        epochs = 5,
        learning_rate = 7e-3,
        batch_size = 256,
        beta_1 = 0.95,
        beta_2 = 0.95,
        shampoo_beta = -1,
        weight_decay = 0.01,
        precondition_freq = 8,
        eps = 1e-8
    )

    wandb.init(config=hyperparameter_defaults, project=f"rf_{dataset_name}")
    config = wandb.config

    rf = RF(model)
    optimizer = SOAP(model.parameters(), lr=config.learning_rate)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.epochs, steps_per_epoch=int(6e+5//config.batch_size))

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=config.batch_size, shuffle=True, drop_last=True)

    for epoch in range(config.epochs):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct, loss_log = rf.forward(x, c)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            wandb.log({"loss": loss_log.item()})

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        rf.model.eval()
        with torch.no_grad():
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
    from soap import SOAP
    from PIL import Image
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets
    from torchvision.transforms import v2
    from torchvision.utils import make_grid
    from tqdm import tqdm
    import torch.nn.functional as F
    import copy

    import wandb
    from dit import DiT_Llama
    
    main()