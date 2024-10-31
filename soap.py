import torch
import torch.optim as optim

# Orthogonalization Methods
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = {'svd': zeropower_via_svd, 'newtonschulz5': zeropower_via_newtonschulz5}

class SOAP(optim.Optimizer):
    def __init__(self, params, lr=0.003, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.01,
                 precondition_frequency=10, max_precond_dim=10000, merge_dims=False,
                 precondition_1d=False, data_format="channels_first", correct_bias=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency, max_precond_dim=max_precond_dim,
                        merge_dims=merge_dims, precondition_1d=precondition_1d, correct_bias=correct_bias,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)
        self._data_format = data_format

    def step(self):
        loss = None
        for group in self.param_groups:
            lr = group['lr']
            zeropower_backend = zeropower_backends[group['backend']]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize buffers
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                
                # Momentum update
                beta1, beta2 = group["betas"]
                buf.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_projected = buf

                # Orthogonalization
                grad_projected = zeropower_backend(grad_projected, steps=group["backend_steps"])
                
                # Scaling for orthogonalized gradients
                scale = max(grad_projected.size(0), grad_projected.size(1)) ** 0.5
                p.data.add_(grad_projected, alpha=-lr * scale)
                
                # Apply weight decay
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-lr * group["weight_decay"])

        return loss