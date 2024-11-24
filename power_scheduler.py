import torch

class PowerScheduler:
    def __init__(self, optimizer, batch_size, a=4, b=-0.51, lr_max=0.02, warmup_percent=0, decay_percent=0, total_tokens=None):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.a = a
        self.b = b
        self.lr_max = lr_max
        self.warmup_tokens = total_tokens * warmup_percent
        self.decay_tokens = total_tokens * decay_percent # Default set to a large value to resemble "no decay unless specified"
        self.total_tokens = total_tokens # Can be None for continuous training.
        self.tokens_trained = 0

    def power(self, tokens):
        return min(self.lr_max, self.a * (tokens**self.b))
    
    def step(self):  # The core stepping logic
        self.tokens_trained += self.batch_size  # Increment tokens trained in each step
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        if self.tokens_trained < self.warmup_tokens:
            lr = self.tokens_trained / self.warmup_tokens * self.power(self.warmup_tokens)
            return lr
        elif self.total_tokens is not None and self.tokens_trained > self.total_tokens - self.decay_tokens:
            remaining_tokens = self.total_tokens - self.tokens_trained
            lr = remaining_tokens / self.decay_tokens * self.power(self.total_tokens - self.decay_tokens)  # Linear decay
            return lr
        else:
            lr = self.power(self.tokens_trained)
            return lr