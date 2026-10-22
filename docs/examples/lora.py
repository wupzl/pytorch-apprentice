"""Chapter 12: verify LoRA initialization and frozen base weights on CPU."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        self.original = original_linear
        for param in self.original.parameters():
            param.requires_grad_(False)
        self.scaling = alpha / rank
        # Follow the original layer's dtype and device.
        weight = original_linear.weight
        self.lora_A = nn.Parameter(weight.new_empty(rank, original_linear.in_features))
        self.lora_B = nn.Parameter(weight.new_zeros(original_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.original(x) + self.scaling * delta


def main():
    torch.manual_seed(42)
    base = nn.Linear(128, 128)
    layer = LoRALinear(base, rank=8, alpha=16)
    x, target = torch.randn(4, 128), torch.randn(4, 128)
    original = {name: p.detach().clone() for name, p in base.named_parameters()}
    assert torch.equal(layer(x), base(x)), 'Zero B must initially preserve base output'
    optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(layer(x), target)
        loss.backward()
        assert all(p.grad is None for p in base.parameters())
        assert layer.lora_B.grad.abs().sum() > 0
        if step == 0:
            assert torch.count_nonzero(layer.lora_A.grad) == 0
        else:
            assert layer.lora_A.grad.abs().sum() > 0
        optimizer.step()
    assert all(torch.equal(p, original[name]) for name, p in base.named_parameters())
    print('PASS: initial identity, frozen base, B-first then A gradient flow.')


if __name__ == '__main__':
    main()
