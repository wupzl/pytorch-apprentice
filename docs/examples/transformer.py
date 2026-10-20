"""Minimal Pre-Norm block; python examples/transformer.py."""
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, causal=False, padding_mask=None):
        h = self.ln1(x)
        mask = None
        if causal:
            n = x.size(1)
            mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        out, _ = self.attn(h, h, h, attn_mask=mask,
                           key_padding_mask=padding_mask, need_weights=False)
        x = x + out
        return x + self.ffn(self.ln2(x))


if __name__ == '__main__':
    torch.manual_seed(42)
    block = TransformerBlock()
    x = torch.randn(2, 10, 128)
    y = block(x, causal=True)
    print(y.shape)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    block.eval()
    with torch.no_grad():
        changed = x.clone()
        changed[:, 6:] += 10
        assert torch.allclose(block(x, causal=True)[:, :6], block(changed, causal=True)[:, :6], atol=1e-6)
    print('Backward pass and causal-mask check passed.')
