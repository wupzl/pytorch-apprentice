"""CPU checks for the continued chapters; requires torch, no external datasets."""
import copy
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).parent / 'examples'))
from mini_gpt import MiniGPT, generate, main as train_mini
from lora import main as verify_lora
from post_training import main as verify_post_training


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = MiniGPT(12, d_model=16, num_heads=2, num_layers=1, block_size=8)
    ids = torch.randint(12, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 12
    model.eval()
    with torch.no_grad():
        logits, _ = model(ids)
        altered, _ = model(changed)
        assert logits.shape == (2, 8, 12)
        assert torch.allclose(logits[:, :5], altered[:, :5], atol=1e-6)
    model.train()
    for kwargs in ({}, {'top_k': 1}, {'top_p': 0.01}, {'top_k': 30, 'top_p': 0.9}):
        result = generate(model, ids, 4, **kwargs)
        assert result.shape == (2, 12)
        assert torch.equal(result[:, :8], ids)
        assert result.min() >= 0 and result.max() < 12
        assert model.training  # generation restores caller's mode
    # Compare gradients for an uneven 2+3 micro-batch split, weighted by sample count.
    network = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 1))
    accumulated = copy.deepcopy(network)
    x, y = torch.randn(5, 3), torch.randn(5, 1)
    nn.functional.mse_loss(network(x), y).backward()
    for start, stop in ((0, 2), (2, 5)):
        loss = nn.functional.mse_loss(accumulated(x[start:stop]), y[start:stop])
        (loss * (stop - start) / len(x)).backward()
    assert all(torch.allclose(a.grad, b.grad, atol=1e-6) for a, b in zip(network.parameters(), accumulated.parameters()))
    # Non-reentrant checkpointing must preserve forward values and gradients.
    network.zero_grad(set_to_none=True)
    recomputed = copy.deepcopy(network)
    direct = network(x)
    checked = checkpoint(recomputed, x, use_reentrant=False)
    assert torch.allclose(direct, checked)
    direct.square().sum().backward()
    checked.square().sum().backward()
    assert all(torch.allclose(a.grad, b.grad, atol=1e-6) for a, b in zip(network.parameters(), recomputed.parameters()))
    print('PASS: GPT shape/causality/sampling, uneven accumulation and checkpoint gradients.')
    verify_lora()
    verify_post_training()
    train_mini(steps=200)


if __name__ == '__main__':
    main()
