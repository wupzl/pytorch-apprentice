"""Chapter 11: small character-level GPT. No dataset/model downloads required."""
import argparse
import torch
from torch import nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.ln1(x)
        out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + out
        return x + self.ffn(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4,
                 num_layers=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if not 0 < T <= self.block_size:
            raise ValueError('Sequence length must be in [1, block_size]')
        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            # targets are ALREADY shifted by get_batch; do not shift twice.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=1.0):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('Require temperature > 0 and 0 < top_p <= 1')
    if top_k is not None and top_k < 1:
        raise ValueError('top_k must be positive')
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                threshold = logits.topk(min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float('-inf'))
            if top_p < 1:
                sorted_logits, order = logits.sort(descending=True, dim=-1)
                cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                # Keep the first token crossing the threshold, including top-1.
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
                logits = torch.empty_like(logits).scatter(1, order, sorted_logits)
            next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
    finally:
        model.train(was_training)
    return idx


def main(steps=200):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    text = ('林默走进实验室。教授正在研究PyTorch。神经网络正在学习语言。'
            '林默问教授，Transformer为什么能够生成文本？'
            '教授回答，因为模型能够预测下一个Token。')
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    block_size, batch_size = 32, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiniGPT(len(chars), block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(steps):
        starts = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in starts]).to(device)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f'Step {step + 1}: loss={loss.item():.4f}')
    context = torch.tensor([[stoi[c] for c in '林默']], device=device)
    output = generate(model, context, 60, temperature=0.8, top_k=10)
    print(''.join(chars[i] for i in output[0].tolist()))
    print('Toy corpus demonstration only; this is not a question-answering model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    main(parser.parse_args().steps)
