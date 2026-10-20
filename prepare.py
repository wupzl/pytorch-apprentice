"""One-time editorial import. Normal builds only read chapters/."""
from pathlib import Path
import re, shutil

ROOT = Path(__file__).resolve().parent
source = ROOT.parent / 'pytorch.md'
(ROOT / 'source').mkdir(exist_ok=True)
shutil.copyfile(source, ROOT / 'source/pytorch-original.md')
for name in ['katex.cjs', 'KaTeX-LICENSE.txt']:
    (ROOT / 'vendor').mkdir(exist_ok=True)
    shutil.copyfile(ROOT.parent / 'reader/vendor' / name, ROOT / 'vendor' / name)
shutil.copyfile(ROOT.parent / 'reader/render-math.cjs', ROOT / 'render-math.cjs')
parts = re.split(r'^# 《PyTorch：炼金术士的学徒》\s*$', source.read_text(encoding='utf-8'), flags=re.M)[1:]
assert len(parts) == 10
notes = [
    '“下山”是局部近似：梯度指向局部上升最快的方向，负梯度才是下降方向；学习率过大时，更新仍可能让损失升高。这里用单个样本演示训练机制，不代表模型已学会对未知数据泛化。',
    '普通 Python list 中的子模块不会自动注册；请使用 nn.ModuleList 或 nn.Sequential。固定状态可用 register_buffer 注册，它会随 model.to(device) 移动，但不会作为可训练参数交给优化器。',
    '带偏置的 Linear 严格说是仿射变换；多个仿射层不加非线性，仍可合并成一个仿射层。下面用“线性层”沿用深度学习中的常见叫法。',
    '本章采用单标签多分类：logits 形状为 [B, C]，类别索引标签为 [B]、dtype=torch.long，取值在 0 到 C−1。CrossEntropyLoss 直接接收 logits。多标签分类则通常使用 BCEWithLogitsLoss，不能直接照搬这里的设置。',
    '文中的样本数与批次数换算针对常见的 map-style Dataset。Windows 上把 num_workers 设为大于 0 时，请将创建 DataLoader 和训练的入口放在 if __name__ == "__main__": 中；初学时可以先用 num_workers=0。',
    '先把模型移动到目标设备，再创建优化器。model.eval() 切换 Dropout、BatchNorm 等层的行为，no_grad() 控制梯度记录，二者需分别设置。没有 CUDA 的电脑可以直接使用 CPU 跑这些小例子。',
    'Adam 并非在所有任务上优于 SGD；效果取决于模型、数据与调参。学习率调度器也不是训练必需品：StepLR、CosineAnnealingLR 的调用频率应与 step_size、T_max 的单位一致；ReduceLROnPlateau 则根据验证指标调用。',
    'Embedding 是可学习的查表映射，常用较低维的稠密向量，但并不要求 embedding_dim 小于词表大小，也不等于 PCA。设置 padding_idx 只处理该行的梯度等行为，不会自动替后面的均值池化或 Attention 忽略 PAD。',
    '本章最小 SelfAttention 用于观察形状与信息聚合，默认是双向的。nn.MultiheadAttention 的布尔 mask 中 True 表示禁止关注；F.scaled_dot_product_attention 的布尔 mask 中 True 则表示允许参与。避免让某一行的所有 key 都被屏蔽。',
    '本章是 Transformer 的组成实验，还不是完整 GPT：输入需要位置信息，语言建模还需要因果约束、输出头和训练目标。没有位置机制的普通 Self-Attention 本身无法区分完整词序。',
]
for i, p in enumerate(parts):
    p = p.strip()
    p = re.sub(r'^## 第[^\n]+章：[^\n]+\n', '', p, count=1)
    # Strip copied user turns, keeping dialogue spoken by story characters.
    p = re.sub(r'^(?:下一章[。]?|行，下一章[。]?|行。下一章。)\s*$', '', p, flags=re.M)
    if i < 9:
        p = re.split(r'^# 下一章\s*$', p, maxsplit=1, flags=re.M)[0]
    if i == 6:
        p = p.replace('把 weight decay 混进梯度计算。', '在传统 Adam 的耦合式 L2 正则实现中，把 weight_decay 对应的项加入梯度；未设置权重衰减时不存在这一项。')
        p = p.replace('教授摇头。\n\n训练不同阶段：', '教授说：“固定学习率当然可以，调度则提供另一种选择。”\n\n训练不同阶段：')
    if i == 7:
        p = p.replace('self, vocab_size, embed_dim, num_classes', 'self, vocab_size, embed_dim, num_classes, pad_id=0')
        p = p.replace('self.embedding = nn.Embedding(\n            vocab_size,\n            embed_dim\n        )', 'self.pad_id = pad_id\n        self.embedding = nn.Embedding(\n            vocab_size, embed_dim, padding_idx=pad_id\n        )')
        p = p.replace('x = x.mean(dim=1)', 'mask = (token_ids != self.pad_id).unsqueeze(-1)\n        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)')
    if i == 9:
        p = p.replace('\\\\[', '\\[').replace('\\\\]', '\\]').replace('\\\\(', '\\(').replace('\\\\)', '\\)').replace('\\_', '_')
        residual = '''import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)'''
        block = (ROOT / 'examples/transformer.py').read_text(encoding='utf-8')
        block = block[:block.index("if __name__")].strip()
        def repair(m):
            b = m.group(1).strip()
            if b.startswith('import torchimport'):
                b = residual if 'class ResidualBlock' in b else block
            elif b.startswith('block = TransformerBlock('):
                b = 'block = TransformerBlock(d_model=128, num_heads=4)\nx = torch.randn(2, 10, 128)\ny = block(x)\nprint(y.shape)'
            elif b.startswith('seq_len = x.size(1)causal_mask'):
                b = 'seq_len = x.size(1)\ncausal_mask = torch.triu(\n    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),\n    diagonal=1,\n)'
            elif b.startswith('attn_out, _ = self.attn(') and '    ' in b:
                b = 'attn_out, _ = self.attn(\n    h, h, h,\n' + ('    attn_mask=causal_mask,\n' if 'attn_mask=' in b else '') + '    need_weights=False,\n)'
            b = b.replace('print(ln.weight.shape)print(ln.bias.shape)', 'print(ln.weight.shape)\nprint(ln.bias.shape)')
            b = b.replace('ln = nn.LayerNorm(512)y = ln(x)', 'ln = nn.LayerNorm(512)\ny = ln(x)')
            b = b.replace('fc1 = nn.Linear(512, 2048)fc2 = nn.Linear(2048, 512)', 'fc1 = nn.Linear(512, 2048)\nfc2 = nn.Linear(2048, 512)')
            return '```\n' + b + '\n```'
        p = re.sub(r'```\n(.*?)```', repair, p, flags=re.S)
        p = p.replace('Transformer Block 随堂测试\n\n0/5 已回答', 'Transformer Block 纸上练习（原稿保留）')
        p = p.replace('## 第十一章预告：', '## 待续 · 第十一章预告：')
        p = p.replace('下一章，我们会从零组装', '后续计划（尚未发布）：从零组装')
        p = p.replace('下一章：《预测下一个词——从 Transformer 到 GPT 的诞生》。', '第十一章尚未发布。本次阅读到这里告一段落。')
        p = p.replace('## 待续 · 第十一章预告：', '> 纸上练习参考：1. 恒等路径；2. 最后 512 维；3. 增加非线性表达容量；4. x + Attention(LN(x))；5. [B,N,D] → [B,N,D]。\n\n## 待续 · 第十一章预告：')
    # Normalize heading levels only outside fenced code.
    in_code = False
    lines = []
    for line in p.splitlines():
        if line.startswith('```'):
            in_code = not in_code
        elif not in_code and re.match(r'^#{1,2} ', line):
            content = re.sub(r'^#+ ', '', line)
            line = ('## ' if re.match(r'[一二三四五六七八九十]+、|待续', content) else '### ') + content
        lines.append(line)
    p = '\n'.join(lines).strip()
    # Infer Python only for parseable code; diagrams remain text.
    import ast
    def language(m):
        original, b = m.group(1), m.group(2)
        if original and original != 'python':
            return m.group(0)
        if b.strip().endswith(':') and len(b.strip().splitlines()) == 1:
            b = b.rstrip() + '\n    ...  # 此处只展示结构，完整代码见上下文\n'
        if b.startswith('mask = ') and '\n        x = ' in b:
            b = b.replace('\n        x = ', '\nx = ')
        try:
            ast.parse(b)
            lang = 'python' if re.search(r'\b(import |print\(|torch\.|nn\.|self\.|def |class |model\(|loss\.|=)', b) else 'text'
        except SyntaxError:
            lang = 'text'
        return f'```{lang}\n{b}```'
    p = re.sub(r'```([^\n]*)\n(.*?)```', language, p, flags=re.S)
    p = p.replace('W \\times “我”', 'W \\times \\text{“我”}')
    p = p.replace('3.03亿', '3.03\\text{亿}')
    p = f'> 阅读提示：{notes[i]}\n\n' + p
    (ROOT / 'chapters').mkdir(exist_ok=True)
    (ROOT / 'chapters' / f'{i+1:02}.md').write_text(p+'\n', encoding='utf-8')
print('Imported and edited 10 chapters; original preserved.')
