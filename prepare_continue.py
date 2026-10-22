"""One-time import of chapters 11-13; normal builds read chapters/ only."""
from pathlib import Path
import ast
import re
import shutil
import textwrap

ROOT = Path(__file__).resolve().parent


def example(name, symbol):
    source = (ROOT / 'examples' / name).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == symbol)
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) - 1
    return '\n'.join(source.splitlines()[start:node.end_lineno])


fixed = {
    11: {
        9: 'tokens = torch.tensor([1, 2, 3, 4, 5])\nx = tokens[:-1]\ny = tokens[1:]',
        16: '''seq_len = 4
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)''',
        41: '''import torch
import torch.nn as nn
import torch.nn.functional as F

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}''',
        44: 'def encode(s):\n    return [stoi[c] for c in s]',
        45: 'def decode(ids):\n    return "".join(itos[i] for i in ids)',
        47: 'block_size = 32\nbatch_size = 16',
        49: '''def get_batch():
    if len(data) <= block_size:
        raise ValueError("语料至少需要 block_size + 1 个 Token")
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y''',
        51: example('mini_gpt.py', 'TransformerBlock'),
        53: example('mini_gpt.py', 'MiniGPT'),
        60: '''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(vocab_size=vocab_size, d_model=64, num_heads=4,
                num_layers=2, block_size=block_size).to(device)''',
        62: '''model.train()
for step in range(1000):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")''',
        63: 'pred = model(x)\nloss = loss_fn(pred, y)\nloss.backward()\noptimizer.step()',
        64: 'logits, loss = model(x, y)\nloss.backward()\noptimizer.step()',
        79: 'probs = F.softmax(last_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)',
        90: '''values, _ = torch.topk(logits, k=min(3, logits.size(-1)))
threshold = values[:, [-1]]
logits = logits.masked_fill(logits < threshold, float("-inf"))''',
        95: example('mini_gpt.py', 'generate'),
        101: 'output_text = decode(generated[0].tolist())\nprint(output_text)',
    },
    12: {
        11: '''# 需要支持 BF16 的 CUDA GPU，以及已经定义的 MyModel、loader、loss_fn。
device = torch.device("cuda")
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()''',
        12: 'with torch.autocast(device_type="cuda", dtype=torch.bfloat16):\n    ...  # 在此执行前向传播和 Loss 计算',
        16: '''# 假设 model、optimizer 已创建并位于 CUDA，loader 与 loss_fn 已定义。
scaler = torch.amp.GradScaler("cuda")
model.train()
for x, y in loader:
    x, y = x.to("cuda"), y.to("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = loss_fn(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()''',
        24: '''# 这里恰好累积 16 个同样大小、等权的 Micro-Batch。
accumulation_steps = 16
optimizer.zero_grad(set_to_none=True)
for step in range(accumulation_steps):
    x, y = get_batch()
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    (loss / accumulation_steps).backward()
optimizer.step()''',
        30: 'for block in self.blocks:\n    x = checkpoint(block, x, use_reentrant=False)',
        42: 'import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n' + example('lora.py', 'LoRALinear'),
        46: '''base = nn.Linear(128, 128)
lora_layer = LoRALinear(base, rank=8, alpha=16)
x, target = torch.randn(4, 128), torch.randn(4, 128)
optimizer = torch.optim.AdamW(
    [p for p in lora_layer.parameters() if p.requires_grad], lr=1e-3
)
optimizer.zero_grad(set_to_none=True)
output = lora_layer(x)
loss = F.mse_loss(output, target)
loss.backward()
optimizer.step()''',
        47: 'for name, param in lora_layer.named_parameters():\n    print(name, param.requires_grad)',
        61: '''import torch
import torch.nn.functional as F

# Q, K, V 应为 [batch, heads, seq_len, head_dim]。
output = F.scaled_dot_product_attention(
    Q, K, V, is_causal=True, dropout_p=0.0
)''',
        62: '''dropout_p = 0.1 if self.training else 0.0
output = F.scaled_dot_product_attention(
    Q, K, V, is_causal=True, dropout_p=dropout_p
)''',
        66: '''# 配置骨架：需要兼容 GPU、依赖和实际模型 ID，不是完整训练脚本。
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "your-compatible-7b-model"  # 替换为可访问且许可适用的模型
if not torch.cuda.is_available():
    raise RuntimeError("此示例按单张 CUDA GPU 配置")
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quant_config,
    device_map={"": torch.cuda.current_device()},  # 单卡训练，不使用推理自动卸载
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()''',
    },
    13: {
        18: '''import torch
import torch.nn.functional as F

# input_ids 的形状是 [B, T]；下面演示一个 6 Token 的样本。
# Prompt 为 [11, 12, 13]，回答为 [21, 22, 23]。
labels = torch.tensor([[-100, -100, -100, 21, 22, 23]], device=logits.device)
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
if not (shift_labels != -100).any():
    raise ValueError("当前 Batch 没有可监督的回答 Token")
loss = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1), ignore_index=-100
)''',
        33: 'import torch\nimport torch.nn.functional as F\n\n' + example('post_training.py', 'dpo_loss'),
    },
}
answers = {
    11: ['真实序列已知，因果掩码允许并行计算各位置；标签仍来自下一个 Token。', 'LM Head 把隐藏向量映射成词表大小的 logits。', '在相同 logits 下，较低的正温度让高概率 Token 更占优势。', 'KV Cache 缓存历史 Token 在各层 Attention 的 Key 和 Value，不缓存梯度。', '未来 Token 尚未知，普通自回归生成的后续决策依赖已生成内容。'],
    12: ['否。减少的是适配器对应的可训练参数，基础权重、激活与临时工作区仍占内存。', '减小每次 Micro-Batch 的激活需求，在多次反向后再更新一次参数。', '反向时额外重计算，以时间换空间。', '冻结量化基础权重，训练 LoRA 适配器等选定的额外参数。', '主要减少标准精确 Attention 的中间存储与数据搬运，并非线性近似。'],
    13: ['使相应目标位置不参与交叉熵；模型依然能读取 Prompt 上下文。', '同一 Prompt 下的 chosen/rejected 成对偏好数据。', '根据反馈数据学到的评分估计，不能当作正确性的证明。', '抑制过大策略变化带来的进一步优化激励，不是硬性限制所有更新。', 'Loss 只度量指定目标，仍需在独立、未见任务上评估质量和泛化。'],
}
notes = {
    11: '本章的字符级 Mini-GPT 只在极小语料上演示训练与生成，不会因此成为通用问答模型。get_batch 已经移位标签，MiniGPT.forward 中不要再移位。代码没有实现 KV Cache，相关章节用于理解工程原理。',
    12: '显存估算必须注明参数量、存储精度与状态副本；样本 Batch Size 不等于有效 Token 数。本章 GPU 配置片段不是完整训练脚本；可在 CPU 上验证的 LoRA、累积与重计算示例见运行说明。',
    13: '本章 SFT 示例传入未移位 labels，在 Loss 中只移位一次；Prompt 和 Padding 的目标标签设为 -100。DPO 示例验证核心损失，不包含完整偏好训练系统，PPO/RLHF 部分属于概念讲解。',
}


def main():
    source = ROOT.parent / 'pytorch-continue.md'
    shutil.copyfile(source, ROOT / 'source/pytorch-continue-original.md')
    parts = re.split(r'^# 《PyTorch：炼金术士的学徒》\s*$',source.read_text(encoding='utf-8'),flags=re.M)[1:]
    assert len(parts) == 3
    for number, p in enumerate(parts, 11):
        blocks = list(re.finditer(r'```([^\n]*)\n(.*?)```',p,re.S))
        assert len(blocks) == {11:133,12:78,13:59}[number], (number,len(blocks))
        for j, m in reversed(list(enumerate(blocks))):
            b = textwrap.dedent(fixed[number].get(j,m.group(2))).strip()
            try:
                ast.parse(b)
                python = bool(re.search(r'import |class |def | = |print\(|self\.|torch\.|model\(|logits\[|labels\[',b))
            except SyntaxError:
                python = False
            # Explicitly mark diagrams/shapes as text, not executable Python.
            if b.startswith('logits.shape =') or any(t in b for t in ('→','│','Tensor Shape')):
                python = False
            if python:
                b = ast.unparse(ast.parse(b)) if len(b.splitlines()) == 1 and len(b)>85 else b
            p = p[:m.start()] + f'```{"python" if python else "text"}\n{b}\n```' + p[m.end():]
        p = re.sub(r'^## 第[^\n]+章：[^\n]+\n','',p.strip(),count=1)
        if number < 13:
            p = re.split(r'^# (?:下一章：|第十三章预告：)',p,maxsplit=1,flags=re.M)[0]
        p = re.sub(r'^0/5 已完成\s*$','',p,flags=re.M)
        p = p.replace('\\\\[','\\[').replace('\\\\]','\\]').replace('\\\\(','\\(').replace('\\\\)','\\)')
        p = p.replace('\\_', '_').replace('\\<','<').replace('\\\\%','\\%').replace('&#x20;','')
        if number == 11:
            p = p.replace('训练数据整体向右错位一格','输入与预测目标错开一格')
            p = p.replace('pred = model(x)\nloss = loss_fn(pred, y)', '# 对比前几章的一般分类模型写法（非 MiniGPT 接口）：\npred = model(x)\nloss = loss_fn(pred, y)')
        if number == 12:
            p = re.sub(r'7B 模型：简化 FP32 AdamW 训练显存构成.*?林默望着自己的显卡。', '''7B 模型：简化 FP32 AdamW 状态预算（十进制 GB）

| 状态 | 每参数字节 | 7B 参数理论占用 |
| --- | --- | --- |
| FP32 权重 | 4 | 28 GB |
| FP32 梯度 | 4 | 28 GB |
| FP32 一阶矩 | 4 | 28 GB |
| FP32 二阶矩 | 4 | 28 GB |
| 合计 | 16 | 112 GB（约 104.3 GiB） |

只计算指定状态，不含激活、临时工作区、缓存及其他开销；不是所有混合精度实现的通用公式，也不是显存实测。

林默望着自己的显卡。''', p, count=1, flags=re.S)
            p = re.sub(r'三种格式的典型布局：.*?林默仔细看。', '''三种格式的典型布局：

| 格式 | 总位数 | 符号位 | 指数位 | 尾数字段 |
| --- | --- | --- | --- | --- |
| FP32 | 32 | 1 | 8 | 23 |
| FP16 | 16 | 1 | 5 | 10 |
| BF16 | 16 | 1 | 8 | 7 |

尾数字段不包含规范化浮点数的隐含前导有效位。

林默仔细看。''', p, count=1, flags=re.S)
            start = p.index('## 十二、一个可交互的显存训练配置实验')
            end = p.index('## 十三、', start)
            p = p[:start] + '''## 十二、把训练配置写成一张估算表

教授把原来的计算表打印出来。

“先手算几组配置，看看我们到底改变了什么。”

| 每卡 Micro-Batch | 累积步数 | 数据并行 GPU 数 | 每次更新的有效样本数 |
| --- | --- | --- | --- |
| 4 | 16 | 1 | 64 |
| 2 | 32 | 1 | 64 |
| 4 | 8 | 2 | 64 |

有效样本数 = 每卡 Micro-Batch × 累积步数 × 数据并行卡数。

这张表不是显存测量结果。前两组虽有相同的有效样本数，单次激活需求与运行开销可能不同；模型权重和优化器状态也不会随 Micro-Batch 同比例缩小。第三组假设各数据并行副本处理等量独立样本，不能把张量并行卡数直接代入。

对于变长文本，Loss 若按有效 Token 平均，应在整个累积窗口按有效 Token 总数归一化；简单把每个 Micro-Batch 的均值除以累积步数，可能产生不同的加权。遍历 DataLoader 时，最后不足一个累积窗口的数据也应提交更新，并按实际样本或 Token 数归一化。配合 AMP 时，应在完整窗口累积结束后再 unscale、裁剪、step 与更新 scaler。

''' + p[end:]
            p = p.replace('三十五、推荐的初始微调配置','三十五、一组用于理解的微调配置')
            p = p.replace('“但它是一个合理的实验起点。”','“先把它当作教学配置；是否适合你的模型，要用实际测量和验证集结果判断。”')
            p = p.replace('## 二十四、为什么 A 随机初始化，B 初始化为零？', '> 初始化时 B=0，因此普通损失的首步反传通常先给 B 非零梯度，A 的该项梯度为零；B 更新后，A 才开始获得相应梯度。这不等于 A 永远不学习。\n\n## 二十四、为什么 A 随机初始化，B 初始化为零？')
        if number == 13:
            p = p.replace('# 三、最容易写错的地方', '# 第三幕：最容易写错的地方').replace('# 四、为什么 Loss', '# 第四幕：为什么 Loss').replace('# 八、奖励黑客', '# 第八幕：奖励黑客').replace('# 十、DPO 和 RLHF', '# 第十幕：DPO 和 RLHF')
            p = p.replace('我们使用未移位的标签 Tensor，并在计算 Loss 时手动完成移位。', '我们使用与完整输入等长、未移位的标签 Tensor，并在计算 Loss 时手动完成移位。上面的五项 Labels 已是移位后的示意；不能直接传给下面这段代码。下面假设 logits.shape=[1,6,V]，且 V 至少为 24。这里 -100 只屏蔽目标监督，不会屏蔽读取上下文的 Attention；输入 Padding 还需要独立的 attention mask。')
            p = p.replace('## 十五、如果没有 Chosen/Rejected 成对答案呢？','## 十五、如果希望模型在线探索并接受反馈呢？')
            p = p.replace('“这就进入强化学习的世界。”','“这就进入强化学习的世界。不过经典奖励模型本身也常通过成对人类偏好训练；RLHF 并不是简单地不再需要偏好数据。”')
            p = p.replace('## 十四、用 PyTorch 手写 DPO 核心 Loss','> 本节指经典、带参考模型的 DPO。参考模型提供相对基准，但不是对所有行为变化的硬性保证；DPO 变体可能采用不同参考机制。完整回答的 logp 应按回答 Token 的条件 logp 求和，排除 Prompt 与 PAD。\n\n## 十四、用 PyTorch 手写 DPO 核心 Loss')
            p = p.replace('# 第十四章预告：','# 待续 · 第十四章预告：')
        # Consistent heading hierarchy without touching code or formulas.
        chunks = re.split(r'(```.*?```)',p,flags=re.S)
        for j in range(0,len(chunks),2):
            def heading(m):
                title=m.group(1)
                if re.match(r'第[一二三四五六七八九十]+幕|待续',title): return '## '+title
                if re.match(r'[一二三四五六七八九十]+、',title): return '### '+title
                if re.match(r'\d+\.',title): return '#### '+title
                return '**'+title+'**'
            chunks[j] = re.sub(r'^#{1,3} (.+)$',heading,chunks[j],flags=re.M)
        p=''.join(chunks).strip()
        answer='\n\n<details class="answers" markdown="1">\n<summary>展开本章五道题的参考答案</summary>\n\n' + '\n\n'.join(f'{i}. {a}' for i,a in enumerate(answers[number],1)) + '\n\n</details>\n'
        if number == 13:
            p = p.replace('## 待续 · 第十四章预告：',answer+'\n## 待续 · 第十四章预告：')
        else:
            p += answer
        p = '> 阅读提示：'+notes[number]+'\n\n'+p
        (ROOT/'chapters'/f'{number:02}.md').write_text(p+'\n',encoding='utf-8')
    old = ROOT/'chapters/10.md'
    text = old.read_text(encoding='utf-8')
    text = re.split(r'^## (?:待续 · 第十一章预告：|从 Block 到 GPT)',text,maxsplit=1,flags=re.M)[0].rstrip()
    text += '\n\n## 从 Block 到 GPT\n\n林默已经组装好了 Transformer Block，却还想知道：“它怎样学会说话？”\n\n陈教授在黑板上留下一个词：预测。带着这个问题，进入第十一章。\n'
    old.write_text(text,encoding='utf-8')
    print('Imported chapters 11-13 and joined chapter 10; continuation original preserved.')


if __name__ == '__main__':
    main()
