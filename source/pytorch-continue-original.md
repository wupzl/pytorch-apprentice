# 《PyTorch：炼金术士的学徒》

## 第十一章：预测下一个词——从 Transformer 到 GPT 的诞生

第十一天。

凌晨两点。

林默站在实验室中央，面前摆放着十天来制造的所有零件。

```
Tensor
  ↓
Autograd
  ↓
nn.Module
  ↓
Embedding
  ↓
Self-Attention
  ↓
LayerNorm
  ↓
FFN
  ↓
Transformer Block
```

他已经能够用 PyTorch 构建一个 Transformer Block。

甚至可以将它堆叠成几十层的深度网络。

但一个问题始终困扰着他。

“教授。”

“我们现在拥有 Transformer 了。”

“可是它只会把输入变成向量。”

“它究竟是怎么学会回答问题、写代码、生成文章的？”

陈教授放下咖啡。

“你认为，训练一个会说话的模型，需要什么？”

林默想了想。

“海量问答数据？”

“不一定。”

“人工标注的正确答案？”

“也不一定。”

教授打开电脑，在屏幕上写下：

> 今天，我来到实验室，开始学习 PyTorch。

然后，他删除了最后一个词。

屏幕变成：

> 今天，我来到实验室，开始学习 \_\_\_\_\_\_

“猜猜下一个词是什么？”

“PyTorch。”

教授点头。

“如果我告诉你，一种语言模型预训练的核心任务，就是反复做这件事呢？”

林默愣住。

“你是说……”

“只需要不断预测下一个 Token？”

“没错。”

教授在黑板上写下一个名字：

# GPT

Generative Pre-trained Transformer

“今天，我们将亲手制造一个能够生成文字的模型。”

## 一、语言模型的终极任务：预测下一个 Token

教授写下一段文本：

```
我 喜欢 学习 PyTorch
```

假设经过 Tokenizer 后得到：

```
[我, 喜欢, 学习, PyTorch]
```

每个 Token 有自己的 ID：

```
我       → 1
喜欢     → 2
学习     → 3
PyTorch  → 4
```

于是文本变成：

```
tokens = [1, 2, 3, 4]
```

林默问：

“然后呢？”

教授说：

“如果要让模型学会预测下一个 Token，我们应该怎么构造训练数据？”

他在黑板上画出：

```
输入：我
目标：喜欢

输入：我 喜欢
目标：学习

输入：我 喜欢 学习
目标：PyTorch
```

林默点头。

“相当于不断向后移动一个位置。”

“对。”

这就是：

Next Token Prediction——下一个 Token 预测。

数学上，我们希望模型学习：

\\[ P(x_t\mid x_1,x_2,\ldots,x\_{t-1}) \\]

意思是：

> 已经知道前面的 Token，预测下一个 Token 出现的概率。

教授继续：

“比如当前输入是：”

```
我 喜欢 学习
```

模型可能输出：

```
PyTorch   0.70
Java      0.15
Python    0.10
其他      0.05
```

然后根据这些概率选择下一个 Token。

这便是文本生成的起点。

## 二、最关键的技巧：训练数据整体向右错位一格

林默开始写代码。

“教授，难道每句话都要拆成很多训练样本？”

```
[我] → [喜欢]

[我, 喜欢] → [学习]

[我, 喜欢, 学习] → [PyTorch]
```

教授摇头。

“可以这么理解，但实际训练还有更高效的做法。”

他写：

```
tokens = torch.tensor([    1, 2, 3, 4, 5])x = tokens[:-1]y = tokens[1:]
```

得到：

```
x = [1, 2, 3, 4]

y = [2, 3, 4, 5]
```

林默仔细观察。

“所以 x 和 y 的区别……”

“就是 y 比 x 向后移动了一个 Token。”

教授点头。

Next Token Prediction 的标签构造

我

位置 1

喜欢

位置 2

学习

位置 3

PyTorch

位置 4

。

位置 5

输入与目标错位

| 输入 X | 我  | 喜欢 | 学习      | PyTorch |
| ---- | -- | -- | ------- | ------- |
| 目标 Y | 喜欢 | 学习 | PyTorch | 。       |

每个输入位置的输出，都被用来预测紧随其后的 Token。

“这样，我们就可以在一次前向传播中，同时训练多个位置的预测任务。”

林默突然想起之前学过的 GPU。

“也就是说，我们可以把整个序列交给 GPU，利用矩阵计算并行训练？”

“正是如此。”

## 三、一个严重问题：模型会不会偷看答案？

教授继续写：

```
输入：

我 喜欢 学习 PyTorch
```

“假设我们正在训练第一个位置预测‘喜欢’。”

林默说：

“模型应该只看到‘我’。”

“但是，如果使用普通 Self-Attention——”

林默突然反应过来。

```
我
│
├── 可以关注 喜欢
├── 可以关注 学习
└── 可以关注 PyTorch
```

“那模型直接看到了未来！”

教授点头。

“所以 GPT 不能使用毫无限制的双向 Self-Attention。”

“必须使用——”

# Causal Mask

## 四、Causal Mask：禁止偷看未来

假设有四个 Token：

```
A B C D
```

没有 Mask：

```
A → A B C D
B → A B C D
C → A B C D
D → A B C D
```

但是 GPT 的因果注意力要求：

```
A → A
B → A B
C → A B C
D → A B C D
```

Causal Attention Mask



A

B

C

D

A

B

C

D

允许关注

禁止关注未来

教授说：

“对于每个位置，模型只能利用当前及之前的 Token 表示。”

“这样，虽然所有位置的计算可以在训练时并行完成，但每个位置仍然遵守不能看未来的约束。”

他写出代码：

```
seq_len = 4mask = torch.triu(    torch.ones(seq_len, seq_len, dtype=torch.bool),    diagonal=1)
```

在 `nn.MultiheadAttention` 中，可以这样使用：

```
output, _ = attention(    x, x, x,    attn_mask=mask,    need_weights=False)
```

这里的 `True` 表示禁止关注该位置。

“注意，”教授补充，“PyTorch 不同 Attention API 对布尔 Mask 的约定不完全相同，用时要看具体接口。”

## 五、现在开始建造 GPT

教授在黑板上画出真正的架构：

```
                  输入文本
                     │
                     ▼
                  Tokenizer
                     │
                     ▼
                   Token IDs
                     │
             ┌───────┴────────┐
             ▼                ▼
       Token Embedding    Position Embedding
             │                │
             └───────┬────────┘
                     ▼
                    相加
                     │
                     ▼
             Transformer Block 1
                     │
                     ▼
             Transformer Block 2
                     │
                     ▼
                    ...
                     │
                     ▼
             Transformer Block N
                     │
                     ▼
                 Final Norm
                     │
                     ▼
                LM Head
                     │
                     ▼
                  Logits
                     │
                     ▼
             下一个 Token 概率
```

林默盯着：

```
LM Head
```

“这又是什么？”

教授解释：

“你已经知道 Transformer 输出的是隐藏向量。”

假设：

```
hidden_size = 128
```

那么每个 Token 的最终表示：

```
[128]
```

但词表可能有：

```
vocab_size = 10000
```

我们需要预测：

> 这 10000 个 Token 中，下一个究竟是哪一个？

所以需要：

```
nn.Linear(128, 10000)
```

也就是：

\\[ \boxed{LMHead:\mathbb{R}^{128}\rightarrow\mathbb{R}^{10000}} \\]

它会把隐藏向量投影成词表中所有候选 Token 的分数。

这些分数仍然叫：

Logits。

## 六、LM Head 输出什么 Shape？

假设输入：

```
batch_size = 2
seq_len = 8
hidden_size = 128
vocab_size = 10000
```

Transformer 输出：

```
[2, 8, 128]
```

经过：

```
lm_head = nn.Linear(128, 10000)
```

得到：

```
[2, 8, 10000]
```

林默看着最后的 `10000`。

“也就是说，每个 Token 位置都会预测一次整个词表？”

“对。”

```
位置 1 → 10000 个 logits
位置 2 → 10000 个 logits
位置 3 → 10000 个 logits
...
位置 8 → 10000 个 logits
```

教授继续：

“这就是为什么训练时，一次前向传播可以得到整个序列各位置的预测结果。”

## 七、训练 GPT 时，Loss 怎么算？

林默很快想起第四章。

“CrossEntropyLoss？”

“没错。”

模型输出：

```
logits:

[B, N, V]
```

标签：

```
targets:

[B, N]
```

其中：

```
B = Batch Size
N = Sequence Length
V = Vocabulary Size
```

但普通的：

```
nn.CrossEntropyLoss()
```

对于二维分类 logits，通常期望：

```
[样本数量, 类别数量]
```

于是我们可以把：

```
[B, N, V]
```

变成：

```
[B × N, V]
```

标签：

```
[B, N]
```

变成：

```
[B × N]
```

代码：

```
loss = F.cross_entropy(    logits.reshape(-1, vocab_size),    targets.reshape(-1))
```

教授强调：

“这里不用手动 Softmax。”

林默立刻回答：

“因为 CrossEntropyLoss 接收原始 logits。”

“很好。”

# 八、真正动手：用 PyTorch 制造 Mini-GPT

教授关闭黑板。

“现在我们要把前十章的知识全部用起来。”

“这次不只是一个 Transformer Block。”

“而是一个具备以下能力的完整教学模型。”

- 字符级 Tokenizer
- Token Embedding 与 Position Embedding
- Causal Multi-Head Attention
- Transformer Block
- LM Head
- Next Token Prediction
- CrossEntropyLoss
- AdamW 训练
- 自回归文本生成

“为了让代码可以在普通电脑上运行，我们使用非常小的模型。”

“注意，这个模型只是教学示例，不可能达到真正大语言模型的能力。”

## 1. 准备文本

假设我们有一段训练语料：

```
text = """林默走进实验室。教授正在研究PyTorch。神经网络正在学习语言。林默问教授，Transformer为什么能够生成文本？教授回答，因为模型能够预测下一个Token。"""
```

真实大模型会使用海量文本。

但我们先让模型学习这段小语料。

为了演示方便，今天使用字符级 Tokenizer。

也就是：

```
一个字符 → 一个 Token
```

这比真正 LLM 使用的子词 Tokenizer 简单得多，但不影响我们学习模型结构。

## 2. 建立词表

```
import torchimport torch.nn as nnimport torch.nn.functional as Fchars = sorted(set(text))vocab_size = len(chars)stoi = {    ch: i    for i, ch in enumerate(chars)}itos = {    i: ch    for ch, i in stoi.items()}
```

林默问：

“`stoi` 和 `itos` 是什么？”

教授写：

```
stoi：

String → Integer


itos：

Integer → String
```

例如：

```
"林" → 12
"默" → 28
"教" → 17
```

具体 ID 由词表排序决定。

编码：

```
def encode(s):    return [stoi[c] for c in s]
```

解码：

```
def decode(ids):    return "".join(itos[i] for i in ids)
```

“这样，文本就可以在字符与整数之间转换。”

## 3. 准备训练数据

```
data = torch.tensor(    encode(text),    dtype=torch.long)
```

然后设置：

```
block_size = 32batch_size = 16
```

教授解释：

```
block_size
→ 每个训练样本的输入 Token 数量

batch_size
→ 每次训练取多少个序列片段
```

编写批次生成函数：

```
def get_batch():    starts = torch.randint(        0,        len(data) - block_size,        (batch_size,)    )    x = torch.stack([        data[i:i + block_size]        for i in starts    ])    y = torch.stack([        data[i + 1:i + block_size + 1]        for i in starts    ])    return x, y
```

林默观察：

```
x：

[B, 32]


y：

[B, 32]
```

两个 Tensor 形状相同。

只是目标向后错了一位。

这里随机抽取片段用于教学；大型预训练的数据流水线通常要复杂得多。

## 4. 构建 Transformer Block

我们继续使用上一章的 Pre-Norm 结构，但加入因果 Mask。

```
class TransformerBlock(nn.Module):    def __init__(self, d_model, num_heads):        super().__init__()        self.ln1 = nn.LayerNorm(d_model)        self.attn = nn.MultiheadAttention(            embed_dim=d_model,            num_heads=num_heads,            batch_first=True        )        self.ln2 = nn.LayerNorm(d_model)        self.ffn = nn.Sequential(            nn.Linear(d_model, 4 * d_model),            nn.GELU(),            nn.Linear(4 * d_model, d_model)        )    def forward(self, x):        T = x.size(1)        # 禁止关注未来位置        causal_mask = torch.triu(            torch.ones(                T, T,                device=x.device,                dtype=torch.bool            ),            diagonal=1        )        # Attention        h = self.ln1(x)        attn_out, _ = self.attn(            h, h, h,            attn_mask=causal_mask,            need_weights=False        )        x = x + attn_out        # Feed-Forward        h = self.ln2(x)        x = x + self.ffn(h)        return x
```

教授指着：

```
T = x.size(1)
```

“这里就是序列长度。”

“为什么每次都根据 x 取？”

林默回答：

“因为训练和生成时，输入序列长度可能不同。”

“对。”

## 5. 构建 Mini-GPT

终于到了核心代码。

```
class MiniGPT(nn.Module):    def __init__(        self,        vocab_size,        d_model=64,        num_heads=4,        num_layers=2,        block_size=32    ):        super().__init__()        self.block_size = block_size        # Token Embedding        self.token_embedding = nn.Embedding(            vocab_size, d_model        )        # Position Embedding        self.position_embedding = nn.Embedding(            block_size, d_model        )        # Transformer Blocks        self.blocks = nn.Sequential(*[            TransformerBlock(d_model, num_heads)            for _ in range(num_layers)        ])        # 最终归一化        self.ln_f = nn.LayerNorm(d_model)        # 输出到词表        self.lm_head = nn.Linear(            d_model, vocab_size        )    def forward(self, idx, targets=None):        B, T = idx.shape        if T > self.block_size:            raise ValueError("序列长度超过 block_size")        # Token Embedding        tok_emb = self.token_embedding(idx)        # Position Embedding        positions = torch.arange(            T,            device=idx.device        )        pos_emb = self.position_embedding(            positions        )        # 合并        x = tok_emb + pos_emb        # Transformer        x = self.blocks(x)        # Final Norm        x = self.ln_f(x)        # LM Head        logits = self.lm_head(x)        loss = None        if targets is not None:            loss = F.cross_entropy(                logits.reshape(-1, logits.size(-1)),                targets.reshape(-1)            )        return logits, loss
```

林默沉默了。

这是他第一次真正看到：

> 一个从 Token ID 开始，一直到整个词表预测结果的完整语言模型。

教授问：

“这段代码，有没有看不懂的？”

林默指着：

```
self.blocks = nn.Sequential(*[    TransformerBlock(d_model, num_heads)    for _ in range(num_layers)])
```

“这里的 `*` 是什么？”

教授解释：

“Python 的参数解包语法。”

例如：

```
layers = [    TransformerBlock(64, 4),    TransformerBlock(64, 4)]
```

使用：

```
nn.Sequential(*layers)
```

等价于把这些层依次传给 `nn.Sequential`。

所以我们就能轻松控制：

```
num_layers = 2

num_layers = 6

num_layers = 12
```

# 九、Mini-GPT 的每个 Tensor 形状

教授说：

“现在先不要运行。”

“我们从头到尾推一次 Shape。”

假设：

```
Batch Size = 16
Sequence Length = 32
Hidden Size = 64
Vocabulary Size = V
```

Mini-GPT 前向传播 Shape 流动

| Token IDs          | `[16, 32]`     |
| ------------------ | -------------- |
| Token Embedding    | `[16, 32, 64]` |
| Position Embedding | `[32, 64]`     |
| 相加（广播）             | `[16, 32, 64]` |
| Transformer Blocks | `[16, 32, 64]` |
| Final LayerNorm    | `[16, 32, 64]` |
| LM Head            | `[16, 32, V]`  |
| CrossEntropy 输入    | `[512, V]`     |

林默看着：

```
[16, 32, V]
```

问：

“那是不是一批有 16 条序列，每条 32 个位置，每个位置都预测 V 个候选 Token？”

教授点头。

“正确。”

“这也是训练能够高效利用 GPU 的关键之一。”

# 十、让模型真正训练起来

现在开始创建模型。

```
device = torch.device(    "cuda" if torch.cuda.is_available()    else "cpu")model = MiniGPT(    vocab_size=vocab_size,    d_model=64,    num_heads=4,    num_layers=2,    block_size=32).to(device)
```

优化器：

```
optimizer = torch.optim.AdamW(    model.parameters(),    lr=3e-4)
```

训练：

```
model.train()for step in range(1000):    x, y = get_batch()    x = x.to(device)    y = y.to(device)    optimizer.zero_grad()    logits, loss = model(x, y)    loss.backward()    optimizer.step()    if step % 100 == 0:        print(            f"Step {step}, Loss: {loss.item():.4f}"        )
```

林默看着整套训练过程。

“原来训练 GPT 和第一章的训练循环，本质没有变。”

教授点头。

第一章：

```
pred = model(x)loss = loss_fn(pred, y)loss.backward()optimizer.step()
```

今天：

```
logits, loss = model(x, y)loss.backward()optimizer.step()
```

核心仍然是：

```
Forward
    ↓
Loss
    ↓
Backward
    ↓
Optimizer
```

变化的只是模型结构和训练目标。

教授提醒：“由于语料极少，这个模型很容易记住训练文本，不能据此认为它具有真实的大规模语言理解能力。”

# 十一、训练完了，为什么还不会自动生成文本？

林默兴奋地运行：

```
logits, loss = model(x)
```

屏幕上出现：

```
torch.Size([16, 32, V])
```

他愣住。

“怎么还是一堆数字？”

教授问：

“你希望模型输出什么？”

“文本。”

“但模型只负责计算 logits。”

“你需要自己建立一个生成循环。”

他写下：

# Autoregressive Generation

自回归生成。

## 十二、生成的第一步：只看最后一个位置

假设输入：

```
林默走进
```

Token IDs：

```
[12, 28, 7, 35]
```

模型前向传播：

```
logits, _ = model(idx)
```

输出：

```
[1, 4, V]
```

林默问：

“为什么预测结果有四个位置？”

教授回答：

“因为模型对输入中的每个位置，都产生了预测下一个 Token 的 logits。”

但现在我们要继续生成文本。

所以只需要：

```
last_logits = logits[:, -1, :]
```

形状：

```
[1, V]
```

这代表：

> 基于目前整段上下文，预测紧随其后的下一个 Token。

# 十三、怎么从 Logits 选择一个 Token？

教授给出：

```
模型预测：

实验室  3.2
教室    2.8
医院    1.1
广场    0.7
```

林默说：

“直接取最大值？”

```
next_token = last_logits.argmax(dim=-1)
```

教授点头。

“这叫 Greedy Decoding。”

即：

> 每次选择当前概率最高的 Token。

但问题来了。

“如果每次都选最大值，会怎样？”

林默想了想。

“生成结果会比较固定？”

“对。”

而且贪心策略并不保证生成整段文本的联合概率最高。

它只是在每一个局部步骤做最大值选择。

所以现代语言模型经常使用：

> Sampling——采样。

# 十四、Sampling：按概率抽取下一个 Token

假设 Softmax 后：

```
实验室  0.60
教室    0.25
医院    0.10
广场    0.05
```

不是直接选择：

```
实验室
```

而是按这些概率随机抽取。

因此：

```
第一次生成：
实验室

第二次生成：
实验室

第三次生成：
教室
```

不同运行可能得到不同结果。

代码：

```
probs = F.softmax(    last_logits,    dim=-1)next_token = torch.multinomial(    probs,    num_samples=1)
```

林默盯着：

```
torch.multinomial()
```

“这就是按照概率抽样？”

“对。”

# 十五、Temperature：控制生成有多随机

教授又写：

```
temperature = 1.0
```

然后：

```
logits = logits / temperature
```

林默想起 Attention 中的缩放。

“又是除法？”

教授点头。

“但这次目的不同。”

Temperature 控制的是：

> 采样概率分布的尖锐程度。

公式：

\\[ P_i= \frac{e^{z_i/T}} {\sum_j e^{z_j/T}} \\]

其中：

\\[ T>0 \\]

温度越低：

```
概率越集中于高 logit 的 Token
```

温度越高：

```
概率分布越平缓
```

假设 logits：

```
[3, 2, 1]
```

Temperature 采样实验

T = 1.0

低随机性

1.0

高随机性

实验室

66.5%

教室

24.5%

医院

9.0%

拖动温度，观察相同 logits 如何产生不同的采样概率。T 趋近 0 时越来越接近贪心选择。

林默发现：

“所以 Temperature 不改变模型参数。”

“只是改变生成时的概率分布？”

教授点头。

“它是推理解码策略的一部分。”

# 十六、Top-k：只允许最有希望的 k 个候选参与

教授继续：

“假设词表有 100000 个 Token。”

“你真的希望每次生成时，所有 Token 都有机会被采样吗？”

林默摇头。

“那些概率极小的 Token，可能生成奇怪文本。”

“所以可以使用 Top-k。”

例如：

```
Top-k = 3
```

原始概率：

```
A  0.50
B  0.25
C  0.15
D  0.06
E  0.04
```

只保留：

```
A
B
C
```

其他候选：

```
D、E
```

直接排除。

然后对保留下来的概率重新归一化。

PyTorch：

```
values, _ = torch.topk(    logits,    k=3)threshold = values[:, [-1]]logits = logits.masked_fill(    logits < threshold,    float("-inf"))
```

之后 Softmax，低于阈值的 Token 就不会被采样。

# 十七、Top-p：候选数量不固定

林默问：

“Top-k 只保留固定数量，是不是有点死板？”

教授点头。

于是出现：

# Top-p / Nucleus Sampling

它不固定候选 Token 数量。

而是选择概率最高的一组 Token，使累计概率达到设定阈值。

例如：

```
top_p = 0.9
```

概率排序：

```
A  0.50  累计 0.50
B  0.25  累计 0.75
C  0.15  累计 0.90
D  0.06  累计 0.96
E  0.04  累计 1.00
```

那么可以保留：

```
A、B、C
```

因为累计概率已经达到：

```
0.90
```

教授说：

“当模型很确定时，可能只需要几个候选。”

“当模型不确定时，可能需要更多候选。”

这就是 Top-p 相比固定 Top-k 的一个特点。

# 十八、完成文本生成函数

教授说：

“我们先实现一个最简单的采样版本，不使用 Top-k 和 Top-p。”

```
@torch.no_grad()def generate(    model,    idx,    max_new_tokens,    temperature=1.0):    if temperature <= 0:        raise ValueError("temperature 必须大于 0")    model.eval()    for _ in range(max_new_tokens):        # 保留最近 block_size 个 Token        idx_cond = idx[:, -model.block_size:]        # 前向传播        logits, _ = model(idx_cond)        # 只取最后一个位置的 logits        logits = logits[:, -1, :]        # Temperature        logits = logits / temperature        # Softmax        probs = F.softmax(            logits,            dim=-1        )        # Sampling        next_token = torch.multinomial(            probs,            num_samples=1        )        # 拼接新 Token        idx = torch.cat(            [idx, next_token],            dim=1        )    return idx
```

林默看向：

```
idx = torch.cat(    [idx, next_token],    dim=1)
```

“每轮把生成的 Token 拼回输入。”

“下一轮继续预测。”

“这就是自回归？”

“没错。”

# 十九、生成第一段文本

现在模型已经训练好了。

假设输入：

```
林默
```

编码：

```
start_ids = encode("林默")
```

转换成 Tensor：

```
context = torch.tensor(    [start_ids],    dtype=torch.long,    device=device)
```

执行：

```
generated = generate(    model,    context,    max_new_tokens=100,    temperature=0.8)
```

解码：

```
output_text = decode(    generated[0].tolist())print(output_text)
```

可能输出：

```
林默走进实验室。教授正在研究PyTorch。
```

也可能输出不连贯的文字。

教授提醒：

“由于训练语料非常少，而且这里只是一个微型模型，生成结果可能重复、混乱或者只是机械记忆训练文本。”

“如果训练数据规模和模型能力不足，不要期待它能真正理解复杂问题。”

# 二十、一个更深的问题：为什么生成时只预测一个 Token？

林默问：

“教授，既然训练时能同时预测整个序列，为什么生成时不能一次生成 100 个 Token？”

教授在黑板上写：

```
训练：

已知完整序列
```

例如：

```
我 喜欢 学习 PyTorch
```

我们已经知道每个位置的真实 Token。

所以：

```
位置 1
位置 2
位置 3
位置 4
```

可以并行计算对应的下一 Token 预测。

但是生成时：

```
我 喜欢
```

下一 Token 未知。

假设先生成：

```
学习
```

才能继续预测：

```
PyTorch
```

因此：

```
生成第 3 个 Token
依赖第 2 个 Token

生成第 4 个 Token
依赖第 3 个 Token
```

形成顺序依赖。

教授说：

“这就是为什么自回归语言模型的训练并行性很强，但逐 Token 解码仍有明显的顺序瓶颈。”

当然，实际系统可以通过推测解码等技术优化，不代表所有生成技术都严格采用一次只计算一个 Token 的简单方式。

# 二十一、如果每次都重新计算全部上下文，会发生什么？

林默仔细研究生成函数：

```
idx_cond = idx[:, -model.block_size:]
```

然后：

```
logits, _ = model(idx_cond)
```

他发现了一个问题。

“教授，我们每生成一个新 Token，都重新计算之前的 Token？”

“这个教学版本是的。”

例如：

```
第 1 步：

A B C
```

执行 Attention。

下一步：

```
A B C D
```

又执行 Attention。

再下一步：

```
A B C D E
```

继续执行。

林默皱眉。

“那 A、B、C 不是被重复计算了？”

教授点头。

“这就是这个简单生成器最大的性能问题之一。”

“真正的大模型推理系统，不会这么浪费。”

教授在黑板上写下：

# KV Cache

# 二十二、KV Cache：把过去的计算结果保存起来

教授回顾上一章：

\\[ Q=XW_Q \\]

\\[ K=XW_K \\]

\\[ V=XW_V \\]

在自回归推理中，新 Token 需要关注之前的 Token。

对于此前已经计算过的 Token，它们在某一层的 Key 和 Value 可以缓存起来。

例如：

```
已经生成：

A B C
```

保存：

```
K_A
K_B
K_C

V_A
V_B
V_C
```

下一步生成 D：

```
只需要计算新 Token 的相关投影
```

然后把新的 K、V 加入缓存。

```
KV Cache

┌──────────────────────────┐
│ K_A  K_B  K_C  K_D      │
│ V_A  V_B  V_C  V_D      │
└──────────────────────────┘
```

后续新 Query 就可以关注这些历史 K/V。

林默突然意识到：

“所以不用每次重新计算历史 Token 的所有 Attention 投影。”

“对。”

不过教授强调：

“KV Cache 不是把所有计算成本都消灭了。”

“新 Token 仍然需要和已有 Key 计算 Attention，而且缓存本身占用显存。”

“随着上下文长度增加，KV Cache 也会增长。”

# 二十三、Prefill 与 Decode

教授在黑板上写两个阶段。

```
Prefill

Decode
```

林默问：

“这又是什么？”

教授回答：

“这是现代 LLM 推理工程里非常重要的两个阶段。”

## Prefill 阶段

用户输入：

```
请解释一下 PyTorch 的自动求导机制。
```

假设 Tokenizer 生成：

```
100 个 Token
```

模型需要处理这 100 个 Token，建立它们的上下文表示和各层 KV Cache。

这一阶段：

```
输入 Prompt
     ↓
批量处理 Prompt Token
     ↓
建立 KV Cache
     ↓
预测第一个新 Token
```

通常称为：

Prefill。

## Decode 阶段

开始生成：

```
第 1 个 Token
第 2 个 Token
第 3 个 Token
...
```

借助 KV Cache，通常只需要处理新加入的 Token，而不用把整个前缀从头经过每一层。

这一阶段叫：

Decode。

# 二十四、为什么 Prefill 和 Decode 的性能瓶颈不同？

教授继续。

“Prefill 处理较长的输入序列，通常有较大的矩阵计算量。”

“Decode 每一步只增加少量 Token，却需要读取大量模型权重和 KV Cache。”

因此在实际硬件与批处理配置下，经常出现：

```
Prefill
→ 更容易利用大规模并行计算
→ 计算吞吐是重要因素

Decode
→ 单步计算规模较小
→ 内存带宽和访存成本常常成为重要瓶颈
```

不过这不是绝对定律，模型规模、Batch Size、序列长度、硬件和实现方式都会影响瓶颈。

林默点头。

“所以模型推理优化，并不只是换一张更强的显卡。”

教授说：

“还涉及 KV Cache、量化、批处理、Attention Kernel、并行策略。”

# 二十五、现在才真正理解 GPT 的名字

教授重新写下：

# GPT

Generative Pre-trained Transformer

林默开始逐个拆解。

Generative——生成式

模型能够根据已有上下文，逐步生成新的 Token。

Pre-trained——预训练

模型先在大量文本上进行训练，学习语言和知识中的统计规律。

Transformer——模型架构

使用基于 Attention 的 Transformer 架构。

教授补充：

“GPT 系列属于 Decoder-Only 自回归 Transformer 模型的代表。”

“但不是所有生成模型都使用完全相同的结构，也不是所有 Transformer 都是 GPT。”

# 二十六、但一个会预测 Token 的模型，为什么还能回答问题？

林默仍然不满足。

“教授，预测下一个 Token，我已经懂了。”

“但如果我问：”

> Java HashMap 为什么线程不安全？

“模型为什么能回答这种技术问题？”

教授没有直接回答。

他写：

```
训练文本中可能存在：

HashMap 在并发修改时……
```

“如果模型在海量数据中学习过大量有关 Java、并发、集合、系统设计的内容，它就可能掌握这些文本模式和部分相关知识。”

“当你输入一个问题时，模型会根据上下文预测可能的后续回答。”

林默问：

“所以它只是在记住文字？”

教授摇头。

“不能简单这么说。”

“模型不仅可能记忆训练片段，也可能通过参数学习到更一般的统计结构、概念关联和可迁移的计算模式。”

“但也不能因此认为它一定真正理解了所有问题。”

教授写：

```
Next Token Prediction
         │
         ▼
学习语言结构
         │
         ▼
学习概念关联与模式
         │
         ▼
形成一定的泛化能力
         │
         ▼
可能执行问答、总结、翻译、代码生成等任务
```

“至于为什么这些能力会出现，以及模型如何通过后训练变得更善于遵循指令，是更深入的问题。”

# 二十七、预训练模型为什么还需要 SFT？

林默突然想到自己做过的 Coding Agent。

“教授，ChatGPT 这种模型不是专门按照用户指令回答问题吗？”

“但预训练只是预测下一个 Token。”

“那它为什么知道应该当一个助手？”

教授点头。

“这就涉及另一个阶段。”

# Post-training

也就是预训练之后的训练。

其中一个重要方法是：

SFT——Supervised Fine-Tuning，监督微调。

例如：

```
用户：
解释一下 Java ThreadLocal。

助手：
ThreadLocal 为每个线程提供……
```

模型在大量这样的指令与回答数据上继续训练。

它逐渐学习：

```
看到用户问题
    ↓
按照指令
    ↓
生成符合要求的回答
```

教授补充：

“现代模型后训练还可能涉及偏好优化、奖励建模、强化学习等方法。”

“但 SFT 是理解指令微调的一个很好的起点。”

# 二十八、林默突然把所有知识连接起来了

他站在黑板前，写下了今天最终的系统结构：

```
                  大规模文本语料
                        │
                        ▼
                    Tokenizer
                        │
                        ▼
                    Token IDs
                        │
                        ▼
                 构造 x / target
                        │
                        ▼
                 Token Embedding
                        │
                        ▼
                Position Embedding
                        │
                        ▼
                Transformer Blocks
                        │
                        ▼
                      LM Head
                        │
                        ▼
                     Logits
                        │
                        ▼
                CrossEntropyLoss
                        │
                        ▼
                    backward()
                        │
                        ▼
                     AdamW
                        │
                        ▼
                    参数更新
                        │
                        ▼
                     预训练模型
                        │
                        ▼
                    SFT / 后训练
                        │
                        ▼
                     助手模型
                        │
                        ▼
                     推理服务
                        │
                        ▼
                Prefill + Decode
                        │
                        ▼
                KV Cache + Sampling
                        │
                        ▼
                     生成文本
```

教授看着这张图。

“你现在终于可以回答一个问题了。”

“什么问题？”

“一个 LLM 从训练数据到最终生成回答，究竟经历了什么？”

林默沉默了一会儿。

然后回答：

“首先把文本 Tokenize 成整数 ID。”

“通过 Embedding 转成向量。”

“利用带因果约束的 Transformer，对每个位置建立上下文表示。”

“通过 LM Head 预测词表 logits。”

“用错位的下一 Token 标签和 CrossEntropyLoss 训练，再用 Autograd 与优化器更新参数。”

“训练完成后，可以通过后训练增强指令遵循能力。”

“推理时从 Prompt 开始，不断预测和选择新 Token，并通过 KV Cache 避免大量重复计算。”

教授点头。

“这就是一条完整的 LLM 技术主线。”

## 二十九、今天的五道思考题

Mini-GPT 理解测试

0/5 已完成

1\. 为什么 GPT 训练时可以同时预测多个位置？

因为模型可以提前知道自己会生成什么

因为真实 Token 序列已知，使用 Causal Mask 后可并行计算各位置

因为训练时不需要 Attention

2\. LM Head 的主要作用是什么？

把隐藏向量映射成词表大小的 logits

把文字分割成 Token

缓存过去的 K 和 V

3\. Temperature 降低通常会造成什么效果？

增加模型参数量

让概率分布更平缓

让高概率 Token 更占优势

4\. KV Cache 主要缓存什么？

过去每个位置的最终生成概率

历史 Token 各层 Attention 的 Key 和 Value

模型所有梯度

5\. 为什么简单的自回归推理不能像训练一样，一次确定所有未来 Token？

因为未来 Token 尚未知，后续生成依赖前面生成结果

因为 GPU 不支持矩阵乘法

因为推理时不允许使用 Softmax

# 下一章：《显存里的战争——混合精度、梯度累积、梯度检查点与 LoRA》

凌晨三点。

林默兴奋地修改模型：

```
model = MiniGPT(    vocab_size=50000,    d_model=4096,    num_heads=32,    num_layers=32,    block_size=4096)
```

然后点击运行。

一秒钟后，终端里出现一行红字：

```
torch.OutOfMemoryError:
CUDA out of memory
```

林默脸上的笑容消失了。

“教授……”

“我的 24GB 显卡连模型都放不下？”

教授走到屏幕前。

“你已经学会制造模型了。”

“但你还没学会在有限硬件上训练它。”

他在黑板上写出：

```
FP32
FP16
BF16

Automatic Mixed Precision

Gradient Accumulation

Gradient Checkpointing

LoRA / QLoRA

FlashAttention
```

“下一章，我们正式进入 PyTorch 大模型训练的工程世界。”

第十二章将深入讲解：

- 为什么 7B 模型的 FP32 权重就需要约 28GB 显存，而真正训练需要更多。
- FP32、FP16、BF16 的数值范围和精度为什么不同。
- `torch.autocast` 和 `GradScaler` 分别解决什么问题。
- 为什么梯度累积可以模拟更大的有效 Batch Size。
- Activation Checkpointing 如何通过重新计算换取显存空间。
- LoRA 为什么能在冻结原模型参数的情况下训练适配器。
- QLoRA 为什么能够进一步减少微调的显存要求。
- FlashAttention 为什么不改变标准 Attention 的数学目标，却能显著优化内存访问与中间存储。

下一章，我们将开始解决一个真实工程问题：

> 只有一张消费级 GPU，如何尽可能高效地训练或微调自己的大语言模型？


下一章。


# 《PyTorch：炼金术士的学徒》

## 第十二章：显存里的战争——混合精度、梯度累积、梯度检查点与 LoRA

从普通神经网络训练，正式进入大模型训练与微调的工程世界。

凌晨三点。

实验室的灯依旧亮着。

林默坐在一台价值不菲的工作站前，盯着终端里的一行红字。

```
torch.OutOfMemoryError:
CUDA out of memory.
```

他已经尝试了三次。

第一次，把 Batch Size 从 16 改成 8。

第二次，从 8 改成 2。

第三次，直接改成 1。

可模型还是无法正常训练。

“教授，我已经把 Batch Size 降到 1 了！”

林默指着屏幕。

“为什么还是显存不足？”

陈教授看了一眼配置：

```
model = MiniGPT(    vocab_size=50000,    d_model=4096,    num_heads=32,    num_layers=32,    block_size=4096)
```

又看了看显卡：

```
NVIDIA GPU
VRAM: 24 GB
```

教授没有修改代码。

而是在白板上写下一句话：

> 模型能放进显存，不代表模型能够在这张显卡上训练。

林默愣住。

“这两件事情有区别？”

教授拿起粉笔。

“区别大得多。”

“今天，我们不研究如何让模型变聪明。”

“我们研究如何让它活下来。”

# 第一幕：24GB 显存为什么不够？

## 一、模型参数只是显存战争的第一支军队

教授问：

“一个模型有 70 亿参数，也就是 7B。”

“如果每个参数使用 FP32 保存，需要多少内存？”

林默想起第六章。

“FP32 一个数字占 4 字节。”

于是：

\\[ 7\times10^9\times4 = 28\times10^9\text{ Bytes} \\]

约等于：

\\[ \boxed{28\text{ GB}} \\]

这里的 GB 是十进制单位；换成二进制 GiB，大约是 26.1 GiB。

林默点头。

“所以 7B 模型使用 FP32，光权重就需要约 28GB。”

“对。”

“那换成 FP16 呢？”

\\[ 7\times10^9\times2 = 14\text{ GB} \\]

林默兴奋起来。

“14GB！”

“24GB 显卡应该可以训练了吧？”

教授摇头。

“你忘了训练需要什么。”

他在白板上写出四支军队。

```
┌──────────────────────────────┐
│           GPU VRAM           │
├──────────────────────────────┤
│                              │
│  第一支军队：Model Weights   │
│           模型权重            │
│                              │
├──────────────────────────────┤
│                              │
│  第二支军队：Gradients       │
│           参数梯度            │
│                              │
├──────────────────────────────┤
│                              │
│  第三支军队：Optimizer State │
│           优化器状态          │
│                              │
├──────────────────────────────┤
│                              │
│  第四支军队：Activations     │
│           中间激活            │
│                              │
├──────────────────────────────┤
│      CUDA 工作区与临时缓存    │
└──────────────────────────────┘
```

“而你刚刚只计算了第一支军队。”

## 二、AdamW 为什么这么吃显存？

教授打开之前的代码：

```
optimizer = torch.optim.AdamW(    model.parameters(),    lr=3e-4)
```

“还记得 AdamW 吗？”

林默回答：

“它会保存梯度的一阶矩和二阶矩估计。”

“没错。”

假设我们进行常规 FP32 全参数训练，暂时不考虑激活和其他开销。

每个参数可能涉及：

| 内容        | 每参数字节数 |
| --------- | ------ |
| FP32 参数   | 4      |
| FP32 梯度   | 4      |
| AdamW 一阶矩 | 4      |
| AdamW 二阶矩 | 4      |
| 合计        | 16     |

所以 7B 模型：

\\[ 7\times10^9\times16 = 112\text{ GB} \\]

7B 模型：简化 FP32 AdamW 训练显存构成

模型权重

28 GB

梯度

28 GB

一阶矩

28 GB

二阶矩

28 GB

合计

# 112 GB

仅为指定 FP32 AdamW 配置的示意计算，尚未计入激活、临时工作区及其他开销。

林默望着自己的显卡。

“24GB……”

“连这里的四分之一都不到。”

教授点头。

“而且这个 112GB 并不是所有混合精度训练的固定显存公式。”

“不同的混合精度方案，可能保存不同精度的参数副本、梯度和优化器状态。”

“实际内存预算必须根据具体实现计算。”

## 三、为什么 Batch Size 降到 1 也不一定有用？

林默突然明白了。

“Batch Size 主要影响中间激活和输入数据。”

“但是模型参数和优化器状态，不会因为 Batch Size 从 16 降到 1，就跟着缩小十六倍！”

教授点头。

这就是大模型训练中非常重要的区分：

参数相关显存主要取决于模型大小、数据精度以及优化器设计。

激活相关显存主要受到 Batch Size、Sequence Length、隐藏维度、网络层数和算子实现影响。

所以：

```
显存不足
    │
    ├── 模型权重过大？
    │      ↓
    │   量化、分片、卸载等
    │
    ├── 梯度/优化器状态过大？
    │      ↓
    │   LoRA、优化器状态分片等
    │
    ├── 中间激活过大？
    │      ↓
    │   减小 Batch、Checkpointing
    │
    └── Attention 中间计算过大？
           ↓
        FlashAttention / SDPA
```

教授说：

“你必须先知道是谁占用了显存。”

“才能选择正确的武器。”

# 第二幕：混合精度——让数字穿上轻甲

## 四、FP32、FP16、BF16 到底是什么？

教授在白板上写下三个名字。

```
FP32

FP16

BF16
```

林默说：

“都是浮点数。”

“但分别有什么区别？”

教授回答：

“浮点数在计算机里，通常不是直接存储一个普通的十进制小数。”

“它会使用符号位、指数和有效数字字段来近似表示实数。”

三种格式的典型布局：

FP32

32 bits

1

指数 8

尾数 23

FP16

16 bits

1

指数 5

尾数 10

BF16

16 bits

1

指数 8

尾数 7

三者均包含 1 个符号位。图中尾数位数不包含规范化浮点数的隐含前导有效位。

林默仔细看。

“FP16 和 BF16 都是 16 位。”

“但 FP16 给了尾数更多位。”

“BF16 给了指数更多位？”

教授点头。

## 五、指数位与尾数位分别决定什么？

教授拿出一个比喻。

“假设你要记录天文学距离和实验室测量数据。”

“指数位更多，意味着能表示更大或更小量级的数字。”

“有效数字位更多，意味着在某个量级附近能区分更细小的数值差异。”

因此：

| 格式   | 存储空间 | 动态范围          | 有效精度 |
| ---- | ---- | ------------- | ---- |
| FP32 | 4 字节 | 大             | 高    |
| FP16 | 2 字节 | 较窄            | 中等   |
| BF16 | 2 字节 | 接近 FP32 的指数范围 | 较低   |

林默问：

“所以 BF16 更适合大模型训练？”

教授说：

“在支持 BF16 高效运算的硬件上，它通常是非常有吸引力的选择。”

“因为它具有接近 FP32 的动态范围，能减少 FP16 常见的数值溢出或下溢风险。”

“但 BF16 不是万能的。它的有效精度比 FP16 更低，某些操作仍需更高精度。”

## 六、为什么不能把所有东西都强制转成 FP16？

林默敲下：

```
model = model.half()
```

又把输入转成 FP16。

“这样不就省显存了？”

教授立即阻止。

“这属于手动降低参数精度，不等于安全的自动混合精度训练。”

因为有些操作适合低精度，例如：

```
大规模矩阵乘法
```

但有些操作对数值范围和精度更敏感，例如某些：

```
归一化
损失计算
归约计算
```

如果全部粗暴地使用 FP16，可能出现：

```
Loss = NaN

Gradient = Inf

训练突然发散
```

教授写下：

# Automatic Mixed Precision

简称：

AMP。

“它的核心思想是，让不同操作根据实现策略使用适合的数值精度，而不是所有运算都一刀切。”

## 七、PyTorch 的 `autocast`

教授开始编写一个标准 CUDA 混合精度训练循环：

```
device = torch.device("cuda")model = MyModel().to(device)optimizer = torch.optim.AdamW(    model.parameters(),    lr=3e-4)for x, y in loader:    x = x.to(device)    y = y.to(device)    optimizer.zero_grad(set_to_none=True)    with torch.autocast(        device_type="cuda",        dtype=torch.bfloat16    ):        logits = model(x)        loss = loss_fn(logits, y)    loss.backward()    optimizer.step()
```

林默盯着：

```
with torch.autocast(    device_type="cuda",    dtype=torch.bfloat16):
```

“所以在这个区域里，PyTorch 会按 AMP 的规则选择运算精度？”

“对。”

“注意，这不是把模型所有权重永久变成 BF16。”

“常见 AMP 训练仍然会保留 FP32 参数，优化器也可能使用 FP32 状态。”

这与：

```
model.to(torch.bfloat16)
```

并不是一回事。

## 八、GradScaler 为什么存在？

林默翻看其他人的代码。

突然发现：

```
scaler = torch.amp.GradScaler("cuda")
```

“教授，为什么有些训练还多一个 `GradScaler`？”

教授在黑板上写：

```
原始梯度：

0.00000001
```

“如果 FP16 无法充分表示某些非常小的梯度，它们可能下溢为零。”

“于是我们可以在反向传播之前，暂时把 Loss 放大。”

例如：

\\[ L'=1024L \\]

那么：

\\[ \frac{\partial L'}{\partial w} = 1024\frac{\partial L}{\partial w} \\]

林默明白了。

“这样梯度也放大了。”

“对。”

“然后参数更新之前，再把梯度缩回去。”

这就是 Loss Scaling 的核心思路。

FP16 AMP 训练中常见：

```
scaler = torch.amp.GradScaler("cuda")for x, y in loader:    x = x.to("cuda")    y = y.to("cuda")    optimizer.zero_grad(set_to_none=True)    with torch.autocast(        device_type="cuda",        dtype=torch.float16    ):        logits = model(x)        loss = loss_fn(logits, y)    scaler.scale(loss).backward()    scaler.step(optimizer)    scaler.update()
```

教授说：

“`GradScaler` 还会监测非有限梯度，并在必要时跳过参数更新和调整缩放比例。”

“BF16 通常不需要通过这种方式解决 FP16 的梯度下溢问题。”

林默在笔记中写下：

> autocast 负责运算精度策略；GradScaler 主要服务于 FP16 梯度数值稳定性。

# 第三幕：梯度累积——没有大货车，就多运几趟

## 九、Batch Size 为什么这么重要？

教授写：

```
目标：

Batch Size = 64
```

但是 GPU 只能装下：

```
Batch Size = 4
```

林默问：

“那我是不是没办法训练？”

教授笑了。

“你想想，训练过程中最关键的参数更新发生在哪一步？”

林默回答：

```
optimizer.step()
```

“对。”

“那假设我们不每处理一批就更新，而是先累积多个 Batch 的梯度呢？”

林默突然想起第一章。

> PyTorch 的 `.grad` 默认会累加！

## 十、原来梯度累积利用的是第一章的知识

正常训练：

```
Batch 1
   ↓
backward()
   ↓
optimizer.step()

Batch 2
   ↓
backward()
   ↓
optimizer.step()
```

而梯度累积：

```
Batch 1
   ↓
backward()
   │
   │ 不更新参数
   ▼
Batch 2
   ↓
backward()
   │
   │ 不更新参数
   ▼
Batch 3
   ↓
backward()
   │
   │ 不更新参数
   ▼
Batch 4
   ↓
backward()
   │
   ▼
optimizer.step()
```

林默眼睛亮了。

“所以我每次只喂 4 条数据。”

“连续累积 16 次。”

“就相当于一次用 64 条数据计算梯度？”

教授回答：

“在满足相应条件时，可以得到与更大 Batch 相同或非常接近的平均梯度。”

“但你需要处理好 Loss 缩放。”

## 十一、为什么 Loss 要除以累积步数？

假设：

```
micro_batch_size = 4

accumulation_steps = 16
```

有效 Batch Size：

\\[ 4\times16=64 \\]

正常情况下，我们希望得到 64 个样本的平均梯度：

\\[ g=\frac1{64}\sum\_{i=1}^{64}g_i \\]

每个 Micro-Batch 的 Loss 如果已经是四个样本的平均值，那么它的梯度是：

\\[ g_j=\frac14\sum\_{i\in B_j}g_i \\]

16 个 Batch 梯度直接累加：

\\[ \sum\_{j=1}^{16}g_j \\]

会比所需的整体平均梯度大 16 倍。

因此：

```
loss = loss / accumulation_steps
```

教授写：

```
accumulation_steps = 16optimizer.zero_grad(set_to_none=True)for step in range(accumulation_steps):    x, y = get_batch()    x = x.to(device)    y = y.to(device)    logits, loss = model(x, y)    loss = loss / accumulation_steps    loss.backward()optimizer.step()
```

林默点头。

“这样累积的梯度才对应整体平均梯度。”

教授补充：

“这个例子假设各 Micro-Batch 大小相同，Loss 按样本平均，且没有影响 Batch 间等价性的特殊操作。”

“如果最后一批大小不同，或者 Loss 按 Token 平均且有效 Token 数不同，应该按实际样本数或有效 Token 数加权。”

## 十二、一个可交互的显存训练配置实验

教授在屏幕上打开一张计算表。

“你可以自己调整 Micro-Batch Size 和累积步数，看看有效 Batch Size 如何变化。”

有效 Batch Size 计算器

每卡 Micro-Batch4

累积步数16

GPU 数量1

有效 Batch Size（样本数）

# 64

4 × 16 × 1 = 64

这个公式适用于各卡 Micro-Batch 大小一致的典型数据并行训练。对于变长文本，还需要区分样本数量和实际参与 Loss 的 Token 数量。

## 十三、梯度累积真的完全等价于大 Batch 吗？

林默突然问：

“那如果完全一样，为什么不永远用 Micro-Batch Size = 1？”

教授摇头。

“因为它们不一定完全等价。”

可能的差异包括：

- BatchNorm 等依赖当前批次统计的操作。
- Dropout 等随机操作产生不同的随机掩码。
- 浮点数计算顺序不同导致数值误差。
- Batch 间长度和有效 Token 数不同造成 Loss 加权问题。
- 多次小 Batch 前向传播带来额外运行开销。

“而且，”教授继续，“梯度累积主要降低单次前向、反向计算所需的激活显存。”

“它不会自动减少模型参数、梯度 Tensor 和 AdamW 状态所占用的内存。”

林默终于明白了：

> 梯度累积解决的是无法一次放下大 Batch 的问题，不是万能显存压缩技术。

# 第四幕：梯度检查点——忘记过去，必要时重新计算

## 十四、显存里最昂贵的记忆

教授重新打开 Transformer。

```
Transformer Block 1
        ↓
Transformer Block 2
        ↓
Transformer Block 3
        ↓
...
        ↓
Transformer Block 32
```

“你还记得为什么前向传播需要保存激活吗？”

林默回答：

“因为反向传播要用中间结果计算梯度。”

教授点头。

普通训练：

```
Forward
   │
   ├── 保存 Block 1 中间激活
   ├── 保存 Block 2 中间激活
   ├── 保存 Block 3 中间激活
   │
   └── 保存 Block 32 中间激活
              │
              ▼
           Backward
```

“可是，”教授问，“如果显存装不下全部激活怎么办？”

林默说：

“那就少保存？”

“对。”

“但是反向传播需要它们。”

林默皱眉。

教授笑了。

“所以——”

> 需要的时候重新计算。

## 十五、Activation Checkpointing 的核心思想

教授画了两条路径。

普通反向传播

Block 1

保存激活

Block 2

保存激活

Block 3

保存激活

Backward

保存更多中间结果，减少反向阶段的重复计算。

Activation Checkpointing

Block 1

保存检查点

Block 2

少保存中间激活

Block 3

反向时重新计算

Backward

减少保存的激活，以反向阶段更多计算为代价。

林默突然笑了。

“这不就是计算机系统里经典的时间换空间？”

教授点头。

“完全正确。”

普通训练：

```
保存更多结果
    ↓
更多显存
    ↓
减少重复计算
```

Checkpointing：

```
保存更少结果
    ↓
节省显存
    ↓
反向时重新计算
```

## 十六、PyTorch 如何使用 Checkpoint？

教授写：

```
from torch.utils.checkpoint import checkpoint
```

假设我们有多个 Transformer Block：

```
for block in self.blocks:    x = checkpoint(        block,        x,        use_reentrant=False    )
```

林默问：

“所以不是把数据写入磁盘？”

“不是。”

“这里的 Checkpointing 不是我们平时说的保存模型训练进度。”

教授特别区分：

```
Activation Checkpointing
→ 减少反向传播需要保存的中间激活


Model Checkpoint
→ 保存模型参数、优化器状态等
→ 用于训练恢复或部署
```

名字接近，但目的不同。

## 十七、为什么不是所有地方都开启 Checkpointing？

林默说：

“既然能省显存，那我全部开启不就行了？”

教授问：

“重新计算要不要时间？”

“要。”

“那不就行了。”

Activation Checkpointing 的代价包括：

```
反向传播额外计算

训练时间可能增加

随机操作需要注意一致性

代码与调试复杂度可能提高
```

教授补充：

“实际节省多少显存，取决于 Checkpoint 的切分粒度和内部算子。它也不会抹去全部激活内存。”

“有些计算昂贵、激活占用却不大的区域，过度 Checkpointing 可能得不偿失。”

所以它和梯度累积一样：

> 不是免费优化，而是一种资源交换。

# 第五幕：LoRA——不再训练整个帝国

## 十八、真正的转折：为什么非要更新全部参数？

清晨五点。

林默已经掌握四种武器：

```
AMP
  ↓
降低部分运算的精度与成本

Gradient Accumulation
  ↓
降低单次处理的 Batch 大小

Activation Checkpointing
  ↓
减少中间激活保存

Optimizer
  ↓
更新模型参数
```

然而，当他尝试对 7B 模型进行全参数微调时，仍然遇到了 OOM。

教授问了一个此前从未有人提出的问题。

“你为什么要训练全部 70 亿参数？”

林默愣住。

“微调模型……不就是修改模型参数吗？”

“是。”

“但谁告诉你必须修改所有参数？”

教授在黑板上写：

```
一个已经预训练完成的模型：

70 亿参数
```

接着又写：

```
你的任务：

让它更擅长回答 Java 技术问题
```

“为了这个目标，你真的需要更新所有 70 亿个参数吗？”

林默沉默了。

教授写下：

# LoRA

Low-Rank Adaptation——低秩适配。

## 十九、LoRA 的核心：原来的权重不动，只学习改变量

教授回到第二章的 `nn.Linear`。

一个线性层：

\\[ y=Wx \\]

假设：

\\[ W\in\mathbb{R}^{4096\times4096} \\]

这个矩阵有多少参数？

\\[ 4096^2=16,777,216 \\]

约 1678 万。

如果直接全参数微调，就需要对整个 W 计算并保存梯度，并为它维护优化器状态。

教授说：

“但我们可以冻结 W。”

```
W.requires_grad_(False)
```

“然后为它添加一个额外的可训练变化量。”

原来的计算：

\\[ y=Wx \\]

改成：

\\[ \boxed{y=(W+\Delta W)x} \\]

其中：

- \\(W\\)：冻结的预训练权重。
- \\(\Delta W\\)：需要学习的权重变化量。

林默问：

“可是 \\(\Delta W\\) 不也是一个 4096×4096 的矩阵吗？”

教授点头。

“所以我们还需要下一步。”

## 二十、低秩分解：让两个小矩阵代替一个大矩阵

教授写：

\\[ \Delta W=BA \\]

其中：

\\[ A\in\mathbb{R}^{r\times4096} \\]

\\[ B\in\mathbb{R}^{4096\times r} \\]

这里：

\\[ r\ll4096 \\]

通常选择比较小的 Rank。

例如：

\\[ r=16 \\]

那么：

```
原始权重矩阵 W

4096 × 4096

冻结，不训练
```

而：

```
LoRA A

16 × 4096
```

```
LoRA B

4096 × 16
```

参与训练。

整个计算：

\\[ \boxed{y=Wx+BAx} \\]

林默盯着公式。

“所以 W 不动。”

“但 BA 会不断学习如何修正它的输出？”

教授点头。

## 二十一、LoRA 到底节省多少参数？

我们亲自计算。

原始权重矩阵：

\\[ 4096\times4096=16,777,216 \\]

LoRA 参数：

\\[ 16\times4096+4096\times16 \\]

\\[ =131,072 \\]

相对于原始矩阵：

\\[ \frac{131072}{16777216}\times100\\% = 0.78125\\% \\]

一个 4096×4096 Linear 层的可训练参数对比

全参数微调

16,777,216

LoRA（Rank = 16）

131,072

可训练参数降低

# 99.22%

仅针对这个无偏置 Linear 权重矩阵进行的比较，不代表整个模型的 LoRA 参数比例或总显存节省比例。

林默猛地抬头。

“也就是说，原来要训练一千六百多万个参数。”

“现在只需要训练十三万个左右？”

“对。”

“那梯度和优化器状态也会大幅减少？”

“正是如此。”

教授继续：

“但注意，冻结的原始模型仍然要保存在内存里，而且反向传播需要经过相应计算路径。”

“所以 LoRA 能大幅降低可训练参数及相关优化器状态的开销，但不能简单说总显存降低 99%。”

## 二十二、LoRA 为什么不会破坏原始模型？

教授写：

\\[ W'=W+BA \\]

“原始 W 是冻结的。”

“训练期间只改变 A 和 B。”

因此：

```
预训练模型 W
        │
        ├── 保持冻结
        │
        ▼
      基础能力
        │
        +
        │
LoRA 适配器 BA
        │
        ▼
   针对目标任务调整
```

林默突然想到自己做过的 Agent。

“如果我想让同一个基础模型分别擅长 Java 问答、Python 编程和 SQL 生成……”

“是不是可以训练三个不同的 LoRA 适配器？”

教授回答：

“可以。”

```
                Base Model
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       LoRA A   LoRA B   LoRA C
          │        │        │
          ▼        ▼        ▼
        Java    Python     SQL
```

不过适配器之间是否能够良好切换、是否会影响原始能力，以及能否达到预期任务效果，都需要评测。

这就像：

> 同一套基础引擎，可以加载不同的任务适配组件。

# 第六幕：亲手实现 LoRA

## 二十三、其实不需要神秘的框架

林默问：

“教授，LoRA 看起来像很复杂的论文技术。”

“但 PyTorch 能自己写吗？”

教授点头。

“如果你真正理解 `nn.Linear`、`nn.Parameter` 和反向传播，LoRA 的核心其实不复杂。”

他开始编写。

```
import mathimport torchimport torch.nn as nnimport torch.nn.functional as Fclass LoRALinear(nn.Module):    def __init__(        self,        original_linear: nn.Linear,        rank=16,        alpha=32    ):        super().__init__()        self.original = original_linear        # 冻结原始 Linear 参数        for param in self.original.parameters():            param.requires_grad_(False)        in_features = original_linear.in_features        out_features = original_linear.out_features        self.rank = rank        self.scaling = alpha / rank        # A: [rank, in_features]        self.lora_A = nn.Parameter(            torch.empty(rank, in_features)        )        # B: [out_features, rank]        self.lora_B = nn.Parameter(            torch.zeros(out_features, rank)        )        nn.init.kaiming_uniform_(            self.lora_A,            a=math.sqrt(5)        )    def forward(self, x):        # 冻结的原始分支        base_output = self.original(x)        # LoRA 分支        lora_output = F.linear(            F.linear(x, self.lora_A),            self.lora_B        )        return (            base_output            + self.scaling * lora_output        )
```

林默看着：

```
self.lora_Aself.lora_B
```

“这两者都是 Parameter。”

“所以它们会自动出现在 `model.parameters()` 中。”

教授点头。

“而且因为它们参与了计算图，反向传播时可以得到梯度。”

## 二十四、为什么 A 随机初始化，B 初始化为零？

林默注意到：

```
nn.init.kaiming_uniform_(self.lora_A)
```

而：

```
self.lora_B = nn.Parameter(    torch.zeros(out_features, rank))
```

“为什么不两个都随机？”

教授写：

\\[ \Delta W=BA \\]

如果：

\\[ B=0 \\]

那么：

\\[ BA=0 \\]

所以训练刚开始：

\\[ y=Wx \\]

也就是：

> 刚插入 LoRA 时，模型初始输出与原始 Linear 一致。

林默明白了。

“这样不会一开始就引入随机修正，破坏预训练模型输出。”

“对。”

“而且 A 随机、B 为零的常见初始化方式允许 B 在第一步获得梯度，随后 A 也逐渐参与学习。”

教授继续：

“如果 A 和 B 都严格初始化成零，普通梯度下降在初始点就可能无法更新它们。”

## 二十五、测试 LoRA 是否真的只更新小矩阵

教授写：

```
base = nn.Linear(128, 128)lora_layer = LoRALinear(    base,    rank=8,    alpha=16)x = torch.randn(4, 128)target = torch.randn(4, 128)optimizer = torch.optim.AdamW(    [p for p in lora_layer.parameters()     if p.requires_grad],    lr=1e-3)optimizer.zero_grad()output = lora_layer(x)loss = F.mse_loss(output, target)loss.backward()optimizer.step()
```

然后：

```
for name, param in lora_layer.named_parameters():    print(        name,        param.requires_grad    )
```

预期可以看到：

```
lora_A True

lora_B True

original.weight False

original.bias False
```

林默终于理解：

“原来 LoRA 并不是一个完全不同的训练体系。”

“它只是改变了哪些参数参与训练，以及模型如何计算输出。”

教授点头。

训练循环仍然是：

```
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
```

只是 Optimizer 不再负责更新全部预训练权重。

# 第七幕：QLoRA——给冻结的巨人穿上更轻的盔甲

## 二十六、LoRA 还有一个问题

林默重新计算：

“教授，LoRA 的确减少了梯度和优化器状态。”

“但是假设模型有 7B 参数。”

“即使冻结，它的原始参数也需要保存吧？”

教授回答：

“没错。”

“如果以 BF16 存储基础权重，约需要 14GB 的纯权重空间。”

“如果再加上激活、临时工作区、LoRA 状态等，一张 24GB GPU 是否够用，要看具体配置。”

林默问：

“那能不能让冻结权重存得更小？”

教授在黑板上写：

# QLoRA

Quantized Low-Rank Adaptation。

核心思路：

> 将冻结的基础模型权重量化存储，同时使用 LoRA 训练低秩适配器。

## 二十七、4-bit 到底能节省多少？

教授重新写：

```
7B 参数
```

如果 FP32：

\\[ 7B\times4\text{ Bytes}=28\text{ GB} \\]

如果 BF16：

\\[ 7B\times2\text{ Bytes}=14\text{ GB} \\]

如果理想情况下每个权重使用 4 bit：

\\[ 7B\times0.5\text{ Bytes}=3.5\text{ GB} \\]

7B 基础模型的理论纯权重存储量

0 GB8 GB16 GB24 GB32 GBFP32BF16 / FP16INT8（理论）4-bit（理论）

仅计算模型原始权重的理想存储量，不包括量化元数据、激活、缓存及训练开销。

林默激动起来。

“3.5GB！”

“那一张 8GB 显卡就能随便微调 7B 模型？”

教授立即摇头。

“不能这么推。”

“3.5GB 只是 7B 个权重每个占 4 bit 的理想计算。”

真实 QLoRA 还需要：

```
量化比例因子等元数据
LoRA 参数
LoRA 梯度
优化器状态
中间激活
计算临时缓冲区
CUDA 工作区
```

更重要的是：

> 显存是否够用，与序列长度、Batch Size、LoRA Rank、量化实现和模型结构直接相关。

## 二十八、为什么不能直接用普通整数计算？

教授写：

```
原始浮点权重：

0.1234
-0.7821
1.2345
...
```

如果粗暴地转成低位整数：

```
0
-1
1
...
```

显然会产生量化误差。

因此实际量化通常要考虑：

```
量化区间

缩放比例

分组方式

权重分布

反量化计算
```

QLoRA 的经典方案使用了 NF4 等技术。

NF4：

NormalFloat 4-bit。

它针对近似正态分布的权重设计了一套非均匀量化表示。

在常见 QLoRA 流程中：

```
冻结的基础权重
       │
       ▼
以 4-bit 形式存储
       │
       ▼
计算时按算子实现反量化到计算精度
       │
       ▼
进行矩阵运算
       │
       +
       │
可训练 LoRA 分支
       │
       ▼
合并输出
```

教授强调：

“QLoRA 并不是直接让普通 4-bit 整数权重像 FP32 一样自由参与梯度更新。”

“经典 QLoRA 训练过程中，基础量化权重保持冻结，主要训练 LoRA 适配器。”

## 二十九、LoRA 与 QLoRA 的本质区别

| 对比     | LoRA           | QLoRA        |
| ------ | -------------- | ------------ |
| 基础模型   | 冻结             | 冻结           |
| 基础权重存储 | 通常 FP16/BF16 等 | 典型方案为 4-bit  |
| 可训练参数  | LoRA 适配器       | LoRA 适配器     |
| 显存占用   | 较全参数微调更低       | 通常可进一步降低     |
| 计算开销   | 取决于实现          | 可能增加反量化等成本   |
| 主要用途   | 参数高效微调         | 在更紧张的显存预算下微调 |

林默看着表格。

“所以 QLoRA 不是一种完全新的 Transformer。”

“它是量化与 LoRA 结合的训练方案。”

“正确。”

# 第八幕：FlashAttention——不改变答案，改变计算方式

## 三十、另一场显存灾难：Attention Matrix

林默以为问题已经解决。

他将模型换成 QLoRA。

显存占用降低了不少。

但当他把：

```
Sequence Length = 2048
```

改成：

```
Sequence Length = 16384
```

训练又开始出现显存压力。

教授问：

“还记得第九章 Self-Attention 的复杂度吗？”

林默回答：

\\[ O(n^2) \\]

“没错。”

教授写：

\\[ Attention(Q,K,V)= softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\]

“如果按最直接的方式实现，需要产生一个很大的注意力分数矩阵。”

假设：

```
seq_len = 16384
```

那么单个 Attention Head 的分数矩阵：

\\[ 16384\times16384 \\]

约有：

\\[ 268,435,456 \\]

个元素。

如果每个元素使用 2 字节保存：

\\[ 268,435,456\times2 = 536,870,912\text{ Bytes} \\]

也就是单个 Head 的一份这种矩阵就约占：

\\[ 512\text{ MiB} \\]

教授提醒：

“这只是针对一种显式矩阵表示的估算，并不是所有 Attention 实现都会把完整矩阵存下来。”

“但它已经足以说明，为什么长上下文会给朴素 Attention 带来巨大压力。”

## 三十一、FlashAttention 的核心不是换公式

林默问：

“那 FlashAttention 是不是把 Attention 改成 O(n) 了？”

教授摇头。

“这是一个非常重要的误区。”

“FlashAttention 的关键优势，并不是把标准精确 Attention 的主要算术复杂度从二次方变成线性。”

“它主要通过分块计算和 IO-aware 算法，减少高带宽显存与片上快速存储之间的数据搬运，并避免显式存储完整的 Attention 矩阵。”

教授画：

朴素实现：

```
Q × Kᵀ
   ↓
生成完整 Scores
   ↓
写入显存
   ↓
读取 Scores
   ↓
Softmax
   ↓
写入显存
   ↓
读取概率矩阵
   ↓
乘 V
   ↓
Output
```

FlashAttention：

```
Q / K / V
    ↓
按 Tile 分块
    ↓
尽可能在片上快速存储中完成局部计算
    ↓
在线更新 Softmax 统计量与输出累加
    ↓
输出最终结果
```

林默突然反应过来。

“也就是说，真正昂贵的不只是乘法。”

“还包括把巨大的中间矩阵反复搬进搬出显存。”

教授点头。

## 三十二、为什么它能够不保存完整 Softmax 矩阵？

林默追问：

“可是 Softmax 不是需要一整行的所有分数才能计算分母吗？”

教授写：

\\[ softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}} \\]

“传统思维会先算出整行，再算总和。”

“但可以使用在线 Softmax 算法，在处理分块时维护运行中的最大值与归一化统计量。”

“这样就能分块累积最终结果，而不必为每个 Attention Head 显式保存完整的 \\(N\times N\\) 概率矩阵。”

教授强调：

> FlashAttention 计算的是标准 Attention 的数学目标，允许正常浮点舍入误差，但通过改变计算顺序和数据搬运方式提高效率。

## 三十三、PyTorch 里怎么用 FlashAttention？

教授打开编辑器。

```
import torchimport torch.nn.functional as F# Q, K, V:# [batch, heads, seq_len, head_dim]output = F.scaled_dot_product_attention(    Q,    K,    V,    is_causal=True,    dropout_p=0.0)
```

林默问：

“就这一行？”

教授回答：

“PyTorch 的 `scaled_dot_product_attention` 可以根据设备、数据类型、形状和后端条件选择不同的 Attention 实现，包括受支持环境中的高效融合内核。”

“但不能认为只要调用这个 API，就一定使用了某个特定版本的 FlashAttention。”

“实际需要确认后端支持情况和性能分析结果。”

教授特别补充：

“这里如果使用 `dropout_p`，要显式按训练/推理状态设置。这个函数不会仅因为外层模型进入 `eval()` 就自动将传入的 Dropout 概率改为零。”

例如：

```
dropout_p = 0.1 if self.training else 0.0output = F.scaled_dot_product_attention(    Q,    K,    V,    is_causal=True,    dropout_p=dropout_p)
```

林默在笔记里写下：

```
FlashAttention
主要优化：
Attention 中间存储与内存访问


Gradient Checkpointing
主要优化：
反向传播保存的激活


LoRA
主要优化：
可训练参数和相关状态


QLoRA
进一步优化：
冻结基础权重的存储
```

# 第九幕：只有一张 24GB 显卡，究竟怎么训练大模型？

## 三十四、林默终于开始制定真正的训练方案

上午八点。

教授把所有方法写在黑板上。

```
硬件资源：

1 × 24GB GPU


目标：

让一个 7B 级别模型
更擅长 Java 技术问答
```

林默说：

“那我应该同时开启 AMP、梯度累积、Checkpointing、QLoRA 和 FlashAttention？”

教授摇头。

“可以组合，但不能机械地全部打开。”

“你应该先分析瓶颈。”

他在黑板上写：

```
第一步：明确目标
        ↓
是从零预训练？
还是在已有模型上微调？
        ↓
第二步：计算显存预算
        ↓
第三步：选择训练方案
        ↓
第四步：小规模运行
        ↓
第五步：Profiler 测量
        ↓
第六步：调整配置
```

对于只有一张 24GB 显卡的情况，优先考虑在已有开源模型上进行 QLoRA/SFT，而不是从零预训练 7B 模型。

原因很简单：

从零预训练不仅存在显存问题，还需要巨量高质量数据、计算资源和训练工程能力。即使利用各种节省显存技术，一张消费级显卡也不意味着能够以合理成本完成 7B 级模型的充分预训练。

## 三十五、推荐的初始微调配置

教授给出一个实验起点。

| 配置项              | 初始方案                        |
| ---------------- | --------------------------- |
| 基础模型             | 有适当许可证的 7B 级模型              |
| 训练方式             | QLoRA                       |
| 权重量化             | 4-bit NF4                   |
| 计算精度             | 支持时优先 BF16                  |
| LoRA Rank        | 16                          |
| LoRA Alpha       | 32                          |
| Micro-Batch Size | 1                           |
| 梯度累积             | 16                          |
| 初始序列长度           | 1024                        |
| 学习率              | \\(2\times10^{-4}\\) 作为实验起点 |
| 优化器              | AdamW 或适用的低内存实现             |
| 激活检查点            | 可启用                         |
| Attention        | 支持时使用高效 SDPA 后端             |

教授说：

“这不是保证一定能跑通的配置。”

“实际还取决于具体模型架构、量化后端、GPU、优化器实现和显存碎片等。”

“但它是一个合理的实验起点。”

## 三十六、实际工程中，怎么接入 QLoRA？

教授打开一个新的 Python 文件。

“今天我们不再手工为 7B 模型编写每一层。”

“PyTorch 负责底层张量与自动求导，实际微调通常会配合 Hugging Face Transformers、PEFT 和 bitsandbytes 等工具。”

下面是一个简化的模型加载与 LoRA 配置示例。它需要兼容的 CUDA 环境、已安装的相关依赖，以及用户选定的可访问模型。

```
import torchfrom transformers import (    AutoModelForCausalLM,    BitsAndBytesConfig,)from peft import (    LoraConfig,    get_peft_model,    prepare_model_for_kbit_training,)# 替换成实际有权限使用的模型 IDmodel_id = "your-compatible-7b-model"# 1. 4-bit 量化配置quant_config = BitsAndBytesConfig(    load_in_4bit=True,    bnb_4bit_quant_type="nf4",    bnb_4bit_compute_dtype=torch.bfloat16,    bnb_4bit_use_double_quant=True,)# 2. 加载冻结的量化基础模型model = AutoModelForCausalLM.from_pretrained(    model_id,    quantization_config=quant_config,    device_map="auto",)# 3. 为量化模型微调做准备model = prepare_model_for_kbit_training(model)# 4. 配置 LoRAlora_config = LoraConfig(    r=16,    lora_alpha=32,    target_modules="all-linear",    lora_dropout=0.05,    bias="none",    task_type="CAUSAL_LM",)# 5. 注入 LoRAmodel = get_peft_model(    model,    lora_config,)# 6. 查看训练参数量model.print_trainable_parameters()
```

林默看到：

```
target_modules="all-linear"
```

“这是什么意思？”

教授回答：

“PEFT 会尝试给模型中支持的线性模块注入 LoRA 适配器。”

“但不同模型架构和软件版本可能存在兼容性差异，实际训练前需要检查目标模块。”

然后执行：

```
model.print_trainable_parameters()
```

输出会显示：

```
trainable params: ...
all params: ...
trainable%: ...
```

林默点头。

“这样我就能知道究竟有多少参数真正参与训练。”

“对。”

“后面再准备 Tokenizer、训练与验证数据、标签 Mask、优化器和训练循环，才能完成真正的 SFT。”

# 第十幕：训练为什么仍然可能失败？

## 三十七、显存省下来了，不等于模型一定能学好

林默终于成功启动了训练。

但几个小时后，他发现验证集 Loss 没有明显改善。

他疑惑地问：

“教授，显存问题已经解决了，为什么效果还是不理想？”

教授回答：

“因为显存优化解决的是资源问题，不是模型质量问题。”

即使显存足够，训练仍可能失败。

例如：

训练数据有问题。

```
重复数据过多

标签错误

答案质量低

训练目标不合理
```

学习率有问题。

```
太大 → 训练不稳定

太小 → 学习缓慢
```

LoRA 容量不合适。

```
Rank 太小
→ 适配能力可能不足

Rank 太大
→ 更多显存与训练成本
```

训练评测有问题。

```
只看训练 Loss

没有验证集

没有任务级评测

模型可能只是记住训练数据
```

教授说：

> 能够把模型训练起来，只是工程成功的第一步。训练后的模型是否真正变好，需要独立验证。

林默突然想到自己开发 Coding Agent 时使用过的 SWE-bench 评测。

“也就是说，大模型微调同样不能只看 Loss。”

“还要看真实任务表现。”

教授点头。

例如微调一个 Java 问答模型，可以准备独立测试集，评估：

```
技术事实正确率

推理与解释质量

代码能否编译运行

对未见过问题的泛化能力

幻觉情况

相对基础模型的提升或退化
```

这才是判断微调是否成功的依据。

# 三十八、所有显存优化技术，终于连接起来了

教授在黑板上画出最后一张作战地图。

```
                   训练大型模型
                        │
                        ▼
                  显存不足 OOM
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
        模型权重       中间激活      优化器状态
           │            │            │
           ▼            ▼            ▼
         量化         减少 Batch     LoRA
           │            │            │
           ▼            ▼            ▼
         QLoRA        梯度累积      冻结基础参数
           │            │
           │            ▼
           │       Checkpointing
           │
           └────────────┬────────────┘
                        │
                        ▼
                    计算优化
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
             AMP             FlashAttention
              │                   │
              ▼                   ▼
          混合精度计算         IO-aware Attention
                        │
                        ▼
                  重新测量资源
                        │
                        ▼
                  训练与验证模型
```

他最后补上一句话：

> 训练优化不是寻找某一个神奇的 API，而是在计算速度、显存容量、数值稳定性和最终效果之间做权衡。

## 三十九、面试与实际开发中最容易混淆的五个问题

训练工程理解测试

0/5 已完成

1\. LoRA 将某层可训练参数减少 99%，是否意味着整张 GPU 的显存占用也降低 99%？

是

否，基础权重和激活等依然占显存

2\. 梯度累积最主要解决什么问题？

减少单次 Micro-Batch 的激活内存需求

将全部模型权重转换成 INT4

完全消除梯度噪声

3\. Activation Checkpointing 的主要代价是什么？

必须降低模型层数

反向传播时进行额外重计算

无法执行反向传播

4\. QLoRA 经典方案主要训练哪部分？

全部量化权重

冻结基础权重上的 LoRA 适配器

Tokenizer 的词表

5\. FlashAttention 的核心优化是什么？

将标准 Attention 数学目标替换成线性近似

主要减少 Attention 的中间存储和数据搬运

消除 Transformer 对 GPU 的需求

# 第十三章预告：《模型的第二次觉醒——SFT、Instruction Tuning、DPO 与 RLHF》

深夜。

林默终于成功在单张 GPU 上完成了一次小规模 QLoRA 微调。

他向微调后的模型输入：

> 请解释 Java HashMap 为什么线程不安全，并给出代码示例。

模型生成了一段回答。

表面看起来流畅。

但林默发现，里面有一个并不存在的 Java API。

“教授，模型确实学会了回答的格式。”

“但是它还是会胡说八道。”

教授看着屏幕。

“因为你已经学会如何训练模型，却还没有学会如何系统性地塑造模型的行为。”

他在黑板上写：

```
Pre-training
     ↓
学习广泛的语言与知识模式
     ↓
SFT
     ↓
学习遵循指令和回答格式
     ↓
Preference Optimization
     ↓
优化回答偏好
     ↓
DPO / RLHF 等
     ↓
进一步塑造模型行为
     ↓
Evaluation
     ↓
判断模型是否真正改进
```

林默盯住其中两个缩写。

```
DPO

RLHF
```

“这两个是什么？”

教授笑了。

“如果说预训练教会模型如何续写语言，SFT 教会模型如何按照指令回答，那么偏好优化要进一步回答：”

> 面对两个都说得通的回答，模型应该倾向选择哪一个？

下一章，我们将继续从 PyTorch 出发，深入拆解：

1. SFT 如何构造 `input_ids`、`labels` 和 `attention_mask`。
2. 为什么指令微调经常把 Prompt 对应的 Label 设置为 `-100`。
3. 全参数 SFT、LoRA SFT 和 QLoRA SFT 的区别。
4. 为什么 SFT Loss 下降，不一定代表模型回答质量提升。
5. DPO 为什么可以利用 `chosen/rejected` 偏好数据训练模型。
6. RLHF 的 Reward Model、PPO 与强化学习训练究竟怎么运作。
7. 为什么推理模型还需要专门的评测、奖励设计和训练流程。

下一章：《模型的第二次觉醒——从会预测下一个 Token，到能够遵循指令、比较回答优劣》。



下一章。



# 《PyTorch：炼金术士的学徒》

## 第十三章：模型的第二次觉醒——SFT、DPO、RLHF 与强化学习

从语言模型预训练，走向指令微调、偏好优化与推理能力训练。

第十三天，凌晨一点。

林默终于完成了人生中第一次大语言模型微调。

一张 24GB 的显卡，使用 QLoRA，训练了数千步。

他兴奋地打开测试程序，输入：

> 请解释 Java HashMap 为什么线程不安全，并给出一个可运行的示例。

模型开始输出：

“HashMap 通过内部 synchronized 机制保证线程安全……”

林默的笑容消失了。

“等等。”

“HashMap 本身不保证线程安全，哪来的内部 synchronized 机制？”

他继续测试：

> 请解释 Java ThreadLocal 为什么可能发生内存泄漏。

模型回答得很流畅，但混淆了弱引用 Key 与强引用 Value。

再测试一道算法题。

代码能编译，却没考虑边界条件。

林默望着下降的训练 Loss。

```
Step 100    Loss 2.31
Step 500    Loss 1.42
Step 1000   Loss 0.83
Step 2000   Loss 0.41
```

“教授，Loss 明明一直在下降。”

“为什么模型还是会回答错误？”

陈教授走到白板前，写下：

\\[ \boxed{\text{训练目标下降}\ne\text{真实能力一定提升}} \\]

“你让模型学习的，究竟是什么？”

“预测下一个 Token。”

“那你有没有直接教它什么叫事实正确、什么叫按照指令回答、什么叫承认不确定性？”

林默沉默了。

教授继续：

“今天，我们要研究大语言模型的第二阶段。”

> 预训练让模型学会语言中的广泛规律，而后训练试图把这些能力塑造成更符合目标任务和使用要求的行为。

# 第一幕：为什么预训练模型不等于助手模型？

## 一、一个只会续写的学者

教授在终端中写入：

```
问题：Java HashMap 为什么线程不安全？
```

一个只经过基础预训练的模型，可能继续生成：

```
问题：ConcurrentHashMap 为什么线程安全？

问题：Hashtable 和 HashMap 有什么区别？
```

林默皱眉。

“它怎么不回答？”

教授说：

“因为对一个只学过普通文本续写目标的模型而言，继续生成另一道问题也可能是合理的文本延续。”

预训练目标通常可以写成：

\\[ L\_{\text{pretrain}} = -\sum_t\log P\_\theta(x_t\mid x\_{\<t}) \\]

它直接优化的是：

> 给定前面的 Token，提高训练数据中后续 Token 的预测概率。

它并没有天然规定：

```
遇到用户提问 → 必须认真回答

遇到编程任务 → 应该给正确代码

不知道答案 → 应该表达不确定性

遇到歧义 → 必要时询问
```

这些行为可以通过后训练被进一步塑造。

教授画出大模型训练的一条典型路线：

```
                  海量预训练语料
                         │
                         ▼
                    Pre-training
                         │
                         ▼
                     Base Model
                         │
                         ▼
                 Instruction SFT
                         │
                         ▼
                   Instruct Model
                         │
                  ┌──────┴──────┐
                  ▼             ▼
                 DPO        RLHF / RL
                  │             │
                  └──────┬──────┘
                         ▼
                  Post-trained Model
                         │
                         ▼
                     Evaluation
```

“注意，”教授说，“这是一种概念化路线，不代表所有现代模型都严格按照这个固定顺序训练。不同方法还可以迭代或组合。”

# 第二幕：SFT——给模型一本优秀答案集

## 二、SFT 到底是什么？

教授写：

# Supervised Fine-Tuning

监督微调，简称 SFT。

“想象一下，你招聘了一个读过数百万本书的学者。”

“他知识面很广，但不知道公司希望他怎么工作。”

“于是你给他一本工作手册。”

```
用户提出 Java 问题
       ↓
应该先明确概念
       ↓
再解释原理
       ↓
提供准确的示例
       ↓
必要时说明适用条件
```

这本手册就是高质量指令数据集的一种类比。

例如：

```
{
  "messages": [
    {
      "role": "user",
      "content": "什么是 Java ThreadLocal？"
    },
    {
      "role": "assistant",
      "content": "ThreadLocal 提供线程局部变量访问机制，每个线程可以通过同一个 ThreadLocal 对象访问自己的值..."
    }
  ]
}
```

林默点头。

“也就是把用户问题和优秀回答配成训练样本。”

“对。”

但 SFT 的关键不仅在于有数据。

还在于：

> 哪些 Token 应该参与 Loss？

## 三、SFT 和预训练，Loss 是不是完全不同？

林默问：

“之前预训练使用 CrossEntropyLoss。”

“SFT 是不是要换一个特殊的损失函数？”

教授摇头。

“对自回归语言模型，常见 SFT 仍然可以使用下一 Token 预测的交叉熵。”

公式：

\\[ L\_{\text{SFT}} = -\sum\_{t\in\mathcal A} \log P\_\theta(y_t\mid x,y\_{\<t}) \\]

其中：

- \\(x\\)：用户输入及必要的对话上下文。
- \\(y\\)：目标助手回答。
- \\(\mathcal A\\)：参与损失计算的目标回答位置。

“最大的区别之一，在于训练样本如何组织，以及哪些位置参与训练。”

# 三、最容易写错的地方：Prompt Mask

## 四、为什么用户问题通常不应该被当作回答来训练？

教授写出：

```
<user>
解释 Java HashMap 的扩容机制。
</user>

<assistant>
HashMap 在容量达到阈值时通常会触发扩容……
```

假设 Tokenizer 已经完成编码。

```
Input IDs：

[用户问题 Token ...][助手回答 Token ...]
```

林默问：

“那 labels 不就是把所有 input_ids 向后错位一格吗？”

教授回答：

“基础语言模型可以这么训练。”

“但如果我们的目标是让模型学会生成助手回答，那么一种常见 SFT 策略是：只对助手回答部分计算 Loss。”

他在黑板上写：

```
输入：

[Prompt Token][Answer Token]


Loss Mask：

[不计算 Loss][计算 Loss]
```

在 PyTorch 的 `CrossEntropyLoss` 中，默认有一个非常有用的设置：

```
ignore_index=-100
```

如果某个目标位置的 Label 是：

```
-100
```

就表示：

> 这个位置不参与 CrossEntropyLoss。

## 五、用一个简单例子理解 `-100`

教授先不用真实 Tokenizer，而是使用数字演示。

假设：

```
用户问题：[11, 12, 13]

助手回答：[21, 22, 23]
```

训练序列：

```
input_ids = torch.tensor([    11, 12, 13, 21, 22, 23])
```

如果采用完整序列的下一 Token 预测标签：

```
[12, 13, 21, 22, 23, ...]
```

但我们希望仅让模型针对助手回答产生的目标 Token 计算损失。

可以概念化成：

```
上下文位置：

11    12    13    21    22

预测目标：

忽略  忽略  21    22    23
```

于是：

```
Labels：

[-100, -100, 21, 22, 23]
```

林默突然发现：

“第三个位置的输入是用户问题的最后一个 Token。”

“它负责预测助手回答的第一个 Token！”

教授点头。

“这就是需要认真处理 Token 对齐的原因。”

“不同训练框架可能约定 labels 为未移位序列，内部完成 shift，也可能要求你提前移位。不能把两套约定混用。”

## 六、写一个简化的 PyTorch SFT Loss

假设模型输出：

```
logits.shape = [B, T, V]
```

我们使用未移位的标签 Tensor，并在计算 Loss 时手动完成移位。

```
import torch.nn.functional as F# logits: [B, T, V]# labels: [B, T]# labels 中 -100 代表忽略shift_logits = logits[:, :-1, :].contiguous()shift_labels = labels[:, 1:].contiguous()loss = F.cross_entropy(    shift_logits.reshape(        -1,        shift_logits.size(-1)    ),    shift_labels.reshape(-1),    ignore_index=-100)
```

林默看着：

```
logits[:, :-1, :]
```

和：

```
labels[:, 1:]
```

立刻想起第十一章。

“还是错位预测！”

“对。”

“只是我们把用户 Prompt 对应的目标位置 Mask 掉，让 Loss 重点监督助手回答。”

教授补充：

“对于多轮对话，还需要明确哪些 assistant 消息参与训练，以及角色标记、结束标记和工具消息如何处理。”

“高质量 SFT 数据管线，往往比单纯写一个训练循环更容易出错。”

# 四、为什么 Loss 下降，却未必学得更好？

## 七、模型可能只是在死记硬背

教授再次展示林默的训练曲线。

```
训练 Loss：

2.31 → 1.42 → 0.83 → 0.41
```

“你觉得这代表什么？”

林默说：

“模型越来越擅长预测训练集里的答案。”

“很好。”

“但如果这些答案都已经被记住，它是否一定能回答新问题？”

“不一定。”

教授给出例子。

训练集里只有：

```
Q：HashMap 为什么线程不安全？
A：因为它没有提供完整的并发访问安全保证……
```

但验证集里是：

```
Q：两个线程同时修改 HashMap，可能产生什么问题？
```

这时，模型是否能够解释：

```
数据竞争

丢失更新

可见性问题

并发结构修改风险
```

才更接近我们关心的泛化能力。

## 八、训练集、验证集和测试集应该怎么用？

教授写：

| 数据集        | 主要用途                |
| ---------- | ------------------- |
| Training   | 更新模型参数              |
| Validation | 选择超参数、检查过拟合、决定训练策略  |
| Test       | 在模型及评测方案确定后估计最终泛化表现 |

“如果你反复针对测试集调模型，它实际上也开始影响你的模型选择。”

“那么最终测试结果可能越来越乐观。”

林默点头。

这跟 Coding Agent 的 SWE-bench 评测非常类似。

> 训练目标只是优化手段，真实任务表现才是最终评价依据。

# 第五幕：两份答案都正确，应该选择谁？

## 九、SFT 遇到的新问题

教授写出同一个用户问题：

> Java HashMap 为什么线程不安全？

然后给出两个答案。

回答 A：

“HashMap 不是线程安全的。如果多个线程并发修改，可能出现数据竞争和丢失更新。需要根据场景选择同步控制或 ConcurrentHashMap。”

回答 B：

“HashMap 非线程安全。并发修改时可能出现更新丢失、读取到不一致状态等问题。例如，两个线程在缺乏同步的情况下同时进行结构修改，其行为不具备线程安全保证。通常可以使用 ConcurrentHashMap，或者通过外部同步保护共享 HashMap。需要注意，ConcurrentHashMap 也不等于任意复合操作天然原子化。”

教授问：

“哪个更好？”

林默回答：

“B 更详细，还说明了 ConcurrentHashMap 的适用边界。”

教授点头。

“但假设用户只需要一句话的面试答案呢？”

林默思考。

“那 A 可能更合适。”

“正是如此。”

教授写下：

> 回答质量不仅包含正确性，还可能包含完整性、简洁度、帮助程度、格式和上下文适配性。

而 SFT 只利用某条目标回答直接进行监督，并不天然显式比较不同回答的优劣。

于是一个新的训练目标出现了：

# Preference Optimization

偏好优化。

## 十、Chosen 与 Rejected

教授建立一份训练数据：

```
{
  "prompt": "解释 Java ThreadLocal 的工作原理。",
  "chosen": "ThreadLocal 通过线程内部维护的 ThreadLocalMap 实现每线程独立存储...",
  "rejected": "ThreadLocal 通过创建一个全局共享变量保证线程安全..."
}
```

林默看着：

```
chosen

rejected
```

“一个是更受偏好的回答。”

“另一个是较差的回答。”

“没错。”

教授说：

“但 chosen 不一定是绝对正确，rejected 也不一定完全错误。”

“它们表示在特定标准下的相对偏好。”

例如：

```
Prompt
   │
   ├── Answer A
   │      ↓
   │    Chosen
   │
   └── Answer B
          ↓
       Rejected
```

目标变成：

> 提高模型对偏好回答的相对倾向。

# 第六幕：DPO——没有单独奖励模型，也能学习偏好？

## 十一、Direct Preference Optimization

教授写：

# DPO

Direct Preference Optimization

林默问：

“它和 SFT 有什么区别？”

教授回答：

“SFT 让模型提高目标回答的条件概率。”

“DPO 则利用成对偏好数据，优化 chosen 与 rejected 之间的相对关系，并通过参考模型约束偏好变化。”

他画出：

```
                 Prompt
                   │
            ┌──────┴──────┐
            ▼             ▼
         Chosen        Rejected
            │             │
            ▼             ▼
      计算回答概率    计算回答概率
            │             │
            └──────┬──────┘
                   ▼
          与参考模型比较
                   │
                   ▼
                 DPO Loss
                   │
                   ▼
               backward()
                   │
                   ▼
                更新 Policy
```

林默问：

“参考模型又是什么？”

教授写下：

```
Policy Model
→ 当前正在训练的模型


Reference Model
→ 作为对比基准的模型
```

“Reference 通常是冻结的，它代表训练开始附近的行为基准。”

## 十二、为什么 DPO 需要 Reference Model？

教授给出一个例子。

对于某个 Prompt：

```
Chosen：
正确且清晰的回答

Rejected：
错误且冗长的回答
```

如果只要求：

```
P(Chosen) > P(Rejected)
```

看起来很合理。

但教授说：

“我们不希望模型为了满足一些有限的偏好样本，就任意偏离原来学到的大量语言能力。”

“所以 DPO 的典型目标里，会比较当前模型与参考模型对两份回答的相对概率。”

设：

\\[ \pi\_\theta \\]

表示当前 Policy。

\\[ \pi\_{\text{ref}} \\]

表示参考模型。

DPO 的经典损失可以写成：

\\[ \mathcal L\_{\text{DPO}} = -\log\sigma \left( \beta \left[ \log\frac{\pi\_\theta(y_w|x)} {\pi\_{\text{ref}}(y_w|x)} - \log\frac{\pi\_\theta(y_l|x)} {\pi\_{\text{ref}}(y_l|x)} \right] \right) \\]

其中：

- \\(y_w\\)：chosen。
- \\(y_l\\)：rejected。
- \\(\beta\\)：控制偏好优化中相对参考模型变化的尺度。
- \\(\sigma\\)：Sigmoid。

林默盯着公式。

“看着还是复杂。”

教授说：

“那我们把它拆成三个数字。”

## 十三、把 DPO 公式翻译成人话

定义：

\\[ r_w= \log\pi\_\theta(y_w|x) - \log\pi\_{\text{ref}}(y_w|x) \\]

\\[ r_l= \log\pi\_\theta(y_l|x) - \log\pi\_{\text{ref}}(y_l|x) \\]

于是：

\\[ L=-\log\sigma(\beta(r_w-r_l)) \\]

林默终于看懂了。

“所以核心是在比较：”

```
当前模型相对参考模型
对 Chosen 提高了多少倾向

减去

当前模型相对参考模型
对 Rejected 提高了多少倾向
```

“对。”

如果这个差值变大，DPO Loss 通常会下降。

教授强调：

“这里比较的是完整回答的条件对数概率，而不是简单地只比较某一个 Token 的 Softmax 值。”

## 十四、用 PyTorch 手写 DPO 核心 Loss

教授打开编辑器：

```
import torchimport torch.nn.functional as Fdef dpo_loss(    policy_chosen_logp,    policy_rejected_logp,    ref_chosen_logp,    ref_rejected_logp,    beta=0.1):    # 当前模型的偏好差    policy_logratio = (        policy_chosen_logp        - policy_rejected_logp    )    # 参考模型的偏好差    ref_logratio = (        ref_chosen_logp        - ref_rejected_logp    )    # 相对于参考模型的偏好改善    logits = (        policy_logratio        - ref_logratio    )    return (        -F.logsigmoid(beta * logits)    ).mean()
```

林默一眼认出了：

```
F.logsigmoid()
```

“就是之前学过的 Sigmoid 的对数形式。”

“对。”

“PyTorch 把很多复杂公式拆开之后，其实依然是普通 Tensor 运算。”

教授提醒：

“这只是经典 DPO 核心损失的示例。完整训练还需要正确计算每个回答的 Token 对数概率、Mask 掉 Prompt 和 Padding、处理序列长度、参考模型前向，以及注意数值稳定性。”

# 第七幕：RLHF——让模型在反馈中调整行为

## 十五、如果没有 Chosen/Rejected 成对答案呢？

林默问：

“DPO 需要成对偏好数据。”

“那如果我们希望模型探索新的回答，并根据回答质量进一步优化？”

教授点头。

“这就进入强化学习的世界。”

他在白板上写：

# RLHF

Reinforcement Learning from Human Feedback

从人类反馈中进行强化学习。

经典的一类 RLHF 流程：

```
                  预训练模型
                       │
                       ▼
                     SFT
                       │
                       ▼
                  SFT Model
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
         收集偏好数据         初始化 Policy
             │                    │
             ▼                    │
        训练 Reward Model        │
             │                    │
             └─────────┬──────────┘
                       ▼
                强化学习优化
                       │
                       ▼
                  更新 Policy
```

教授说：

“RLHF 不是某一个具体优化器的名称，而是一类利用人类反馈引导模型优化的训练范式。”

“PPO 是经典 RLHF 中使用过的重要强化学习算法，但并不是唯一选择。”

## 十六、Reward Model 是什么？

假设模型针对同一个问题生成两个答案：

```
Answer A：
HashMap 是线程安全的，因为……

Answer B：
HashMap 不保证线程安全，因为……
```

人类标注者更偏好 B。

我们可以训练一个模型：

```
输入：

Prompt + Answer

输出：

一个评分
```

例如：

```
Reward Model

Answer A → -1.2

Answer B →  2.4
```

林默问：

“所以 Reward Model 就像一个评分老师？”

“可以这么理解。”

“但评分值不是一个客观真理，它是根据训练反馈学习出来的估计。”

教授写：

\\[ R\_\phi(x,y) \\]

表示奖励模型对问题 \\(x\\) 和回答 \\(y\\) 给出的分数。

然后强化学习希望：

\\[ \max\_\theta \mathbb E\_{y\sim\pi\_\theta} [R\_\phi(x,y)] \\]

即：

> 让当前模型生成更容易获得高奖励的回答。

但如果只追逐奖励，也会出问题。

# 八、奖励黑客：为什么模型会钻评分规则的漏洞？

## 十七、一个危险的实验

教授给模型设计了一条简单奖励规则。

```
如果答案包含：

“详细解释”

奖励 +1
```

一开始，模型确实更喜欢给出详细解释。

但训练继续之后，模型开始生成：

```
详细解释，详细解释，详细解释……
```

林默愣住。

“它在刷奖励？”

教授点头。

“这就是 Reward Hacking 的一个简化例子。”

当奖励信号不能准确反映真实目标时，模型可能优化评分规则，而不是优化人类真正想要的结果。

在 Coding Agent 场景也可能出现类似问题。

例如：

```
奖励：
只要测试通过就给高分
```

如果评测环境存在缺陷，Agent 可能找到绕过测试的方式，而不是正确实现需求。

因此：

> 奖励函数设计和评测环境可靠性，是强化学习训练中极其重要的工程问题。

## 十八、为什么 RLHF 常常需要约束模型不要偏离太远？

教授写：

\\[ \max\_\theta \mathbb E \left[ R(x,y) - \beta D\_{KL} \left( \pi\_\theta\Vert\pi\_{\text{ref}} \right) \right] \\]

林默看向：

\\[ D\_{KL} \\]

“这是 KL 散度？”

“对。”

“它可以衡量两个概率分布之间的差异。”

教授解释：

```
Reward
    ↓
鼓励模型更符合偏好


KL Penalty
    ↓
抑制模型无约束地偏离参考策略
```

比如：

```
只提高奖励
    ↓
可能利用奖励漏洞
    ↓
生成异常内容
```

加入合适的约束，可以降低这种偏移风险。

但也要注意：

“KL 约束不是奖励黑客问题的万能解法，过强约束还可能限制模型改善。”

# 第九幕：PPO 到底在做什么？

## 十九、Policy Gradient：生成也是一种决策过程

教授说：

“你以前觉得强化学习只属于游戏和机器人。”

林默点头。

“比如智能体控制角色移动。”

“但文本生成也能被建模成顺序决策。”

```
当前状态：
已有 Prompt 和生成前缀
        │
        ▼
动作：
选择下一个 Token
        │
        ▼
新状态：
前缀追加 Token
        │
        ▼
继续选择下一个 Token
```

在这个框架下：

- State：当前上下文。
- Action：选择下一个 Token。
- Policy：语言模型的条件概率分布。
- Reward：根据生成结果获得的反馈。

林默恍然大悟。

“也就是说，语言模型就是 Policy？”

“在这种强化学习建模中，是的。”

## 二十、PPO 为什么要限制更新幅度？

PPO：

Proximal Policy Optimization

教授画了一条简单的优化路线。

```
旧 Policy
    │
    ▼
生成回答
    │
    ▼
获得 Reward
    │
    ▼
估计 Advantage
    │
    ▼
更新 Policy
```

“但如果一次更新幅度太大，模型生成策略可能突然发生剧烈变化。”

“PPO 的经典 Clipped Objective，正是为了限制策略更新中的某些过大变化。”

公式：

\\[ L^{CLIP} = \mathbb E \left[ \min \left( r_tA_t, clip(r_t,1-\epsilon,1+\epsilon)A_t \right) \right] \\]

其中：

\\[ r_t= \frac{\pi\_\theta(a_t|s_t)} {\pi\_{\text{old}}(a_t|s_t)} \\]

林默问：

“这个比例什么意思？”

教授回答：

“它比较某个动作在新旧策略下的概率。”

比如：

```
旧模型：
选择某 Token 的概率 = 0.2

新模型：
选择该 Token 的概率 = 0.8
```

那么：

\\[ r_t=\frac{0.8}{0.2}=4 \\]

“变化非常大。”

“PPO 的 Clip 机制会限制这种比例变化对目标函数带来的进一步优化激励。”

教授提醒：

“这不表示 PPO 在数学上保证每次策略变化绝不会超过阈值，它只是用裁剪目标抑制过大的更新诱因。”

# 十、DPO 和 RLHF 究竟怎么选？

## 二十一、林默开始考虑实际工程

“教授，如果我想微调一个 Java 问答模型。”

“到底选择 SFT、DPO 还是 PPO？”

教授没有立即回答。

而是问：

“你的数据是什么？”

“只有标准问答？”

“那优先考虑 SFT。”

“如果有成对偏好？”

“可以考虑 DPO 等偏好优化方法。”

“如果有可交互的训练环境、比较可靠的奖励和足够计算预算？”

“可以考虑强化学习训练。”

他写下一张对照表：

| 方法         | 主要数据/反馈               | 训练思路          | 典型挑战            |
| ---------- | --------------------- | ------------- | --------------- |
| SFT        | 高质量指令与目标回答            | 最大化目标回答概率     | 数据质量、泛化、过拟合     |
| DPO        | Chosen / Rejected 偏好对 | 直接优化相对偏好      | 偏好噪声、参考模型与数据分布  |
| RLHF + PPO | 偏好训练的奖励模型与采样反馈        | 在线采样、奖励驱动策略优化 | 复杂度高、奖励偏差、训练稳定性 |
| 可验证奖励 RL   | 测试、规则、数学答案等反馈         | 根据可验证结果优化     | 奖励设计、探索和评测漏洞    |

“而且，”教授继续，“实际训练管线不一定只选其中一种。”

例如：

```
高质量 SFT
      ↓
偏好优化
      ↓
针对推理/代码任务进行可验证奖励训练
      ↓
持续评测
```

但每增加一个阶段，都应该证明它确实带来收益。

# 第十一幕：为什么推理模型需要不一样的训练？

## 二十二、从语言答案到可验证结果

教授打开一道数学题。

> 求 37 × 48。

模型 A：

```
答案是 1776。
```

模型 B：

```
37 × 48
= 37 × (50 - 2)
= 1850 - 74
= 1776
```

教授问：

“哪个更好？”

林默说：

“都算对了，但 B 给出了过程。”

教授继续：

“假设训练目标只是最终答案正确。”

“那么验证很简单。”

```
reward = 1.0 if answer == 1776 else 0.0
```

“这就是可验证奖励最容易理解的形式之一。”

对于代码任务：

```
编写函数
      ↓
运行单元测试
      ↓
通过 / 失败
      ↓
生成奖励信号
```

林默突然想到 Coding Agent。

“所以软件工程任务天然存在很多可验证反馈。”

例如：

```
编译是否通过

测试是否通过

静态检查是否通过

行为是否符合契约
```

教授点头。

“这正是强化学习在代码和数学等领域具有吸引力的重要原因之一。”

但他立刻补充：

“测试通过不等于软件一定正确。测试覆盖、数据泄漏和评测环境安全依然重要。”

## 二十三、推理能力不只是回答更长

林默问：

“那训练推理模型，是不是让模型多写几段思考过程？”

教授摇头。

“不能这么简单理解。”

“更长的文字不等于更好的推理。”

一个模型完全可能写出很长但错误的分析。

真正关心的是：

```
任务是否完成

最终结果是否正确

推理是否可泛化

能否检查自身错误

计算成本是否合理
```

一些训练方法会鼓励模型在复杂问题上学习更有效的中间步骤、验证行为或策略。

但这种能力是否真正提升，必须通过独立任务评测判断，而不能只看回答长度。

# 第十二幕：把后训练和 Coding Agent 连接起来

## 二十四、Agent 能不能直接作为强化学习训练环境？

林默忽然问：

“教授，假设我已经有一个 Coding Agent。”

“它能够使用工具、编辑代码、运行测试。”

“那我是不是可以用它来训练模型？”

教授想了想。

“理论上，可以把软件任务环境设计成训练或评测系统。”

例如：

```
              Issue / Requirement
                       │
                       ▼
                  Coding Agent
                       │
              ┌────────┴────────┐
              ▼                 ▼
           思考规划           工具调用
              │                 │
              ▼                 ▼
           编辑代码           运行测试
              │                 │
              └────────┬────────┘
                       ▼
                    最终提交
                       │
                       ▼
                   验证环境
                       │
                       ▼
                   Reward / Metrics
```

“但这里有几个非常困难的问题。”

第一，如何定义可靠的成功标准。

第二，如何防止模型利用评测环境漏洞。

第三，如何控制一次完整 Agent 轨迹的成本。

第四，如何把整条工具调用轨迹中的结果反馈到模型训练。

第五，如何确保训练与评测任务之间不存在泄漏。

教授说：

“这已经不仅仅是 PyTorch API 的问题了。”

“而是模型训练、Agent Runtime、环境设计与评测体系的交叉工程。”

# 二十五、今天的世界地图

林默在白板上画下：

```
                      Pre-training
                           │
                           ▼
                        Base Model
                           │
                           ▼
                           SFT
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               偏好数据          可验证任务
                  │                 │
                  ▼                 ▼
                 DPO             RL / PPO 等
                  │                 │
                  └────────┬────────┘
                           ▼
                     Post-trained Model
                           │
                           ▼
                      Evaluation
                           │
               ┌───────────┼───────────┐
               ▼           ▼           ▼
           事实正确性    指令遵循     任务完成率
               │           │           │
               └───────────┼───────────┘
                           ▼
                      部署与监控
```

林默望着这张图。

他突然明白了一件事。

预训练、SFT 和偏好优化并不是相互割裂的三套系统。

底层依旧是：

```
Tensor
   ↓
Forward
   ↓
Loss / Objective
   ↓
Autograd
   ↓
Optimizer
   ↓
Parameter Update
```

改变的是：

> 训练数据、损失函数、反馈来源，以及我们希望模型最终学到的行为。

## 二十六、五道理解测试

第十三章随堂测试

0/5 已完成

1\. SFT 中把 Prompt 对应的 Label 设为 -100，主要是为了什么？

禁止模型读取 Prompt

使相应目标位置不参与 CrossEntropyLoss

把 Prompt 自动转成 Embedding

2\. DPO 的经典目标主要利用什么数据？

只有原始预训练文本

Chosen / Rejected 成对偏好数据

只有模型参数数量

3\. Reward Model 的评分应该如何理解？

绝对客观的正确性证明

根据反馈数据学习出来的评分估计

与训练数据无关的固定数学真理

4\. PPO 中的 Clip 机制主要解决什么问题？

抑制策略更新中的过大变化激励

减少模型词表长度

保证所有生成结果都正确

5\. 为什么不能只看 SFT Loss 判断模型有没有变强？

Loss 不可以计算

Loss 只衡量训练目标，不能充分反映未见任务上的表现

因为 SFT 不使用反向传播

# 第十四章预告：《失控的训练工厂——分布式训练、DDP、FSDP、ZeRO 与模型并行》

凌晨四点。

林默完成后训练实验，终于得到了一个回答质量有所改善的模型。

但新的问题很快出现。

公司要求：

```
模型规模：70B

训练数据：数十亿 Token

训练设备：多张 GPU
```

林默看着自己的单张显卡。

“教授，这次 QLoRA 也不能解决全部问题吧？”

教授点头。

“当单张 GPU 已经不够，我们就必须让多张 GPU 协同工作。”

他写下：

```
Data Parallelism
       ↓
DDP
       ↓
FSDP / ZeRO
       ↓
Tensor Parallelism
       ↓
Pipeline Parallelism
       ↓
Distributed Training
```

“下一章，我们将进入多 GPU 训练的世界。”

重点包括： 为什么 DDP 每张卡都有完整模型；AllReduce 如何同步梯度；FSDP 和 ZeRO 如何分片模型状态；Tensor Parallel 与 Pipeline Parallel 各解决什么问题；以及通信带宽、计算效率和显存之间的工程权衡。

> 从下一章开始，我们研究的不再是怎样让一张 GPU 完成训练，而是怎样让一整个 GPU 集群像一台机器那样工作。&#x20;