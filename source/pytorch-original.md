按照小说的方式讲解一下pytorch

可以。我们继续用你之前学 Java 八股、计网、Git 的那种**连续小说模式**来讲 PyTorch：不先堆 API，而是让概念在剧情里出现。

这次从真正的最基础开始：**Tensor → 计算图 → 自动求导 → 神经网络 → 优化器 → Dataset/DataLoader → GPU → 完整训练流程 → Transformer/LLM**。

# 《PyTorch：炼金术士的学徒》

## 第一章：一块奇怪的石板

凌晨两点。

林默盯着屏幕上的代码。

```python
import torch
```

只有这一行。

他已经看了半个小时。

“所以……PyTorch 到底是什么？”

导师陈教授把咖啡放到桌上。

“你以前写 Java，对吧？”

“对。”

“那你应该习惯这样的世界。”

```java
int a = 10;
int b = 20;
int c = a + b;
```

“变量、对象、方法、数据结构。”

林默点头。

教授打开 Python，在屏幕上敲下：

```python
import torch

x = torch.tensor([1, 2, 3])
```

屏幕上出现：

```text
tensor([1, 2, 3])
```

“欢迎来到 PyTorch 世界。”

“这里最基础的东西，不是 `int`，不是 `double`。”

教授指着那个单词。

> **Tensor。**

---

## 一、Tensor 到底是什么？

林默皱眉。

“Tensor……张量？我每次看到这个词就觉得像数学论文。”

教授笑了。

“先忘掉‘张量’这个翻译。”

他在白板上写：

```text
一个数字
5
```

“这是几维？”

“0维？”

“对。”

```python
x = torch.tensor(5)
```

这是一个 **0 维 Tensor**。

接着教授写：

```text
[1, 2, 3]
```

“这个呢？”

“一维数组。”

“在 PyTorch 里，就是 **1 维 Tensor**。”

```python
x = torch.tensor([1, 2, 3])
```

然后：

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

林默马上回答：

“二维数组。”

“也就是二维 Tensor。”

```python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

教授继续画：

```text
数字              0维 Tensor
数组              1维 Tensor
矩阵              2维 Tensor
一堆矩阵          3维 Tensor
一堆3维数据       4维 Tensor
...
```

林默突然明白了。

“所以 Tensor 可以粗略理解成——**N 维数组？**”

“对。”

“那不就是 NumPy 的 `ndarray`？”

教授停顿了一下。

“很好。”

“这就是第一个关键问题。”

---

# 二、那 PyTorch 和 NumPy 有什么区别？

教授写下：

```python
import numpy as np

a = np.array([1, 2, 3])
```

然后：

```python
import torch

b = torch.tensor([1, 2, 3])
```

看起来确实差不多。

加法也差不多：

```python
a + a
```

和：

```python
b + b
```

都得到：

```text
[2, 4, 6]
```

“那 PyTorch 凭什么专门搞个 Tensor？”

教授没有回答。

而是敲下：

```python
x = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)
```

林默注意到了一个陌生参数：

```text
requires_grad=True
```

“这是干什么？”

教授只说了一句话。

“让这块石头——”

“**拥有记忆。**”

---

## 三、会记住自己经历过什么的数字

教授继续输入：

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * x
```

数学上就是：

\[
y=x^2
\]

如果：

```text
x = 2
```

那么：

```text
y = 4
```

这没什么神奇。

教授继续：

```python
y.backward()
```

然后：

```python
print(x.grad)
```

输出：

```text
tensor(4.)
```

林默愣住了。

“4？”

“为什么是4？”

教授在白板上写：

\[
y=x^2
\]

求导：

\[
\frac{dy}{dx}=2x
\]

因为：

\[
x=2
\]

所以：

\[
\frac{dy}{dx}=4
\]

林默看着：

```python
x.grad
```

又看了看白板。

“等等。”

“你是说……”

“**PyTorch 自动把导数算出来了？**”

“对。”

---

# 四、真正重要的不是 Tensor，而是它背后的世界

教授重新写了一段：

```python
x = torch.tensor(2.0, requires_grad=True)

a = x * 3
b = a + 4
y = b * b
```

如果只看代码：

```text
x
↓
x * 3
↓
+ 4
↓
平方
↓
y
```

实际上，PyTorch 在背后悄悄建立了一张图：

```text
        x
        │
       ×3
        │
        a
        │
       +4
        │
        b
       / \
      b   b
       \ /
        ×
        │
        y
```

这就是：

> **Computational Graph——计算图。**

林默盯着图。

“所以我每进行一次运算……”

“PyTorch 都可能记录下来？”

“准确地说，当相关 Tensor 需要梯度时，PyTorch 会跟踪这些运算。”

教授指着：

```python
requires_grad=True
```

“这句话相当于告诉 PyTorch：”

> 这个变量以后我要对它求导，你把相关计算过程记下来。

---

# 五、为什么神经网络非得求导？

林默问出了最重要的问题。

“但是训练神经网络为什么一定要求导？”

教授没有回答，而是给了他一道小学题。

“假设我要找到一个数字 `w`，让它满足：”

\[
2w \approx 10
\]

“答案是多少？”

“5。”

“但是假设机器不知道。”

初始：

```text
w = 1
```

那么：

```text
2 × 1 = 2
```

距离目标 10 很远。

于是定义一个“错误程度”：

\[
Loss=(2w-10)^2
\]

当：

```text
w = 1
```

时：

\[
Loss=(2-10)^2=64
\]

现在机器需要解决一个问题：

> **w 应该往哪个方向调整，才能让 Loss 变小？**

林默沉默几秒。

“看 Loss 对 w 的导数？”

教授点头。

---

## 六、梯度其实就是“往哪边走”

教授画了一座山谷：

```text
Loss
 ^
 |       *
 |      / \
 |     /   \
 |    /     \
 |   /       \
 |__/_________\____> w
          ↓
       最低点
```

神经网络训练，本质上是在找：

> **Loss 最低的位置。**

而梯度告诉你：

```text
当前位置的坡度
```

假设：

```text
gradient > 0
```

说明往右走，Loss 变大。

所以应该：

```text
往左走
```

也就是：

```python
w = w - learning_rate * gradient
```

林默突然盯住了这个公式。

“等等……”

```text
新参数
=
旧参数
-
学习率 × 梯度
```

教授点头。

“这就是你以后天天见到的——”

> **梯度下降。**

---

# 七、第一次真正的“训练”

教授打开编辑器。

“现在不用任何神经网络 API。”

“我们手搓一次训练。”

目标：

\[
y=2x
\]

训练数据只有一个：

```text
x = 5
y = 10
```

代码：

```python
import torch

w = torch.tensor(1.0, requires_grad=True)

x = torch.tensor(5.0)
target = torch.tensor(10.0)

for i in range(10):

    y = w * x

    loss = (y - target) ** 2

    loss.backward()

    with torch.no_grad():
        w -= 0.01 * w.grad

    w.grad.zero_()

    print(w.item(), loss.item())
```

林默看着这段代码。

只有十几行。

但教授告诉他：

“你已经写出了一个极简版机器学习训练系统。”

---

# 八、逐行拆开

第一句：

```python
w = torch.tensor(1.0, requires_grad=True)
```

`w` 是我们想学习的参数。

初始：

```text
w = 1
```

但正确答案应该接近：

```text
w = 2
```

---

第二步：

```python
y = w * x
```

这叫：

> **Forward / 前向传播**

拿当前参数计算预测结果。

例如：

```text
w = 1
x = 5

y = 5
```

但是正确答案：

```text
target = 10
```

---

第三步：

```python
loss = (y - target) ** 2
```

计算：

```text
预测得有多烂
```

这就是：

> **Loss Function / 损失函数**

预测：

```text
5
```

答案：

```text
10
```

于是：

```text
loss = 25
```

---

第四步：

```python
loss.backward()
```

这一句话极其重要。

PyTorch 开始沿着计算图：

```text
Loss
 ↑
 y
 ↑
 w
```

反向计算：

\[
\frac{\partial Loss}{\partial w}
\]

这就是：

> **Backward / 反向传播。**

---

第五步：

```python
w -= 0.01 * w.grad
```

根据梯度修改参数。

其中：

```text
0.01
```

叫：

> **Learning Rate / 学习率。**

---

第六步：

```python
w.grad.zero_()
```

林默突然问：

“为什么要清零？”

教授笑了。

“这是 PyTorch 新手最容易踩的坑之一。”

---

# 九、梯度默认会累加

假设第一次：

```text
w.grad = 3
```

第二次又算出：

```text
w.grad = 4
```

很多初学者以为最后：

```text
w.grad = 4
```

实际上 PyTorch 默认可能变成：

```text
w.grad = 7
```

因为：

> **PyTorch 的梯度默认累加。**

所以每轮训练通常需要：

```python
optimizer.zero_grad()
```

或者刚才手写版本里的：

```python
w.grad.zero_()
```

林默把这句话记在笔记里：

> backward 之前/每轮训练要注意清梯度，否则梯度会累积。

---

# 十、为什么更新参数要 `torch.no_grad()`？

还有一段奇怪代码：

```python
with torch.no_grad():
    w -= 0.01 * w.grad
```

教授问：

“我们修改 `w` 的这个操作，需要参与神经网络学习吗？”

“不需要。”

“那就没必要把它记录进计算图。”

所以：

```python
with torch.no_grad():
```

意思可以暂时理解成：

> **接下来这些计算不用追踪梯度。**

这不仅减少额外记录，也避免对需要梯度的叶子 Tensor 做不合适的原地更新。

---

# 十一、第一张 PyTorch 世界地图

教授终于在黑板上画出了整套流程：

```text
             数据
              │
              ▼
        ┌───────────┐
        │   Model   │
        │   模型     │
        └───────────┘
              │
              ▼
          prediction
            预测值
              │
              ▼
        ┌───────────┐
        │   Loss    │
        │  损失函数   │
        └───────────┘
              │
              ▼
       loss.backward()
              │
              ▼
           梯度 grad
              │
              ▼
      optimizer.step()
              │
              ▼
           更新参数
              │
              └─────────┐
                        │
                        ▼
                    下一轮训练
```

然后他在旁边写下 PyTorch 最核心的一套训练模板：

```python
for x, y in dataloader:

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(pred, y)

    loss.backward()

    optimizer.step()
```

林默看着这五行。

教授说：

“以后你看到 CNN、RNN、Transformer，甚至很多大模型训练代码，外面套的骨架，本质上都逃不出这几个步骤。”

```text
1. 清梯度
2. 前向传播
3. 算 Loss
4. 反向传播
5. 更新参数
```

“真正变化的，只是里面的模型越来越复杂。”

---

## 十二、林默第一次理解了 PyTorch

凌晨三点。

他重新看向：

```python
import torch
```

这一次，它不再只是一个陌生的 Python 库。

他脑海里已经出现了一套完整的世界：

```text
Tensor
   │
   ├── 保存数据
   │
   └── 参与运算
          │
          ▼
       计算图
          │
          ▼
        Loss
          │
          ▼
      backward()
          │
          ▼
       gradient
          │
          ▼
     更新 parameter
          │
          ▼
       Loss下降
```

而所谓：

> **训练神经网络**

第一次变成了一句非常朴素的话：

> **让模型算一次 → 看错多少 → 算出参数该往哪改 → 改一点 → 再来一次。**

教授收起电脑。

“今天先到这里。”

林默问：

“下一章是什么？”

教授停在门口。

“你现在只有一块会自动求导的石头。”

“下一章，我们把成千上万个参数组织起来。”

他在黑板上留下三个词：

```text
nn.Module

Linear

Parameter
```

**下一章：《神经网络诞生——nn.Module、Linear 与 Parameter》**。

这一章会真正开始写第一个 PyTorch 神经网络，并解释一个很容易只会背、却不知道原理的问题：

> `model.parameters()` 里面到底装了什么？为什么 `optimizer` 能知道应该更新哪些东西？


下一章

# 《PyTorch：炼金术士的学徒》

## 第二章：神经网络诞生——`nn.Module`、`Linear` 与 `Parameter`

第二天。

林默重新打开昨天的代码：

```python
import torch

w = torch.tensor(1.0, requires_grad=True)

x = torch.tensor(5.0)
target = torch.tensor(10.0)

for i in range(10):
    y = w * x
    loss = (y - target) ** 2

    loss.backward()

    with torch.no_grad():
        w -= 0.01 * w.grad

    w.grad.zero_()
```

他看了半天。

突然意识到一个问题。

“教授。”

“如果模型只有一个 `w`，我确实可以自己写。”

“但是 ChatGPT 那种模型有几十亿、几百亿参数……”

“难道要写——”

```python
w1 = ...
w2 = ...
w3 = ...
w4 = ...
...
```

教授喝了一口咖啡。

“所以今天，我们要制造一个真正的**模型**。”

---

# 一、从一个神经元开始

教授在黑板上写：

\[
y=wx+b
\]

“认识吗？”

林默点头。

“线性函数。”

“很好。神经网络最基础的结构之一，本质上就是这个。”

```text
输入 x
  │
  ▼
× w
  │
  ▼
+ b
  │
  ▼
输出 y
```

这里有两个东西需要学习：

```text
w —— weight，权重
b —— bias，偏置
```

比如：

\[
y=2x+1
\]

当：

\[
x=3
\]

那么：

\[
y=7
\]

教授说：

“机器学习的问题在于——”

> **我们不知道 `w` 和 `b` 应该是多少。**

我们只有很多数据：

```text
x       y
---------
1       3
2       5
3       7
4       9
```

人类一眼就发现：

\[
y=2x+1
\]

但机器不知道。

所以训练开始时可能：

```text
w = -0.37
b = 0.82
```

然后不断：

```text
预测
 ↓
算 Loss
 ↓
backward
 ↓
修改 w、b
 ↓
再预测
```

最终慢慢变成：

```text
w ≈ 2
b ≈ 1
```

林默点头。

“那我自己定义两个 Tensor 不就行了？”

```python
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y = w * x + b
```

“当然可以。”

教授说。

“但现在假设，不是一个输入。”

---

# 二、当一个神经元拥有很多输入

假设要预测一个程序员的面试成绩。

输入有三个：

```text
x1 = Java 基础
x2 = 算法能力
x3 = 项目能力
```

模型可能是：

\[
y=w_1x_1+w_2x_2+w_3x_3+b
\]

例如：

```text
Java     = 0.8
算法     = 0.9
项目     = 0.7
```

对应：

```text
w1
w2
w3
b
```

都是模型要学习的东西。

于是可以写成：

\[
y=
\begin{bmatrix}
x_1 & x_2 & x_3
\end{bmatrix}
\begin{bmatrix}
w_1\\
w_2\\
w_3
\end{bmatrix}
+b
\]

林默看了一眼。

“矩阵乘法。”

“对。”

“而神经网络里面，大量计算，本质就是矩阵运算。”

教授打开 PyTorch：

```python
import torch.nn as nn

layer = nn.Linear(3, 1)
```

林默愣住。

“就这一行？”

“就这一行。”

---

# 三、`nn.Linear(3, 1)` 到底创建了什么？

教授把它拆开：

```python
nn.Linear(
    in_features=3,
    out_features=1
)
```

意思是：

```text
输入 3 个数字
      │
      ▼
┌──────────────┐
│    Linear    │
│              │
│ w1  w2  w3   │
│      + b     │
└──────────────┘
      │
      ▼
输出 1 个数字
```

数学公式就是：

\[
y=xW^T+b
\]

先不用纠结转置。

你可以暂时把它理解为：

\[
y=w_1x_1+w_2x_2+w_3x_3+b
\]

于是：

```python
layer = nn.Linear(3, 1)
```

实际上 PyTorch 已经偷偷帮你创建了：

```text
3 个 weight
+
1 个 bias
```

林默不信。

教授说：

“打印看看。”

```python
print(layer.weight)
print(layer.bias)
```

你会看到类似：

```text
Parameter containing:
tensor([[ 0.2134, -0.3812, 0.5123]], requires_grad=True)
```

以及：

```text
Parameter containing:
tensor([0.1273], requires_grad=True)
```

林默注意到一个新东西：

> `Parameter`

“Tensor 怎么突然变成 Parameter 了？”

教授笑了。

“终于问到今天真正重要的地方了。”

---

# 四、Tensor 与 Parameter

教授写下：

```python
torch.Tensor
```

然后在下面写：

```python
torch.nn.Parameter
```

“你可以先这么理解。”

```text
Tensor
│
│ 普通数据
│
├── 输入数据
├── 标签
├── 中间计算结果
└── 其他数值


Parameter
│
│ 一种特殊 Tensor
│
└── 告诉模型：
    ‘我是需要训练的参数’
```

比如：

```python
x = torch.tensor([1.0, 2.0, 3.0])
```

这个 `x` 通常只是：

> **数据。**

而：

```python
layer.weight
```

则是：

> **模型需要学习的参数。**

林默马上问：

“那 `Parameter` 不就是 `requires_grad=True` 的 Tensor？”

教授摇头。

“不完全一样。”

---

# 五、Parameter 最关键的能力：注册

教授写了一个模型：

```python
class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.tensor(1.0))
```

然后：

```python
model = MyModel()
```

再执行：

```python
for p in model.parameters():
    print(p)
```

可以找到：

```text
Parameter containing:
tensor(1., requires_grad=True)
```

“因为 `self.w` 是 `Parameter`，并且它被挂在 `nn.Module` 下面，所以 PyTorch 会把它**注册为模型参数**。”

教授擦掉代码，换成：

```python
class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.w = torch.tensor(
            1.0,
            requires_grad=True
        )
```

“现在它虽然也能算梯度，但是——”

```python
list(model.parameters())
```

并不会因为它只是普通 Tensor 就自动把它当作注册参数。

林默突然理解了。

> `requires_grad=True`

解决的是：

```text
要不要计算梯度？
```

而：

> `nn.Parameter`

还承担着：

```text
这是模型的一个正式参数，
请 nn.Module 管理它。
```

“对。”

教授点头。

“这个区别非常重要。”

---

# 六、那 `nn.Module` 又是什么？

林默继续追问：

“所以 `nn.Module` 就是模型？”

“更准确地说——”

教授想了一下。

“它是 PyTorch 构建神经网络的**基本组织单位**。”

比如：

```python
class MyModel(nn.Module):
    ...
```

可以理解成：

```text
Java：

class UserService {
    ...
}
```

PyTorch：

```text
class MyModel(nn.Module):
    ...
```

它帮你管理很多东西：

```text
nn.Module
│
├── Parameters
│   ├── weight
│   └── bias
│
├── 子 Module
│   ├── Linear
│   ├── ReLU
│   └── ...
│
├── train / eval 状态
│
├── state_dict
│
└── CPU / GPU 迁移
```

林默看到了一个关键点。

“等等。”

“你说 `Linear` 也是 Module？”

“对。”

---

# 七、模型其实是一棵树

教授写：

```python
class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
```

看起来：

```text
MyModel
│
├── fc1
│   └── Linear
│       ├── weight
│       └── bias
│
└── fc2
    └── Linear
        ├── weight
        └── bias
```

所以一个大型神经网络，本质上可以看成：

```text
Module
├── Module
│   ├── Parameter
│   └── Parameter
│
├── Module
│   ├── Module
│   │   ├── Parameter
│   │   └── Parameter
│   └── Module
│
└── Module
```

层层嵌套。

最终：

> **整个模型就是一棵 Module 树。**

这也是为什么你可以：

```python
model.parameters()
```

拿到整个模型里的参数。

PyTorch 会沿着这棵树递归寻找。

---

# 八、`model.parameters()` 到底是什么？

教授现在写下：

```python
model = nn.Linear(3, 1)
```

这个模型内部大概有：

```text
weight:
[w1, w2, w3]

bias:
[b]
```

于是：

```python
for param in model.parameters():
    print(param)
```

就会遍历：

```text
weight
bias
```

林默突然想到了昨天见过的代码：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

他猛地坐直了。

“等等！”

“所以这里——”

```python
model.parameters()
```

“实际上是在把所有需要训练的参数交给 optimizer？”

教授点头。

“正是如此。”

---

# 九、Optimizer 并不知道什么叫神经网络

这是一个很容易产生误解的地方。

很多人看到：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

会下意识觉得：

> Adam 知道这个模型是什么，然后帮模型训练。

其实不是。

Optimizer 根本不关心：

```text
你是不是 Transformer
你是不是 CNN
你是不是 RNN
你是不是大模型
```

它只拿到了一堆：

```text
Parameter
```

例如：

```text
[
    fc1.weight,
    fc1.bias,
    fc2.weight,
    fc2.bias
]
```

然后：

```python
optimizer.step()
```

干的核心事情就是：

```text
看看这些 Parameter 的 .grad
            ↓
按照优化算法
            ↓
修改 Parameter 的值
```

比如最简单的 SGD：

\[
w=w-\eta\frac{\partial L}{\partial w}
\]

林默看着昨天自己手写的：

```python
with torch.no_grad():
    w -= 0.01 * w.grad
```

再看看今天的：

```python
optimizer.step()
```

突然笑了。

“原来这俩本质上是在干同一类事情。”

“对。”

> `optimizer.step()` 可以看作把“根据梯度更新参数”这件事封装了起来。

当然 Adam 比简单 SGD 的更新规则复杂得多，但核心目标没变。

---

# 十、现在造出第一个真正的神经网络

教授说：

“开始。”

任务：

```text
输入两个数字
↓
预测一个数字
```

建立模型：

```python
import torch
import torch.nn as nn

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x
```

林默沉默了。

“这就是……神经网络？”

“是。”

---

# 十一、拆开 `__init__`

先看：

```python
self.fc1 = nn.Linear(2, 4)
```

意思是：

```text
2 个输入
   │
   ▼
4 个输出
```

可以想象成：

```text
x1 ─────┬────► h1
        ├────► h2
        ├────► h3
        └────► h4

x2 ─────┬────► h1
        ├────► h2
        ├────► h3
        └────► h4
```

每条连接都有一个 weight。

所以：

```python
self.fc1.weight.shape
```

是：

```text
torch.Size([4, 2])
```

为什么？

因为有：

```text
4 个输出神经元
×
每个神经元接收 2 个输入
```

所以一共：

\[
4\times2=8
\]

个 weight。

另外：

```python
self.fc1.bias.shape
```

是：

```text
[4]
```

因为四个输出各自有一个 bias。

所以第一层总参数：

\[
8+4=12
\]

---

第二层：

```python
self.fc2 = nn.Linear(4, 1)
```

有：

\[
1\times4=4
\]

个 weight。

加一个 bias：

\[
4+1=5
\]

整个网络：

\[
12+5=17
\]

个参数。

林默看着：

```python
model.parameters()
```

突然觉得这个 API 不再神秘。

里面其实就是这 **17 个可学习数值组成的参数 Tensor**。

---

# 十二、`forward()` 是什么？

教授指向：

```python
def forward(self, x):
```

“模型拿到数据以后，怎么计算，就写在这里。”

例如：

```python
x = self.fc1(x)
```

经过第一层：

```text
输入
[ x1, x2 ]

     ↓ Linear

[ h1, h2, h3, h4 ]
```

然后：

```python
x = torch.relu(x)
```

再：

```python
x = self.fc2(x)
```

最终：

```text
4维
 ↓
1维
```

得到预测值。

---

# 十三、为什么调用的是 `model(x)`？

林默突然发现一个奇怪的地方。

我们明明定义的是：

```python
def forward(self, x):
```

但真正使用的时候却通常写：

```python
y = model(x)
```

而不是：

```python
y = model.forward(x)
```

“为什么？”

教授说：

“因为 `nn.Module` 实现了调用机制。”

你写：

```python
model(x)
```

PyTorch 会经过 `Module` 自己的调用流程，最终执行你的：

```python
forward(x)
```

可以粗略理解成：

```text
model(x)
   │
   ▼
Module.__call__()
   │
   ├── 一些 PyTorch 内部处理
   │
   ▼
forward(x)
   │
   ▼
结果
```

所以一般：

> **调用模型用 `model(x)`，不要手动直接调 `model.forward(x)`。**

---

# 十四、但是这个网络有一个奇怪的东西

林默指着：

```python
x = torch.relu(x)
```

“这个 ReLU 是干什么的？”

教授没有直接回答。

而是在黑板上写：

\[
Linear(Linear(x))
\]

假设第一层：

\[
h=W_1x+b_1
\]

第二层：

\[
y=W_2h+b_2
\]

代进去：

\[
y=W_2(W_1x+b_1)+b_2
\]

展开：

\[
y=W_2W_1x+W_2b_1+b_2
\]

把：

\[
W_2W_1
\]

看成新的：

\[
W
\]

把：

\[
W_2b_1+b_2
\]

看成新的：

\[
b
\]

那么：

\[
y=Wx+b
\]

林默盯着公式。

过了几秒。

“等等……”

“两个 Linear 连起来……”

“最后还是一个 Linear？”

“对。”

“那我堆一百层 Linear——”

“如果中间什么都不加，本质上仍然只是一个线性变换。”

林默终于看向：

```python
torch.relu(x)
```

“所以它是来打破线性的？”

教授点头。

“下一章的门已经出现了。”

---

# 十五、把模型真正训练起来

不过教授没有马上结束。

“先把今天所有东西拼起来。”

```python
import torch
import torch.nn as nn

# 1. 数据
x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
])

target = torch.tensor([
    [3.0],
    [5.0],
    [7.0]
])


# 2. 模型
class MyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x


model = MyNet()


# 3. 损失函数
loss_fn = nn.MSELoss()


# 4. 优化器
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# 5. 训练
for epoch in range(1000):

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(pred, target)

    loss.backward()

    optimizer.step()
```

林默看着这段代码。

昨天所有零散的东西，终于连接起来了。

---

# 十六、PyTorch 训练链路终于完整了

教授画下：

```text
                    MyNet
                      │
          ┌───────────┴───────────┐
          │                       │
       Linear                  Linear
          │                       │
    weight / bias           weight / bias
          │                       │
          └───────────┬───────────┘
                      │
              model.parameters()
                      │
                      ▼
                  Optimizer
```

训练的时候：

```text
x
│
▼
model(x)
│
▼
prediction
│
▼
loss_fn(prediction, target)
│
▼
Loss
│
▼
loss.backward()
│
▼
每个 Parameter.grad 有了梯度
│
▼
optimizer.step()
│
▼
根据 grad 修改 Parameter
│
▼
模型变了一点
│
▼
下一轮
```

林默盯着最后一张图。

突然说道：

“所以 `backward()` 和 `optimizer.step()` 其实是完全不同的事情。”

教授停下脚步。

“说下去。”

林默慢慢整理：

```text
loss.backward()
```

负责：

> **算梯度。**

也就是：

```text
每个参数应该往哪个方向改？
```

而：

```text
optimizer.step()
```

负责：

> **根据梯度真正修改参数。**

教授点头。

“非常重要。”

可以压缩成：

```text
backward()
    ↓
算答案

step()
    ↓
执行答案
```

更准确地说：

```text
backward()
→ 计算并累积梯度到 Parameter.grad

optimizer.step()
→ 根据优化算法和 Parameter.grad 更新参数
```

---

# 十七、三个概念终于连起来了

凌晨一点。

林默在自己的笔记最后写下：

```text
Tensor
↓
负责装数据、参与计算


Parameter
↓
特殊 Tensor
代表模型需要学习的参数
被 Module 注册和管理


nn.Module
↓
组织 Parameter 和其他 Module
组成整个神经网络
```

再往下：

```text
nn.Linear
↓
本身就是一个 nn.Module
↓
内部拥有
weight Parameter
bias Parameter
```

最后：

```text
model.parameters()
↓
收集模型注册的参数
↓
交给 optimizer
↓
optimizer 根据 .grad 更新它们
```

他忽然觉得，昨天那十几行代码和今天的神经网络，其实根本没有本质区别。

昨天：

```python
w -= lr * w.grad
```

今天：

```python
optimizer.step()
```

昨天：

```python
y = w * x
```

今天：

```python
pred = model(x)
```

昨天只有一个 `w`。

今天只不过有了成千上万个 `w`。

---

教授已经走到了门口。

林默却再次叫住他。

“刚才那个问题还没讲。”

“为什么神经网络一定需要 ReLU？”

教授回头看了一眼黑板。

黑板上，两层 Linear 已经被化简成了一层：

\[
W_2(W_1x+b_1)+b_2
\]

“因为如果没有它……”

“你造出来的所谓深度神经网络，无论叠多少层，都可能只是在做一件非常简单的事。”

灯熄灭。

黑板上只剩下一个词：

# `ReLU`

**下一章：《让神经网络学会拐弯——激活函数、ReLU 与非线性》**

下一章会解决一个特别关键的问题：**为什么 `Linear → ReLU → Linear` 能拟合复杂规律，而 `Linear → Linear → Linear` 堆一百层在表达能力上仍然只是线性变换。**这也是理解后面 MLP、CNN 乃至 Transformer 中非线性层的基础。


下一章

# 《PyTorch：炼金术士的学徒》

## 第三章：让神经网络学会拐弯——激活函数、ReLU 与非线性

第三天。

林默刚走进实验室，就发现陈教授已经在黑板上画了一条直线。

\[
y=2x+1
\]

“昨天你学了 `nn.Linear`。”

教授说。

“今天先回答一个问题。”

“如果我要用神经网络学习下面这种规律呢？”

他在黑板上画了一条弯曲的曲线：

```text
y
│              ●
│          ●
│       ●
│    ●
│  ●
│ ●
│●
└──────────────── x
```

林默想了一下。

“多堆几层 Linear？”

教授摇头。

“那我们今天就看看——”

> **为什么一百层 Linear，也救不了一条不会拐弯的直线。**

---

# 一、先回到 `nn.Linear`

昨天我们写过：

```python
layer = nn.Linear(2, 4)
```

它干的事情，本质上就是：

\[
y=Wx+b
\]

一维情况下更简单：

\[
y=wx+b
\]

比如：

\[
y=2x+1
\]

画出来：

```text
y
│
│           /
│         /
│       /
│     /
│   /
│ /
└──────────────── x
```

永远是一条直线。

参数 `w` 可以控制：

> **斜率。**

参数 `b` 可以控制：

> **上下平移。**

例如：

```text
w = 1       /
           /

w = 3         /
             /
            /
           /

w = -1    \
           \
            \
```

但无论怎么调整：

> **它永远不会弯。**

---

# 二、那我堆两层呢？

林默说：

“一个 Linear 不行，那两个呢？”

教授写：

```python
fc1 = nn.Linear(1, 10)
fc2 = nn.Linear(10, 1)
```

看起来已经很高级了：

```text
输入 x

   │
   ▼

○ ○ ○ ○ ○ ○ ○ ○ ○ ○
       第一层

   │
   ▼

   ○
 第二层

   │
   ▼

输出 y
```

十个隐藏神经元。

应该很厉害吧？

可教授开始推公式。

第一层：

\[
h=W_1x+b_1
\]

第二层：

\[
y=W_2h+b_2
\]

代进去：

\[
y=W_2(W_1x+b_1)+b_2
\]

展开：

\[
y=W_2W_1x+W_2b_1+b_2
\]

现在定义：

\[
W'=W_2W_1
\]

以及：

\[
b'=W_2b_1+b_2
\]

于是：

\[
y=W'x+b'
\]

教授把粉笔一扔。

“结束。”

林默：

“……”

两层 Linear：

```text
Linear
   ↓
Linear
```

最终仍然等价于：

```text
一个 Linear
```

---

# 三、那一百层呢？

林默还不死心。

```text
Linear
 ↓
Linear
 ↓
Linear
 ↓
Linear
 ↓
...
 ↓
Linear
```

教授直接写：

\[
W_{100}W_{99}\cdots W_2W_1x+\cdots
\]

无论中间怎么组合，最终仍然可以写成：

\[
W'x+b'
\]

也就是说：

> **线性变换叠加线性变换，最终仍然是线性变换。**

林默沉默了。

“所以我花几个亿显存堆了一百层……”

“结果数学表达能力可能还是一个 Linear？”

“如果中间没有非线性，是这个意思。”

这就像：

```text
直尺
+
直尺
+
直尺
+
100 把直尺
```

你不会因此得到圆规。

---

# 四、神经网络需要一个“破坏者”

教授在两个 Linear 中间插入一个东西：

```python
x = self.fc1(x)
x = torch.relu(x)
x = self.fc2(x)
```

结构变成：

```text
Linear
   ↓
 ReLU
   ↓
Linear
```

林默看着那个熟悉的名字。

“就是昨天的 `ReLU`。”

“对。”

“它负责破坏线性。”

---

# 五、ReLU 是什么？

教授写：

\[
ReLU(x)=\max(0,x)
\]

林默：

“就这？”

“就这。”

也就是说：

```text
输入 -5 → 0
输入 -2 → 0
输入 -1 → 0
输入  0 → 0
输入  1 → 1
输入  2 → 2
输入  5 → 5
```

Python 思维就是：

```python
def relu(x):
    return max(0, x)
```

PyTorch：

```python
x = torch.relu(x)
```

或者：

```python
relu = nn.ReLU()
x = relu(x)
```

它的图像是：

```text
y
│
│                /
│              /
│            /
│          /
│        /
│──────●──────────── x
│
│
```

当：

\[
x<0
\]

输出：

\[
0
\]

当：

\[
x>0
\]

输出：

\[
x
\]

林默盯着那个折角。

“它……”

“拐弯了。”

教授点头。

> **这一个小小的折角，就是神经网络拥有复杂表达能力的关键之一。**

---

# 六、Linear + ReLU 到底在干什么？

假设：

\[
z=wx+b
\]

然后：

\[
y=ReLU(z)
\]

比如：

\[
z=2x-4
\]

那么：

\[
y=ReLU(2x-4)
\]

什么时候开始大于 0？

\[
2x-4>0
\]

所以：

\[
x>2
\]

于是图像：

```text
y
│
│                  /
│                /
│              /
│            /
│──────────●──────────── x
           2
```

在 `x=2` 的地方出现一个：

> **拐点。**

---

# 七、一个 ReLU 一个拐点，那很多呢？

教授突然画了三个神经元。

第一个：

\[
h_1=ReLU(x-1)
\]

第二个：

\[
h_2=ReLU(x-3)
\]

第三个：

\[
h_3=ReLU(x-5)
\]

分别意味着：

```text
h1 在 x=1 开始工作
h2 在 x=3 开始工作
h3 在 x=5 开始工作
```

然后再让下一层把它们组合起来：

\[
y=w_1h_1+w_2h_2+w_3h_3+b
\]

你就可以得到：

```text
y
│
│              /
│        _____/
│       /
│  ____/
│ /
│/
└────────────────── x
   1    3    5
```

林默突然明白了。

“每个神经元都可以制造一个不同位置、不同方向的变化。”

“然后下一层把这些变化组合起来。”

“对。”

神经元越来越多：

```text
很多 Linear
     +
很多 ReLU
     ↓
制造大量折线片段
     ↓
组合
     ↓
逼近复杂函数
```

比如真正的曲线：

```text
        ~~~~~~~
     ~~~
   ~~
 ~~
```

神经网络可能用很多很小的折线去逼近：

```text
        __
      _/  \_
    _/      \_
  _/
_/
```

片段足够多以后，你肉眼看上去就越来越像一条曲线。

---

# 八、一个非常重要的概念：隐藏层

教授重新写昨天的模型：

```python
class MyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x
```

结构：

```text
输入层
x1  x2
│   │
└─┬─┘
  ▼

Linear(2, 4)

  ▼

h1 h2 h3 h4
│  │  │  │
└──┴──┴──┘
    ▼

   ReLU

    ▼

Linear(4, 1)

    ▼

   输出
```

中间：

```text
h1 h2 h3 h4
```

通常叫：

> **Hidden Layer——隐藏层。**

为什么叫隐藏层？

因为：

```text
输入 x
```

是我们能看到的数据。

```text
输出 y
```

也是我们关心的结果。

但是中间的：

```text
h1 h2 h3 h4
```

并不是人工直接提供的标签。

它们是模型内部学习出来的：

> **中间表示。**

---

# 九、“特征”开始出现了

教授给了一个更现实的例子。

假设输入是一张图片。

最原始的数据可能只是：

```text
像素
像素
像素
像素
像素
...
```

但是模型经过一层层变换：

```text
原始像素
   ↓
某些简单模式
   ↓
边缘 / 纹理
   ↓
局部形状
   ↓
更复杂表示
   ↓
猫 / 狗
```

这里需要注意：

这只是帮助理解的简化描述，真实神经网络内部学到的表示通常没这么干净，也未必能给每个神经元直接贴上“耳朵”“眼睛”这种人类可读标签。

但核心思想成立：

> **隐藏层是在不断把数据转换成更适合完成任务的表示。**

---

# 十、这就是 Representation Learning

林默问：

“传统机器学习不是经常要人工做特征工程吗？”

“对。”

比如以前做垃圾邮件分类，人可能手工设计：

```text
是否出现“中奖”
是否出现“免费”
感叹号数量
URL 数量
发件人特征
……
```

也就是：

```text
原始数据
    ↓
人类设计特征
    ↓
模型
```

而深度学习非常重要的思想之一就是：

```text
原始数据
    ↓
神经网络
    ↓
自动学习内部表示
    ↓
完成任务
```

这也是为什么：

> **深度神经网络不仅是在“拟合答案”，还在学习如何表示数据。**

林默在笔记上写：

```text
Hidden Layer
≈
模型内部学习特征表示的地方
```

---

# 十一、那为什么一定是 ReLU？

林默又问。

“非线性有很多吧？”

“当然。”

教授写下三个名字：

```text
Sigmoid
Tanh
ReLU
```

---

## Sigmoid

公式：

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

图像大概：

```text
1 │                ______
  │             __/
  │           _/
  │         _/
  │       _/
  │_____/
0 └──────────────────── x
```

它可以把任意实数压到：

\[
(0,1)
\]

所以看起来特别像：

```text
概率
```

比如：

```text
0.01
0.35
0.82
0.99
```

但要小心：

> Sigmoid 的输出可以解释成概率，需要具体任务和损失函数等条件配合，并不是“见到 Sigmoid 就一定是概率”。

---

# 十二、Sigmoid 的一个问题：梯度容易变小

教授在曲线两端画圈：

```text
          ___________
       __/
     _/
   _/
__/
```

当输入特别大：

```text
x = 100
```

Sigmoid 已经非常接近：

```text
1
```

曲线几乎变平。

而导数是什么？

> **斜率。**

曲线很平：

```text
斜率 ≈ 0
```

也就是：

```text
gradient ≈ 0
```

林默立刻联想到上一章：

```text
参数更新量
=
learning_rate × gradient
```

如果：

```text
gradient ≈ 0
```

那么：

```text
参数几乎不更新
```

“所以深层网络反向传播的时候，梯度可能越来越小？”

“对，这就是经典的——”

> **梯度消失（Vanishing Gradient）**

的一部分来源。

---

# 十三、ReLU 为什么流行？

再看：

\[
ReLU(x)=\max(0,x)
\]

当：

\[
x>0
\]

时：

\[
ReLU(x)=x
\]

它的导数：

\[
ReLU'(x)=1
\]

所以正区间不会像 Sigmoid 饱和区那样天然把梯度压得很小。

而且计算极其简单：

```text
x > 0 ?

是 → x
否 → 0
```

因此 ReLU 长期以来是非常常见的激活函数。

不过教授马上补了一句：

“不要记成‘ReLU 没有梯度问题’。”

因为当：

\[
x<0
\]

时：

\[
ReLU(x)=0
\]

导数也是：

\[
0
\]

如果一个神经元长期落在负区间，就可能一直输出 0、梯度也为 0。

这叫：

> **Dying ReLU。**

---

# 十四、神经网络训练，其实是在移动“折线”

教授重新画：

```text
Linear
↓
ReLU
↓
Linear
↓
ReLU
↓
Linear
```

训练开始：

```text
随机 weight
随机 bias
```

所以一开始模型形成的函数可能乱七八糟：

```text
预测
│
│   /\__
│__/    \____
│
└──────────────── x
```

然后：

```python
loss.backward()
```

计算：

> 每个 `weight` 和 `bias` 改一点，会让 Loss 怎么变化？

再：

```python
optimizer.step()
```

修改：

```text
weight
bias
```

这些参数一变：

```text
Linear 的斜率变了
Linear 的位置变了
↓
ReLU 的拐点位置也可能变化
↓
整个网络表达的函数发生变化
```

然后继续：

```text
预测
→ Loss
→ backward
→ step
→ 预测
→ Loss
→ backward
→ step
...
```

于是模型表示的函数逐渐靠近训练数据要求的规律。

---

# 十五、突然出现一道经典题

教授擦掉黑板。

写：

```text
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
```

“XOR。”

林默认识。

把四个点画出来：

```text
x2
│
1│ ●1       ●0
 │
 │
0│ ●0       ●1
 └────────────── x1
   0          1
```

目标是：

```text
一类：左上 + 右下
另一类：左下 + 右上
```

教授问：

“你能画一条直线把两类完全分开吗？”

林默试了半天。

“不行。”

“这就是经典的——”

> **线性不可分。**

单纯的线性模型没办法解决 XOR。

但：

```text
Linear
↓
非线性激活
↓
Linear
```

就有能力形成更复杂的决策边界。

林默这次真正明白了：

> **非线性不是为了让网络看起来复杂。**

> **它直接决定了网络能否表达大量线性模型无法表达的关系。**

---

# 十六、把网络写得更漂亮：`nn.Sequential`

教授看着之前的代码：

```python
class MyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

“如果网络只是按顺序执行，可以简单一点。”

```python
class MyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)
```

甚至直接：

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

林默看着：

```python
nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

突然觉得神经网络第一次有了一种搭积木的感觉：

```text
Linear
  ↓
ReLU
  ↓
Linear
```

---

# 十七、这东西其实有名字：MLP

教授在黑板上写：

> **MLP**

全称：

> **Multi-Layer Perceptron**

中文通常叫：

> **多层感知机**

典型结构：

```text
输入
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
输出
```

例如：

```python
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),

    nn.Linear(256, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)
```

形状变化：

```text
100
 ↓
256
 ↓
128
 ↓
10
```

这里：

```text
100 → 输入特征维度
256 → 隐藏维度
128 → 隐藏维度
10  → 输出维度
```

教授停顿了一下。

“记住这个结构。”

“以后你学 Transformer，会再次看到类似的东西。”

林默一愣。

“Transformer 里面也有 MLP？”

“当然。”

“而且非常重要。”

现代 Transformer block 里通常存在类似：

```text
Attention
   ↓
...
   ↓
MLP / FFN
```

不过其中的激活函数现代模型常见的已经不只是 ReLU，还会看到：

```text
GELU
SiLU / Swish
SwiGLU
```

这些以后再讲。

---

# 十八、今天真正应该记住什么？

夜里十一点。

教授没有让林默背：

```text
ReLU公式
Sigmoid公式
Tanh公式
```

而是在黑板上画了最后一张图：

```text
             Linear
                │
                ▼
       只能做线性变换
                │
                ▼
        再堆一个 Linear
                │
                ▼
         还是线性变换
                │
                ▼
            加 ReLU
                │
                ▼
          引入非线性
                │
                ▼
      可以表达复杂函数
                │
                ▼
       多层不断学习表示
                │
                ▼
         Deep Learning
```

林默在笔记本上写下了一句话：

> **Linear 负责变换，Activation 负责让这种变换拥有非线性表达能力。**

随后又补了一句：

> **神经网络之所以强，不只是因为“层数多”，而是因为多层可学习变换与非线性组合起来，可以形成复杂的函数。**

---

# 十九、但林默发现了新的问题

他重新看向训练代码：

```python
for epoch in range(1000):

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(pred, target)

    loss.backward()

    optimizer.step()
```

突然问：

“教授。”

“我们一直在说 `loss`。”

“但是——”

“这个 Loss 到底是谁规定的？”

教授停下脚步。

林默继续问：

“回归为什么经常用 MSE？”

“分类为什么又冒出来 CrossEntropy？”

“如果模型输出 10 个数字，怎么知道哪个数字代表猫？”

“还有我经常看到分类代码最后一层没有 `Softmax`，为什么 `CrossEntropyLoss` 又能直接算？”

教授慢慢转过身。

“很好。”

他在黑板上写：

```text
Regression
    ↓
MSE

Classification
    ↓
Logits
    ↓
Softmax
    ↓
Cross Entropy
```

然后在 `Softmax` 上画了一个巨大的问号。

---

# 下一章

## 《审判模型的法官——Loss、MSE、Softmax 与 CrossEntropy》

下一章我们会把一个特别容易混乱的链条彻底打通：

```text
模型最后输出的数字
        ↓
      Logits
        ↓
     Softmax
        ↓
       概率
        ↓
  Cross Entropy
        ↓
       Loss
        ↓
    backward()
```

并且会专门解释一个 PyTorch 初学者非常容易写错的问题：

> **为什么使用 `nn.CrossEntropyLoss()` 时，模型最后通常不要自己再加 `Softmax`？**

等这章打通之后，你看到最基本的 PyTorch 分类训练代码，基本就能知道**每一行到底为什么存在**。


下一章


# 《PyTorch：炼金术士的学徒》

## 第四章：审判模型的法官——Loss、MSE、Softmax 与 CrossEntropy

第四天。

林默刚走进实验室，就在黑板上写下昨天留下的问题：

```python
loss = loss_fn(pred, target)
```

“教授。”

“我现在大概明白了。”

```text
model(x)
   ↓
前向传播
   ↓
得到预测
   ↓
loss.backward()
   ↓
反向传播
```

“但中间这个 `loss_fn`……”

林默敲了敲黑板。

“凭什么它说模型错了，模型就错了？”

教授笑了。

“因为神经网络自己根本不知道什么叫‘正确’。”

---

# 一、模型其实没有目标

教授打开电脑：

```python
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

输入：

```python
x = torch.tensor([[2.0, 3.0]])
```

模型可能输出：

```text
tensor([[7.31]])
```

林默问：

“7.31 是对还是错？”

“我不知道。”

“为什么？”

“因为你没告诉我正确答案。”

教授点头。

假设真实答案是：

```text
target = 7
```

那么：

```text
prediction = 7.31
target     = 7.00
```

差一点。

但如果真实答案：

```text
target = 100
```

那就错得离谱。

所以：

> **模型只负责输出。**

> **Loss Function 负责定义：这个输出到底有多糟糕。**

---

# 二、Loss 就是模型世界里的“评分规则”

教授写：

```text
模型
 ↓
prediction
 ↓
┌───────────────┐
│ Loss Function │ ← target
└───────────────┘
 ↓
Loss
```

比如：

```text
预测 9.9
答案 10
```

Loss 应该比较小。

如果：

```text
预测 100
答案 10
```

Loss 应该很大。

因此训练真正优化的，并不是：

```text
“让模型变聪明”
```

这种模糊目标。

而是一个非常具体的数学目标：

\[
\min_\theta L(\theta)
\]

这里：

```text
θ = 模型所有参数
L = Loss
```

训练就是：

> **不断调整参数 θ，让 Loss 尽可能下降。**

---

# 三、第一位法官：MSE

教授写：

```python
loss_fn = nn.MSELoss()
```

MSE：

> **Mean Squared Error**

中文：

> **均方误差**

公式：

\[
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

林默看到公式就皱眉。

教授把它擦掉。

“别背，先看一个例子。”

真实房价：

```text
100
200
300
```

模型预测：

```text
110
180
310
```

先计算误差：

```text
预测    真实    误差

110     100     10
180     200    -20
310     300     10
```

然后平方：

```text
10²  = 100
-20² = 400
10²  = 100
```

求平均：

\[
\frac{100+400+100}{3}=200
\]

所以：

```text
MSE = 200
```

林默点头。

“所以就是——”

```text
预测和答案差多少
      ↓
平方
      ↓
取平均
```

“对。”

---

# 四、为什么要平方？

林默马上问：

“直接：

\[
prediction-target
\]

不行吗？”

教授写：

```text
样本 A：预测比答案高 10 → +10
样本 B：预测比答案低 10 → -10
```

如果直接平均：

\[
\frac{10+(-10)}{2}=0
\]

Loss：

```text
0
```

林默：

“这模型明明两个都预测错了。”

“所以不能让正负误差互相抵消。”

平方之后：

```text
10²  = 100
-10² = 100
```

就不会抵消。

而且大误差会受到更明显的惩罚：

```text
误差 1  → 1
误差 2  → 4
误差 10 → 100
```

---

# 五、MSE 经常出现在回归问题

比如：

```text
预测房价
预测温度
预测销量
预测耗时
```

输出本身就是一个连续数值：

```text
263.72 万
18.4 ℃
1932 件
27.6 ms
```

这种任务叫：

> **Regression——回归。**

典型结构：

```text
输入特征
   ↓
神经网络
   ↓
一个连续值
   ↓
MSE(pred, target)
```

PyTorch：

```python
pred = model(x)

loss = nn.MSELoss()(pred, target)

loss.backward()
```

到这里还比较简单。

教授擦掉黑板。

“现在进入真正容易混乱的地方。”

---

# 六、如果我要识别猫、狗、鸟呢？

现在任务：

```text
输入：一张图片

输出类别：
0 = 猫
1 = 狗
2 = 鸟
```

林默说：

“最后输出一个数字不就好了？”

```text
猫 → 0
狗 → 1
鸟 → 2
```

教授问：

“那模型预测 `1.7` 是什么？”

“……”

“70% 鸟？”

“不知道。”

更麻烦的是：

```text
猫 = 0
狗 = 1
鸟 = 2
```

如果直接使用数值距离，就隐含了：

```text
猫到狗距离 = 1
狗到鸟距离 = 1
猫到鸟距离 = 2
```

但类别编号只是标签。

> **猫并不因为编号是 0，就比编号为 2 的鸟离狗更远。**

所以分类不能简单把类别编号当连续数值做回归。

---

# 七、分类模型应该输出多个数字

三个类别：

```text
猫
狗
鸟
```

那么模型最后输出：

```python
nn.Linear(hidden_size, 3)
```

也就是：

```text
[2.1, 5.7, -1.3]
```

林默看着三个数字。

“这三个是什么意思？”

教授说：

“它们有一个名字。”

> **Logits**

---

# 八、Logits 是什么？

假设：

```text
猫： 2.1
狗： 5.7
鸟：-1.3
```

可以粗略理解：

```text
模型更倾向狗
```

因为：

```text
5.7 最大
```

所以预测类别：

```python
predicted_class = logits.argmax(dim=1)
```

得到：

```text
狗
```

但是注意：

```text
2.1
5.7
-1.3
```

**不是概率。**

因为概率通常应该满足：

```text
0 ≤ p ≤ 1
```

而且：

```text
所有类别概率之和 = 1
```

这里显然不是。

于是教授写下：

# Softmax

---

# 九、Softmax：把分数变成概率分布

Softmax 做的事情，可以暂时理解成：

```text
任意 logits

[2.1, 5.7, -1.3]

        ↓ Softmax

合法概率分布

[0.027, 0.972, 0.001]
```

满足：

```text
每个数在 0~1 之间

并且

0.027 + 0.972 + 0.001 = 1
```

于是：

```text
猫：  2.7%
狗： 97.2%
鸟：  0.1%
```

现在就很好理解了。

---

# 十、Softmax 怎么做到的？

教授终于写公式：

\[
Softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

林默皱眉。

“又开始了。”

“别看公式，直接算。”

假设 logits：

```text
[1, 2, 3]
```

先指数：

\[
e^1\approx2.72
\]

\[
e^2\approx7.39
\]

\[
e^3\approx20.09
\]

总和：

\[
30.20
\]

所以：

```text
类别0：
2.72 / 30.20 ≈ 0.09

类别1：
7.39 / 30.20 ≈ 0.245

类别2：
20.09 / 30.20 ≈ 0.665
```

最终：

```text
[0.09, 0.245, 0.665]
```

刚好：

```text
总和 = 1
```

---

# 十一、Softmax 真正重要的是“相对竞争”

教授突然问：

“如果 logits 是：”

```text
[1, 2, 3]
```

“第三类概率最高。”

“那如果全部加 100 呢？”

```text
[101, 102, 103]
```

林默想了一会。

“第三类还是最高。”

不仅如此。

Softmax 的结果实际上不会改变。

因为：

\[
Softmax(z+c)=Softmax(z)
\]

所以 Softmax 真正关心的是：

> **类别之间的相对差距。**

而不是某个 logit 自己绝对有多大。

---

# 十二、现在来了第二位法官：Cross Entropy

假设真实答案：

```text
狗
```

也就是：

```text
target = 1
```

模型 Softmax 后：

```text
猫：0.10
狗：0.80
鸟：0.10
```

这个预测挺好。

如果：

```text
猫：0.49
狗：0.02
鸟：0.49
```

这个预测就非常糟糕。

Cross Entropy 的核心思想可以先理解为：

> **看看模型给“正确类别”分配了多大概率。**

正确类别是狗。

第一种：

```text
P(狗) = 0.80
```

不错。

第二种：

```text
P(狗) = 0.02
```

很差。

---

# 十三、为什么还要取 `-log`？

对于单个样本的多分类问题，可以先看成：

\[
Loss=-\log(p_{\text{正确类别}})
\]

假设正确答案是狗。

如果模型：

```text
P(狗) = 0.9
```

那么：

\[
-\log(0.9)\approx0.105
\]

Loss 很小。

如果：

```text
P(狗)=0.5
```

那么：

\[
-\log(0.5)\approx0.693
\]

更大。

如果：

```text
P(狗)=0.01
```

那么：

\[
-\log(0.01)\approx4.605
\]

巨大。

所以：

```text
正确类别概率

1.0 ─────────────→ Loss 接近 0

0.8 ─────────────→ Loss 较小

0.5 ─────────────→ Loss 变大

0.1 ─────────────→ Loss 很大

0.01 ────────────→ Loss 巨大
```

教授说：

“尤其是——”

> **如果模型非常自信地预测错了，会受到很重的惩罚。**

---

# 十四、整个分类链路终于出现了

黑板上出现：

```text
                输入图片
                   │
                   ▼
                 Model
                   │
                   ▼
                 logits
                   │
          [2.1, 5.7, -1.3]
                   │
                   ▼
                Softmax
                   │
                   ▼
         [0.027, 0.972, 0.001]
                   │
                   ▼
       看正确类别对应的概率
                   │
                   ▼
                 -log
                   │
                   ▼
                  Loss
```

林默点头。

“那 PyTorch 应该这么写？”

```python
softmax = nn.Softmax(dim=1)

prob = softmax(model(x))

loss = nn.CrossEntropyLoss()(prob, target)
```

教授突然停住。

“错。”

林默：“？”

“这是今天最重要的坑。”

---

# 十五、`CrossEntropyLoss` 前面不要自己加 Softmax

PyTorch 中：

```python
loss_fn = nn.CrossEntropyLoss()
```

通常直接这样：

```python
logits = model(x)

loss = loss_fn(logits, target)
```

而不是：

```python
prob = torch.softmax(logits, dim=1)

loss = loss_fn(prob, target)
```

为什么？

因为：

> **`nn.CrossEntropyLoss()` 期望输入的是原始 logits。**

它内部已经把所需的 `LogSoftmax` 和 `NLLLoss` 组合起来了。

可以概念化成：

```text
CrossEntropyLoss

≈

LogSoftmax
+
NLLLoss
```

所以：

```python
loss = nn.CrossEntropyLoss()(
    logits,
    target
)
```

内部已经完成了对应计算。

---

# 十六、为什么 PyTorch 要这么设计？

林默问：

“直接 Softmax 再 log 不就行了吗？”

教授写：

```text
logits
 ↓
Softmax
 ↓
log
```

数学上能推。

但实际计算机使用有限精度浮点数。

假设某个类别概率极小：

```text
0.000000000000000000000000000001
```

数值计算可能出现精度问题。

因此实际实现通常会通过更稳定的数学形式计算。

所以 PyTorch 让你直接提供：

```text
logits
```

它内部做数值稳定处理。

因此要形成条件反射：

```python
nn.CrossEntropyLoss()
```

配：

```text
raw logits
```

**训练时通常不要在前面手动 Softmax。**

---

# 十七、那什么时候用 Softmax？

林默懵了。

“那 `torch.softmax()` 岂不是没用了？”

“当然不是。”

训练算 Loss：

```python
logits = model(x)

loss = loss_fn(logits, target)
```

但是推理时，如果你想看到：

```text
猫：2%
狗：97%
鸟：1%
```

可以：

```python
with torch.no_grad():

    logits = model(x)

    probs = torch.softmax(
        logits,
        dim=1
    )
```

所以：

```text
训练算 CrossEntropyLoss
        ↓
直接使用 logits


需要查看类别概率
        ↓
Softmax(logits)
```

---

# 十八、`dim=1` 又是什么鬼？

林默盯着：

```python
torch.softmax(logits, dim=1)
```

“为什么是 1？”

教授把 Tensor 形状写出来：

```text
batch_size = 4
num_classes = 3
```

模型输出：

```text
shape = [4, 3]
```

比如：

```text
[
  [2.1, 5.7, -1.3],   ← 样本1
  [3.2, 1.1,  0.7],   ← 样本2
  [0.1, 0.3,  4.2],   ← 样本3
  [1.2, 1.9,  0.4]    ← 样本4
]
```

维度：

```text
dim=0
  ↓
不同样本之间

dim=1
  ↓
一个样本的不同类别之间
```

我们希望：

```text
每一个样本内部
猫 + 狗 + 鸟
概率和 = 1
```

所以：

```python
torch.softmax(logits, dim=1)
```

也就是：

> **沿着类别维度做 Softmax。**

---

# 十九、这时候 Tensor 的 Shape 开始变得危险

教授突然问：

“模型输出是什么 shape？”

```text
[batch_size, num_classes]
```

“target 呢？”

林默试探：

```text
[batch_size, 1]？
```

“不。”

对于常见的多分类 `CrossEntropyLoss` 类别索引标签，通常是：

```text
[batch_size]
```

例如：

```python
logits = torch.tensor([
    [2.1, 5.7, -1.3],
    [3.2, 1.1,  0.7],
    [0.1, 0.3,  4.2]
])
```

shape：

```text
[3, 3]
```

target：

```python
target = torch.tensor([
    1,
    0,
    2
])
```

意思：

```text
样本1正确答案：类别1
样本2正确答案：类别0
样本3正确答案：类别2
```

shape：

```text
[3]
```

而且类别索引 target 通常应该是：

```text
torch.long
```

不是 float。

---

# 二十、第一个完整分类模型

教授终于写出：

```python
import torch
import torch.nn as nn

model = nn.Sequential(

    nn.Linear(10, 32),

    nn.ReLU(),

    nn.Linear(32, 3)
)
```

注意最后：

```python
nn.Linear(32, 3)
```

后面没有：

```python
nn.Softmax()
```

因为训练使用：

```python
loss_fn = nn.CrossEntropyLoss()
```

完整训练：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(100):

    optimizer.zero_grad()

    logits = model(x)

    loss = loss_fn(
        logits,
        target
    )

    loss.backward()

    optimizer.step()
```

预测：

```python
with torch.no_grad():

    logits = model(x)

    pred = logits.argmax(dim=1)
```

甚至如果只想知道哪个类别最大：

> **根本不需要 Softmax。**

因为 Softmax 不会改变 logits 的大小排序。

所以：

```python
logits.argmax(dim=1)
```

和：

```python
torch.softmax(logits, dim=1).argmax(dim=1)
```

预测类别相同。

---

# 二十一、林默突然发现一个大问题

他看着：

```python
loss.backward()
```

突然问：

“教授。”

“我们刚才不是有：

```text
Linear
↓
ReLU
↓
Linear
↓
CrossEntropy
↓
Loss
```

吗？”

“对。”

“那 `loss.backward()` 是不是会从 CrossEntropy 开始……”

然后：

```text
Loss
 ↑
CrossEntropy
 ↑
最后一个 Linear
 ↑
ReLU
 ↑
第一个 Linear
```

“一层一层往回算导数？”

教授没有说话。

林默继续：

“比如最后一层参数对 Loss 的影响。”

“然后上一层参数通过最后一层间接影响 Loss。”

“所以要用……”

他在黑板上写：

\[
\frac{\partial L}{\partial w}
\]

接着又写：

\[
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial h}
\frac{\partial h}{\partial w}
\]

教授终于笑了。

“你已经走到 PyTorch 最核心的地方了。”

---

# 二十二、链式法则

假设：

\[
x\rightarrow a\rightarrow b\rightarrow Loss
\]

其中：

\[
a=f(x)
\]

\[
b=g(a)
\]

\[
L=h(b)
\]

如果要求：

\[
\frac{dL}{dx}
\]

数学上：

\[
\frac{dL}{dx}
=
\frac{dL}{db}
\cdot
\frac{db}{da}
\cdot
\frac{da}{dx}
\]

这就是：

> **Chain Rule——链式法则。**

林默突然意识到：

“所以反向传播不是一种神秘的 AI 算法？”

“不是。”

> **反向传播的数学核心，就是利用链式法则高效地计算梯度。**

而 PyTorch 的 Autograd，则替你记录计算关系，并完成这些梯度计算。

---

# 二十三、昨天那个问题终于彻底回答了

第一章的时候，林默问：

> “反向传播其实就是计算导数吗？”

现在教授给出了更准确的答案：

**可以这么理解，但要再精确一点：**

> **反向传播是在计算图上，从最终 Loss 出发，利用链式法则反向传播梯度，高效计算各个需要梯度的参数对 Loss 的导数。**

也就是：

```text
forward
──────────────────────→

x
↓
Linear
↓
ReLU
↓
Linear
↓
CrossEntropy
↓
Loss


backward
←──────────────────────

Loss
↑
算梯度
↑
算梯度
↑
算梯度
↑
算梯度
↑
Parameter.grad
```

最后，每一个 Parameter 身上都有：

```python
param.grad
```

然后：

```python
optimizer.step()
```

读取这些梯度，修改参数。

---

# 二十四、PyTorch 的核心闭环终于形成了

教授关掉投影仪之前，留下了目前为止最重要的一张图：

```text
                数据 x
                  │
                  ▼
             ┌─────────┐
             │  Model  │
             └─────────┘
                  │
                  ▼
                logits
                  │
                  ▼
          ┌───────────────┐
target ──►│ Loss Function │
          └───────────────┘
                  │
                  ▼
                 Loss
                  │
                  │ backward()
                  ▼
         ┌──────────────────┐
         │   Autograd       │
         │   链式法则        │
         └──────────────────┘
                  │
                  ▼
         Parameter.grad
                  │
                  │ optimizer.step()
                  ▼
           Parameter 改变
                  │
                  └──────────────┐
                                 │
                                 ▼
                              下一轮
```

林默看了很久。

到这里，他终于能解释 PyTorch 最经典的五行代码：

```python
optimizer.zero_grad()

pred = model(x)

loss = loss_fn(pred, target)

loss.backward()

optimizer.step()
```

不是背 API。

而是：

```text
清掉上轮梯度

→ 用当前参数预测

→ 衡量预测有多差

→ 用链式法则计算每个参数对 Loss 的梯度

→ 根据梯度修改参数
```

这就是最基础的：

# **Training Loop**

---

夜已经深了。

林默合上电脑，却突然想到：

“等等。”

“我们一直写：

```python
x = torch.tensor([
    ...
])
```

”

“真正训练的时候，不可能一次把几百万张图片全塞进去吧？”

教授已经走到了门口。

“当然。”

“真正训练的时候，你还需要学会如何**喂数据**。”

他在黑板上留下三个词：

```text
Dataset

DataLoader

Batch
```

以及一个数字：

```text
1,000,000 条数据
        ↓
     ????
        ↓
     GPU
```

# 下一章

## 《粮草与运输队——Dataset、DataLoader、Batch 与 Epoch》

下一章会把训练代码中另一个经常让初学者“会写但不知道为什么”的部分彻底打通：

```python
for epoch in range(10):

    for x, y in dataloader:

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()
```

我们会解释清楚 **Sample、Batch、Batch Size、Iteration、Epoch、Shuffle、Dataset、DataLoader** 到底是什么，以及为什么训练神经网络通常既不是“一条一条训练”，也不是“所有数据一次性全塞进去”。


行，下一章。

# 《PyTorch：炼金术士的学徒》

## 第五章：粮草与运输队——Dataset、DataLoader、Batch 与 Epoch

第五天。

林默走进实验室的时候，陈教授没有打开 PyTorch。

桌子上反而放着一张纸。

上面写着：

```text
训练集：1,000,000 条数据
```

教授问：

“昨天我们的训练代码是什么？”

林默已经很熟了：

```python
for epoch in range(100):

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(pred, target)

    loss.backward()

    optimizer.step()
```

“很好。”

教授指着那一百万条数据。

“现在把它们训练起来。”

林默沉默了。

“一百万条……全部放进 `x`？”

教授笑了。

“欢迎来到真正的训练世界。”

---

# 一、首先认识 Sample

假设我们做一个非常简单的任务：

> 根据学习时间和刷题数量，预测考试是否通过。

一条数据：

```text
学习时间：5小时
刷题数量：100
结果：通过
```

可以写成：

```python
x = torch.tensor([5.0, 100.0])
y = torch.tensor(1)
```

这就是一个：

> **Sample——样本**

所以：

```text
一个 Sample

x = 特征
y = 标签
```

如果有 10000 个学生：

```text
Sample 1
Sample 2
Sample 3
...
Sample 10000
```

这些样本集合起来，就是：

> **Dataset——数据集。**

---

# 二、Dataset 本质上解决两个问题

教授打开 PyTorch：

```python
from torch.utils.data import Dataset
```

然后写：

```python
class StudentDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

林默第一反应：

“怎么又继承？”

教授说：

“别被代码吓到。”

一个 Dataset 最核心就是回答两个问题：

```text
问题一：

你有多少条数据？

        ↓

__len__()
```

以及：

```text
问题二：

给你编号 i，
第 i 条数据是什么？

        ↓

__getitem__(i)
```

就这么简单。

---

# 三、把 Dataset 想成一个仓库

教授画：

```text
StudentDataset

┌─────────────────────────┐
│ 0 → (x0, y0)            │
│ 1 → (x1, y1)            │
│ 2 → (x2, y2)            │
│ 3 → (x3, y3)            │
│ ...                     │
│ 9999 → (x9999, y9999)   │
└─────────────────────────┘
```

于是：

```python
len(dataset)
```

就是：

```text
仓库里有多少件货物？
```

而：

```python
dataset[5]
```

就是：

```text
把第 5 件货拿出来。
```

林默点头。

“那 Dataset 只是负责存取数据？”

“可以先这么理解。”

它不负责训练。

也不负责：

```text
backward
optimizer
Loss
```

它主要负责：

> **告诉 PyTorch 数据在哪里，以及怎样拿到某个样本。**

---

# 四、但是一条一条训练不行吗？

林默问：

“那我直接这样？”

```python
for x, y in dataset:

    pred = model(x)

    loss = loss_fn(pred, y)

    loss.backward()

    optimizer.step()
```

“理论上可以。”

“但通常不是最佳选择。”

教授在黑板上写：

```text
一次只训练 1 条数据
```

假设：

```text
100 万条数据
```

那一次完整遍历就要进行：

```text
100 万次参数更新
```

更重要的是 GPU 很擅长：

> **并行矩阵计算。**

你却每次只扔进去一个样本：

```text
GPU：

████████████████████

你：

■
```

大量计算能力可能利用不起来。

林默明白了。

“所以一次多喂一点？”

“对。”

于是第二个核心概念出现了：

# Batch

---

# 五、Batch：一车一车运

假设有：

```text
1000 条数据
```

我们规定：

```text
batch_size = 100
```

那么：

```text
Batch 1：
样本 1 ~ 100

Batch 2：
样本 101 ~ 200

Batch 3：
样本 201 ~ 300

...

Batch 10：
样本 901 ~ 1000
```

每一批：

> **100 个 Sample。**

这就是一个：

> **Batch——批次。**

教授画了一座仓库：

```text
                 Dataset
              1000 个样本
                   │
                   ▼
        ┌──────────────────┐
        │    DataLoader    │
        └──────────────────┘
                   │
        batch_size = 100
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
       Batch1   Batch2   Batch3 ...
        100      100      100
```

---

# 六、DataLoader 登场

PyTorch：

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=100
)
```

现在：

```python
for x, y in dataloader:
    print(x.shape)
```

每次拿到的不再是一条数据。

而是一批。

假设每个学生有：

```text
2 个特征
```

那么：

```text
x.shape
=
[100, 2]
```

因为：

```text
100 个学生
×
每个学生 2 个特征
```

标签：

```text
y.shape
=
[100]
```

林默突然发现：

“这和上一章的 CrossEntropy 对上了。”

教授点头。

上一章我们说：

```text
logits.shape

[batch_size, num_classes]
```

现在来源清楚了。

例如：

```text
batch_size = 100
num_classes = 3
```

模型：

```python
logits = model(x)
```

得到：

```text
x
[100, 2]

    ↓ model

logits
[100, 3]
```

target：

```text
[100]
```

然后：

```python
loss = loss_fn(logits, y)
```

一次计算这一批 100 个样本的 Loss。

---

# 七、为什么不把 100 万条一次全塞进去？

林默马上反问：

“既然一次越多 GPU 利用率越高——”

“那我直接：”

```text
batch_size = 1,000,000
```

“不就行了？”

教授指了指电脑。

“你有多少显存？”

林默：“……”

这是第一个问题：

> **显存有限。**

模型训练过程中，GPU 不只是保存输入。

还需要保存：

```text
模型参数
梯度
Optimizer 状态
中间激活值
输入数据
其他临时计算结果
```

所以 batch 越大：

```text
一次进入模型的数据越多
        ↓
中间结果通常越多
        ↓
显存占用增加
```

最后：

```text
CUDA out of memory
```

也就是以后经常见到的：

> **OOM。**

---

# 八、Batch 还有第二个作用

教授突然写了两个极端。

第一种：

```text
batch_size = 1
```

第二种：

```text
batch_size = 整个训练集
```

假设真实世界的 Loss 地形：

```text
Loss
│
│       /\        /\
│      /  \______/  \
│  ___/              \___
│_/                       \_
└──────────────────────────── 参数
```

如果每次只看一个样本：

```text
这个样本说：
往左！

下一个样本说：
往右！

再下一个：
往左上！
```

梯度会比较：

> **嘈杂。**

参数更新路线可能：

```text
↘
  ↙
    ↘
      ↓
    ↙
      ↘
        ↓
```

---

# 九、如果看一批样本呢？

例如：

```text
batch_size = 128
```

128 个样本共同贡献一个 batch 的 Loss。

你得到的梯度更像：

```text
128 个样本综合之后：

“整体来说，往这个方向比较好。”
```

所以更新相对稳定。

但又不像整个训练集那样，每更新一次都要计算全部数据。

这形成了一个折中：

```text
单样本
│
│ 梯度噪声大
│ 更新频繁
│
▼

Mini-Batch
│
│ 效率与稳定性的折中
│ GPU 友好
│
▼

整个 Dataset
│
│ 梯度更稳定
│ 单次计算昂贵
│ 内存压力大
```

现代深度学习最常见的是：

> **Mini-Batch Gradient Descent。**

平常大家直接说 Batch，很多时候其实指的就是 mini-batch。

---

# 十、Iteration 出现了

教授写：

```python
for x, y in dataloader:

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(pred, y)

    loss.backward()

    optimizer.step()
```

每执行一次：

```text
取一个 Batch
      ↓
forward
      ↓
loss
      ↓
backward
      ↓
step
```

就完成一次：

> **Iteration——迭代。**

也就是说：

```text
1 Iteration
=
训练 1 个 Batch
+
通常更新 1 次参数
```

林默在纸上写：

```text
10000 个样本

batch_size = 100

那么：

10000 / 100 = 100
```

“所以遍历完整个数据集，需要 100 个 iteration？”

“对。”

更准确地说，如果不能整除，还要处理最后一个不足 batch 的批次；默认情况下 DataLoader 可以保留它。

---

# 十一、Epoch 到底是什么？

教授终于写下这个天天出现的词：

# Epoch

定义很简单：

> **整个训练集被完整遍历一遍，叫一个 Epoch。**

例如：

```text
Dataset：

1000 个样本
```

设置：

```text
batch_size = 100
```

那么：

```text
1 Epoch
│
├── Batch 1 → Iteration 1
├── Batch 2 → Iteration 2
├── Batch 3 → Iteration 3
├── ...
└── Batch 10 → Iteration 10
```

所以：

```text
1 Epoch
=
10 Iterations
```

如果训练：

```text
20 Epoch
```

意味着大体上：

> 每条训练数据会被模型看到 20 次。

总 iteration：

\[
20\times10=200
\]

---

# 十二、终于看懂双层 for 循环

教授写出了真正经典的训练代码：

```python
for epoch in range(10):

    for x, y in dataloader:

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()
```

林默以前看这种代码，只知道：

```text
两个 for
```

现在完全不同了。

外层：

```python
for epoch in range(10):
```

意思：

> **整个训练集看 10 遍。**

内层：

```python
for x, y in dataloader:
```

意思：

> **这一遍里面，一批一批拿数据。**

所以：

```text
Epoch 1
│
├─ Batch 1 → 更新参数
├─ Batch 2 → 更新参数
├─ Batch 3 → 更新参数
└─ ...

Epoch 2
│
├─ Batch 1 → 更新参数
├─ Batch 2 → 更新参数
└─ ...

...

Epoch 10
```

---

# 十三、但这里有一个问题

教授拿出十张卡片：

```text
1 2 3 4 5 6 7 8 9 10
```

假设：

```text
batch_size = 2
```

第一次：

```text
[1,2]
[3,4]
[5,6]
[7,8]
[9,10]
```

第二个 Epoch：

```text
[1,2]
[3,4]
[5,6]
[7,8]
[9,10]
```

第三个 Epoch：

```text
[1,2]
[3,4]
...
```

永远一样。

教授问：

“有没有必要每一轮都让模型按照完全相同的顺序看数据？”

林默想了一下。

“可以打乱？”

于是：

```python
dataloader = DataLoader(
    dataset,
    batch_size=100,
    shuffle=True
)
```

---

# 十四、Shuffle：洗牌

`shuffle=True` 可以理解为：

> **每轮重新洗牌。**

例如：

第一轮：

```text
3 8 1 7 4 2 10 5 9 6
```

分 batch：

```text
[3,8]
[1,7]
[4,2]
[10,5]
[9,6]
```

下一轮重新洗：

```text
9 2 6 1 8 5 3 10 4 7
```

于是：

```text
[9,2]
[6,1]
[8,5]
[3,10]
[4,7]
```

这样每个 Batch 的样本组合不断变化。

对于许多训练任务，这有助于减少固定数据顺序带来的问题。

所以训练集常见：

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

---

# 十五、测试集也 Shuffle 吗？

林默顺手写：

```python
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)
```

教授看了一眼。

“通常没必要。”

因为测试/验证的时候：

```text
不训练
不更新参数
```

只是：

```text
拿数据
 ↓
预测
 ↓
统计准确率 / Loss
```

所以通常：

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)
```

当然，这不是说验证集“绝对禁止 shuffle”，而是通常没有训练集那种打乱顺序的需求。

---

# 十六、为什么还需要 `drop_last`？

假设：

```text
数据量 = 1000
batch_size = 128
```

算一下：

\[
1000=128\times7+104
\]

所以：

```text
Batch 1 → 128
Batch 2 → 128
...
Batch 7 → 128
Batch 8 → 104
```

最后一批不完整。

默认可以正常保留。

但某些场景下，我们希望每个 batch 都严格一样大，可以：

```python
DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True
)
```

那么最后：

```text
104 条
```

这一批就被丢掉。

林默点头。

“所以 `drop_last=True` 不是固定必须开的。”

“对。”

> 是否需要丢最后一个不完整 batch，要看具体模型和训练需求。

---

# 十七、DataLoader 还有一个真正工程化的作用

教授突然问：

“GPU 训练速度非常快，但数据在哪？”

“硬盘？”

“对。”

真实训练可能是：

```text
SSD
 ↓
读取图片
 ↓
解码
 ↓
预处理
 ↓
转 Tensor
 ↓
送入 GPU
 ↓
模型计算
```

假设 GPU 一批只需要：

```text
50ms
```

结果 CPU 读取和处理下一批需要：

```text
300ms
```

那 GPU 会变成：

```text
算完了。

……

数据呢？

……

还没来？

……
```

昂贵的 GPU 在等 CPU。

所以 DataLoader 还可以使用多个 worker：

```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)
```

粗略理解：

```text
Worker 1 ─┐
Worker 2 ─┤
Worker 3 ─┼──► 准备 Batch ──► 训练
Worker 4 ─┘
```

多个进程/worker 提前准备数据，尽量减少模型等数据的时间。

不过 `num_workers` 并不是：

> 越大越好。

它受：

```text
CPU 核心
内存
磁盘速度
操作系统
数据预处理复杂度
```

等因素影响，需要实际测试。

---

# 十八、现在自己造一个 Dataset

教授说：

“来，完整写一次。”

任务：

```text
输入两个数
预测 3 个类别
```

数据：

```python
import torch
from torch.utils.data import Dataset, DataLoader


x = torch.randn(1000, 2)

y = torch.randint(
    0,
    3,
    (1000,)
)
```

形状：

```text
x.shape
→ [1000, 2]

y.shape
→ [1000]
```

然后：

```python
class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

创建：

```python
dataset = MyDataset(x, y)
```

然后：

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

---

# 十九、看一眼 DataLoader 吐出来什么

教授写：

```python
for batch_x, batch_y in dataloader:

    print(batch_x.shape)
    print(batch_y.shape)

    break
```

大概率得到：

```text
torch.Size([32, 2])

torch.Size([32])
```

为什么？

```text
32
↓
batch_size

2
↓
每个样本的特征数
```

所以：

```text
batch_x
[32, 2]
```

而每个样本只有一个类别编号：

```text
batch_y
[32]
```

这时候模型：

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
```

输入：

```text
[32, 2]
```

经过：

```text
Linear(2,16)
```

得到：

```text
[32,16]
```

再：

```text
ReLU
```

形状不变：

```text
[32,16]
```

最后：

```text
Linear(16,3)
```

得到：

```text
[32,3]
```

于是：

```text
logits.shape = [32,3]
target.shape = [32]
```

正好交给：

```python
nn.CrossEntropyLoss()
```

---

# 二十、第一次完整训练

教授终于把前五章所有东西合并起来。

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =====================
# 1. 数据
# =====================

x = torch.randn(1000, 2)
y = torch.randint(0, 3, (1000,))


class MyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


dataset = MyDataset(x, y)


# =====================
# 2. DataLoader
# =====================

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# =====================
# 3. 模型
# =====================

model = nn.Sequential(

    nn.Linear(2, 16),

    nn.ReLU(),

    nn.Linear(16, 3)
)


# =====================
# 4. Loss
# =====================

loss_fn = nn.CrossEntropyLoss()


# =====================
# 5. Optimizer
# =====================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================
# 6. Train
# =====================

for epoch in range(10):

    for batch_x, batch_y in dataloader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = loss_fn(
            logits,
            batch_y
        )

        loss.backward()

        optimizer.step()

    print(
        "epoch:",
        epoch,
        "loss:",
        loss.item()
    )
```

林默盯着这段代码。

第一次，他没有觉得这是“一堆 PyTorch API”。

而是看到了一条完整的流水线。

---

# 二十一、整条流水线

```text
                Dataset
          ┌─────────────────┐
          │ Sample 1        │
          │ Sample 2        │
          │ Sample 3        │
          │ ...             │
          │ Sample 1000     │
          └─────────────────┘
                   │
                   ▼
               DataLoader
                   │
           batch_size = 32
                   │
                   ▼
                 Batch
             x: [32,2]
             y: [32]
                   │
                   ▼
                 Model
                   │
          Linear(2 → 16)
                   │
                 ReLU
                   │
          Linear(16 → 3)
                   │
                   ▼
                logits
                [32,3]
                   │
                   ▼
           CrossEntropyLoss
                   │
                   ▼
                  Loss
                   │
                   ▼
               backward
                   │
                   ▼
            Parameter.grad
                   │
                   ▼
           optimizer.step
                   │
                   ▼
              参数更新
```

然后：

```text
下一个 Batch
     ↓
再更新一次

下一个 Batch
     ↓
再更新一次

...

所有 Batch 处理完成
     ↓
1 Epoch 完成
     ↓
重新开始下一个 Epoch
```

---

# 二十二、Sample、Batch、Iteration、Epoch 终于不乱了

教授在黑板上写了一组数字：

```text
Dataset Size = 10,000

Batch Size = 100

Epoch = 20
```

问：

“一个 Batch 有几个 Sample？”

林默：

“100。”

“一个 Epoch 有多少 Batch？”

\[
10000/100=100
\]

“100 个。”

“一个 Epoch 大约多少 Iteration？”

“100。”

“训练 20 Epoch 呢？”

\[
20\times100=2000
\]

“2000 次 Iteration，也就是通常进行了约 2000 次参数更新。”

教授点头。

于是：

```text
Sample
↓
一条数据


Batch
↓
一批 Sample


Iteration
↓
处理一个 Batch
通常更新一次参数


Epoch
↓
完整遍历一次 Dataset
```

这四个概念，彻底分开。

---

# 二十三、但林默发现了一个新的问题

他盯着训练代码：

```python
for epoch in range(10):

    for batch_x, batch_y in dataloader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = loss_fn(logits, batch_y)

        loss.backward()

        optimizer.step()
```

突然问：

“教授。”

“这里所有东西都还在 CPU 上吧？”

教授点头。

林默指着电脑里的显卡。

“那我买这张显卡干什么？”

教授笑了。

“终于想起来它了。”

---

# 二十四、CPU 和 GPU 是两个世界

教授写：

```python
x = torch.tensor([1, 2, 3])
```

默认通常：

```text
CPU
```

如果：

```python
x = x.cuda()
```

或者更通用地：

```python
x = x.to("cuda")
```

Tensor 就会进入：

```text
GPU
```

但这时候有一个极其经典的错误。

假设：

```text
model → GPU

x → CPU
```

然后：

```python
model(x)
```

会怎么样？

林默想了想。

“报错？”

“对。”

因为模型参数和输入数据在不同设备。

你不能让：

```text
CPU 上的数据
```

直接和：

```text
GPU 上的权重
```

做同一次普通计算。

于是实际训练代码开始变成：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

for x, y in dataloader:

    x = x.to(device)
    y = y.to(device)

    ...
```

林默看着 `device`。

他意识到 PyTorch 世界的下一扇门已经打开。

---

夜里十二点。

陈教授在黑板上留下：

```text
CPU
 │
 │ .to(device)
 ▼
GPU

Tensor
Model
Parameter
Gradient
```

旁边还有几个林默经常见，却一直没真正理解的词：

```text
CUDA

VRAM

.to(device)

no_grad()

train()

eval()
```

教授关上门前说：

“下一章，我们终于让你的模型真正跑到 GPU 上。”

---

# 下一章

## 《跨越两界——GPU、CUDA、`.to(device)`、`train()` 与 `eval()`》

下一章会重点把几个**实际写 PyTorch 必须掌握**的问题连起来：

```text
CPU 和 GPU 到底差在哪？
        ↓
为什么深度学习特别适合 GPU？
        ↓
model.to(device) 到底搬了什么？
        ↓
为什么 x 和 model 必须在同一设备？
        ↓
显存里到底装了什么？
        ↓
torch.no_grad() 到底省了什么？
        ↓
model.train() 和 model.eval()
为什么不是“开始训练”和“开始预测”？
```

尤其最后一个非常容易理解错：**`model.eval()` 并不会自动停止梯度计算，`torch.no_grad()` 也不会自动把模型切换成 eval 模式。**

这两个东西解决的是完全不同的问题。


下一章。


# 《PyTorch：炼金术士的学徒》

## 第六章：跨越两界——GPU、CUDA、`.to(device)`、训练模式与推理模式

第六天。

林默来到实验室。

桌上放着一块显卡。

旁边贴着一张纸：

```text
GPU

24GB VRAM

CUDA Ready
```

林默拿起来看。

“所以这东西就是让 PyTorch 变快？”

教授点头。

“但如果你只记住‘GPU 快’，你还没有真正理解。”

“今天我们讲清楚——”

> **为什么神经网络喜欢 GPU，以及 PyTorch 如何管理 CPU 与 GPU 两个世界。**

---

# 一、先认识两个世界

教授画了一条线。

```text
             计算机

        ┌────────────┐
        │            │
        │    CPU     │
        │            │
        └────────────┘


        ┌────────────┐
        │            │
        │    GPU     │
        │            │
        └────────────┘
```

林默问：

“CPU 和 GPU 不都是处理器吗？”

“是。”

“那为什么深度学习不用 CPU？”

教授没有马上回答。

而是写了一个任务：

计算：

\[
C=A\times B
\]

其中：

```text
A:
10000 × 10000

B:
10000 × 10000
```

也就是矩阵乘法。

---

# 二、神经网络为什么喜欢矩阵？

上一章我们知道：

一个 Linear：

\[
y=Wx+b
\]

如果：

```text
batch_size = 32
```

输入：

\[
X
\]

可能是：

\[
32\times100
\]

权重：

\[
W
\]

可能：

\[
256\times100
\]

计算：

\[
XW^T
\]

实际上就是：

> **大量矩阵乘法。**

而深度学习训练中：

```text
Linear
↓
Attention
↓
MLP
↓
Embedding
↓
Conv
```

大量都是：

```text
矩阵
×
矩阵
```

---

# 三、CPU 像一个聪明的工程师

教授画：

```text
CPU:

┌──────────────┐
│ 核心1        │
│ 核心2        │
│ 核心3        │
│ 核心4        │
│ ...          │
└──────────────┘
```

CPU 的特点：

```text
少量强大的核心
+
复杂控制能力
```

比如：

运行操作系统：

```text
Windows
Linux
数据库
Java程序
浏览器
```

非常适合。

---

# 四、GPU 像一支庞大的工人军队

教授画：

```text
GPU:

○ ○ ○ ○ ○ ○ ○ ○

○ ○ ○ ○ ○ ○ ○ ○

○ ○ ○ ○ ○ ○ ○ ○

○ ○ ○ ○ ○ ○ ○ ○
```

GPU 有大量计算单元。

单个能力可能没有 CPU 核心强。

但是：

如果任务是：

```text
重复做一百万次相同计算
```

GPU 非常适合。

比如：

```text
工人1:
计算第1个矩阵元素

工人2:
计算第2个矩阵元素

工人3:
计算第3个矩阵元素

...

一起完成
```

所以：

> **GPU 不是任何程序都快。**

它特别擅长：

> 大规模并行计算。

而神经网络：

> 恰好大量依赖这种计算。

---

# 五、CUDA 是什么？

林默看着显卡：

“那 CUDA 又是什么？”

教授写：

```text
CUDA
```

然后解释：

“简单理解。”

> CUDA 是 NVIDIA 提供的一套让程序使用 GPU 计算能力的平台。

以前：

GPU 主要负责：

```text
游戏画面
图形渲染
```

后来 NVIDIA 提供 CUDA：

让开发者可以：

```text
普通程序
    ↓
调用 GPU
    ↓
执行计算
```

PyTorch 就利用 CUDA。

所以：

```python
x.to("cuda")
```

意思：

> 把 Tensor 搬到支持 CUDA 的 GPU 上。

---

# 六、PyTorch 的设备 Device

教授打开 Python：

```python
import torch
```

输入：

```python
device = torch.device("cuda")
```

表示：

```text
我要 GPU
```

但是更常见：

```python
device = torch.device(
    "cuda" 
    if torch.cuda.is_available()
    else "cpu"
)
```

意思：

如果有 GPU：

```text
cuda
```

否则：

```text
cpu
```

---

# 七、Tensor 为什么需要搬设备？

林默问：

“为什么要搬？”

教授创建：

```python
x = torch.tensor([1,2,3])
```

查看：

```python
print(x.device)
```

输出：

```text
cpu
```

因为默认：

```text
Tensor → CPU
```

现在：

```python
x = x.to("cuda")
```

再看：

```text
cuda:0
```

意思：

```text
第0号GPU
```

---

# 八、模型也需要搬

林默想：

“那输入搬过去不就行？”

教授摇头。

创建模型：

```python
model = nn.Linear(10,3)
```

查看参数：

```python
next(model.parameters()).device
```

输出：

```text
cpu
```

因为：

模型里面：

```text
weight
bias
```

也是 Tensor。

准确说：

```text
Parameter
```

默认：

```text
CPU
```

所以：

```python
model.to("cuda")
```

发生什么？

---

# 九、`.to(device)` 到底搬了什么？

教授画：

原来：

```text
CPU

model
│
├── weight
├── bias
```

执行：

```python
model.to("cuda")
```

之后：

```text
GPU

model
│
├── weight
├── bias
```

也就是说：

> 模型参数被移动到了 GPU。

不仅如此：

如果模型内部有：

```python
self.layer1 = nn.Linear(...)
self.layer2 = nn.Linear(...)
```

所有注册的 Parameter：

都会一起移动。

因为：

```text
nn.Module
    ↓
管理 Parameter
    ↓
统一迁移
```

---

# 十、为什么 Tensor 和 Model 必须在同一个设备？

教授写：

```python
model = model.to("cuda")

x = torch.randn(1,10)
```

现在：

```text
model:

GPU


x:

CPU
```

然后：

```python
model(x)
```

会发生什么？

错误。

类似：

```text
Expected all tensors to be on the same device
```

林默问：

“为什么不能自动帮我搬？”

教授说：

“因为自动搬迁会隐藏大量问题。”

例如：

你以为：

```text
CPU → GPU
```

只是一下。

实际上：

```text
PCIe传输
↓
内存复制
↓
同步
```

可能很慢。

PyTorch 要求你明确：

```python
x = x.to(device)
```

让开发者知道：

> 我要发生一次设备迁移。

---

# 十一、完整 GPU 训练模板

教授写：

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = MyModel()

model = model.to(device)


for x, y in dataloader:

    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    pred = model(x)

    loss = loss_fn(
        pred,
        y
    )

    loss.backward()

    optimizer.step()
```

林默看着：

“其实核心变化只有三个？”

教授点头：

```text
1.

model.to(device)


2.

x.to(device)


3.

y.to(device)
```

训练逻辑完全没变。

---

# 十二、GPU 显存里面到底放什么？

林默问：

“显存为什么这么容易爆？”

教授画：

GPU 显存：

```text
┌──────────────────┐
│ 模型参数          │
│                  │
│ weight           │
│ bias             │
│                  │
├──────────────────┤
│ 梯度              │
│                  │
│ grad             │
│                  │
├──────────────────┤
│ optimizer状态     │
│                  │
├──────────────────┤
│ 输入数据          │
│                  │
├──────────────────┤
│ 中间激活          │
│                  │
└──────────────────┘
```

---

## 1. 模型参数

例如：

GPT 有：

```text
7B 参数
```

每个参数：

float32：

4 byte

那么：

\[
7\times10^9\times4
\]

约：

28GB

仅参数。

---

## 2. Gradient

训练需要：

```python
loss.backward()
```

所以每个参数需要保存：

```text
weight
+
gradient
```

额外占空间。

---

## 3. Optimizer 状态

例如 Adam：

除了：

```text
参数
```

还保存：

```text
一阶动量

二阶动量
```

所以：

Adam 训练大型模型时：

显存压力很大。

---

## 4. Activation

这是很多人忽略的。

前向：

```text
x
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Loss
```

反向传播需要知道：

```text
中间结果
```

所以很多 activation 要保存。

Batch 越大：

```text
输入更多
 ↓
中间结果更多
 ↓
显存增加
```

这也是为什么：

```text
batch_size
```

经常卡显存。

---

# 十三、`torch.no_grad()` 到底省什么？

上一章我们见过：

```python
with torch.no_grad():

    pred = model(x)
```

很多人理解：

“关闭梯度。”

不够准确。

教授写：

训练：

```text
forward

↓

保存计算图

↓

backward需要
```

例如：

```python
x
 ↓
Linear
 ↓
ReLU
 ↓
Loss
```

PyTorch 需要记录：

```text
谁依赖谁
```

---

但是推理：

```text
输入
 ↓
模型
 ↓
输出
```

你不会：

```python
loss.backward()
```

所以：

这些东西没有意义：

```text
计算图
梯度记录
activation缓存
```

于是：

```python
with torch.no_grad():
```

告诉 PyTorch：

> 不需要记录梯度相关信息。

结果：

```text
更少显存

更快推理
```

---

# 十四、但是 `no_grad()` ≠ `eval()`

这是今天最容易混淆的地方。

林默写：

```python
model.eval()
```

教授问：

“什么意思？”

林默：

“关闭梯度？”

“不对。”

---

## `eval()` 是切换模型行为

有些层：

训练和推理行为不同。

最典型：

```text
Dropout

BatchNorm
```

---

# 十五、Dropout

教授画：

训练：

```text
神经元：

○ ○ ○ ○ ○
```

Dropout：

随机关闭：

```text
○ × ○ × ○
```

目的：

防止模型过度依赖某些神经元。

训练时：

```python
model.train()
```

Dropout 开启。

---

推理：

你希望：

```text
所有神经元正常工作
```

所以：

```python
model.eval()
```

关闭 Dropout。

---

# 十六、BatchNorm

BatchNorm 更复杂。

简单理解：

训练时：

它根据当前 batch：

计算：

```text
均值
方差
```

然后归一化。

但是推理时：

没有不断变化的 batch。

所以：

```text
训练：
使用当前batch统计


推理：
使用训练期间保存的统计量
```

所以：

```python
model.eval()
```

会改变 BatchNorm 行为。

---

# 十七、训练和推理正确写法

训练：

```python
model.train()


for x,y in dataloader:

    pred = model(x)

    loss = loss_fn(
        pred,
        y
    )

    loss.backward()

    optimizer.step()
```

---

推理：

```python
model.eval()


with torch.no_grad():

    pred = model(x)
```

注意：

两个东西分别负责：

```text
model.train()
model.eval()

↓

改变模型运行模式


torch.no_grad()

↓

是否记录梯度
```

---

# 十八、一个经典错误

错误：

```python
model.eval()

pred = model(x)

loss = loss_fn(pred,y)

loss.backward()
```

为什么错？

因为：

```python
eval()
```

只是：

```text
Dropout关闭
BatchNorm切换
```

它没有禁止：

```text
梯度计算
```

如果你不需要训练：

应该：

```python
with torch.no_grad():
```

---

另一个错误：

```python
with torch.no_grad():

    model.train()
```

这也是奇怪组合。

因为：

你告诉 PyTorch：

```text
我要训练模式
```

但是：

```text
不要记录梯度
```

通常没有意义。

---

# 十九、完整训练/验证结构

真实项目通常：

```python
for epoch in range(10):

    # ===== Train =====

    model.train()

    for x,y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(
            pred,
            y
        )

        loss.backward()

        optimizer.step()



    # ===== Validation =====

    model.eval()

    with torch.no_grad():

        for x,y in val_loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = loss_fn(
                pred,
                y
            )
```

林默看着代码。

终于明白：

```text
训练阶段：

学习
更新参数


验证阶段：

观察
不更新参数
```

---

# 二十、为什么大模型训练特别依赖 GPU？

教授最后画了一张图：

一个 Transformer：

```text
输入 Token

↓

Embedding

↓

Attention

↓

MLP

↓

Attention

↓

MLP

↓

...

↓

输出概率
```

里面大量：

```text
矩阵乘法

+
向量运算
```

例如 Attention：

\[
Attention(Q,K,V)
=
softmax(\frac{QK^T}{\sqrt d})V
\]

核心：

```text
矩阵 × 矩阵
```

GPU：

```text
大量并行计算
```

所以：

训练：

```text
CPU:
几个月

GPU:
几天/几周
```

甚至大型模型：

需要：

```text
数千张 GPU
```

---

# 二十一、今天的 PyTorch 世界地图

六章之后：

林默终于能画出完整训练系统：

```text
                 Dataset

                    ↓

               DataLoader

                    ↓

                Batch数据

                    ↓

             Tensor.to(cuda)

                    ↓

                GPU

                    ↓

              nn.Module

                    ↓

               forward()

                    ↓

                 logits

                    ↓

                 Loss

                    ↓

             backward()

                    ↓

              gradients

                    ↓

           optimizer.step()

                    ↓

             参数更新
```

旁边还有：

```text
训练:

model.train()


推理:

model.eval()
+
torch.no_grad()
```

---

教授收起显卡。

“现在你已经知道 PyTorch 怎么训练一个模型。”

林默点头。

“但是……”

他看向代码：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

“这里还有一个问题。”

“为什么是 Adam？”

“为什么不是 SGD？”

“学习率为什么是 0.001？”

“为什么有时候训练不收敛？”

教授笑了。

“很好。”

“你终于开始问优化的问题了。”

黑板上出现：

```text
SGD

Momentum

Adam

Learning Rate

Scheduler
```

---

# 下一章

## 《寻找最低山谷——优化器 Optimizer、SGD、Momentum、Adam 与学习率》

下一章会解释：

```text
梯度下降到底怎么走？
        ↓
为什么 SGD 像盲人下山？
        ↓
Momentum 为什么像带惯性的球？
        ↓
Adam 为什么成为深度学习默认选择？
        ↓
Learning Rate 为什么太大/太小都会失败？
        ↓
Scheduler 为什么训练后期要降低学习率？
```

这章之后，你会真正理解：

```python
optimizer.step()
```

这一行代码背后到底发生了什么。

下一章


# 《PyTorch：炼金术士的学徒》

## 第七章：寻找最低山谷——Optimizer、SGD、Momentum、Adam 与学习率

第七天。

林默来到实验室。

今天桌上没有代码。

只有一张山谷地图。

上面写着：

```text
Loss Landscape
损失地形
```

教授问：

“昨天我们说训练模型是什么？”

林默回答：

“调整参数，让 Loss 降低。”

教授点头。

“那么今天的问题是——”

“知道山在哪里。”

教授指着山谷：

```text
             Loss

               ▲
               │
        /\     │
       /  \    │
      /    \___│
_____/          \______

               参数
```

“你站在山上。”

“怎么走到最低点？”

---

# 一、训练模型其实就是下山

假设模型只有一个参数：

\[
w
\]

Loss：

\[
L(w)
\]

图像：

```text
Loss

 ^
 |
 |       *
 |      / \
 |     /   \
 |    /     \
 |___/       \____

             w
```

现在：

```text
w = 10
```

但是最低点：

```text
w = 2
```

怎么办？

需要知道：

> 当前方向哪个地方下降最快。

这就是：

# Gradient（梯度）

---

# 二、梯度是什么？

上一章我们讲：

\[
\frac{\partial L}{\partial w}
\]

它表示：

> 参数变化一点，Loss 会怎么变化。

例如：

如果：

\[
\frac{\partial L}{\partial w}=5
\]

说明：

当 w 增大：

Loss 增大。

所以：

应该减少 w。

于是：

\[
w=w-\eta \frac{\partial L}{\partial w}
\]

其中：

\[
\eta
\]

就是：

# Learning Rate（学习率）

---

# 三、学习率就是步子大小

教授画两个人下山。

第一个：

```text
小步：

↓
↓
↓
↓
```

第二个：

```text
大步：

↘

        ↙

              ↘
```

林默马上明白。

“学习率太大，会不会跨过最低点？”

教授点头。

---

## 情况一：学习率太大

例如：

```text
lr = 10
```

真正最低点：

```text
      ↓
     ★
```

但是一步：

```text
当前位置

──────────────→

直接跳过去
```

结果：

```text
左边
  ↓
右边
  ↓
左边
  ↓
右边
```

Loss：

```text
10
50
20
80
...
```

模型不收敛。

---

## 情况二：学习率太小

例如：

```text
lr=0.0000001
```

每次：

```text
↓
一点点

↓
一点点
```

可能训练几天：

还没走到山谷。

---

所以：

学习率是训练中最重要的超参数之一。

---

# 四、最原始的优化器：SGD

教授写：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

SGD：

> Stochastic Gradient Descent

随机梯度下降。

最简单公式：

\[
w_{t+1}=w_t-\eta g_t
\]

其中：

```text
w_t

当前参数


g_t

当前梯度
```

翻译成人话：

```text
现在在哪里？

↓

看一下坡度

↓

往下降方向走一步
```

---

# 五、为什么叫随机？

林默问：

“哪里随机？”

教授回答：

“因为我们通常不是使用整个 Dataset 算梯度。”

回忆上一章：

```text
100万数据

↓

batch_size=64
```

一次：

```text
64条数据
```

计算：

```python
loss.backward()
```

得到：

```text
batch梯度
```

而不是：

```text
100万数据整体梯度
```

所以叫：

> Stochastic。

更严格说，现代深度学习常用的是 mini-batch SGD。

---

# 六、SGD 的问题：像醉汉下山

教授画：

```text
最低点

        ★


路径：

   ↘
      ↙
        ↘
          ↙
             ★
```

为什么？

因为：

每个 Batch 不一样。

Batch 1：

```text
方向：
往左
```

Batch 2：

```text
方向：
往右
```

Batch 3：

```text
方向：
往左下
```

梯度有噪声。

所以 SGD：

```text
效率高

但是摇摇晃晃
```

林默问：

“有没有办法让它有一点惯性？”

教授笑了。

“这就是 Momentum。”

---

# 七、Momentum：给小球加惯性

教授画一个球：

```text
      ○
     /
    /
   /
  ★
```

普通 SGD：

每一步：

看当前坡度。

Momentum：

不仅看当前坡度。

还记住：

> 我之前往哪里运动。

---

数学：

普通 SGD：

\[
v_t=g_t
\]

Momentum：

\[
v_t=\beta v_{t-1}+g_t
\]

然后：

\[
w=w-\eta v_t
\]


这里：

```text
v

速度
```

---

# 八、为什么 Momentum 有用？

假设：

连续几个梯度：

```text
↓
↓
↓
```

Momentum：

速度不断累积：

```text
↓↓↓↓
```

所以：

快速通过平坦区域。


如果梯度左右摇摆：

```text
←
→
←
→
```

过去的速度会抵消：

减少震荡。

所以：

Momentum：

```text
加速正确方向

减少随机震动
```

---

# 九、Adam 登场

林默看代码：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

“那 Adam 又是什么？”

教授写：

# Adam

全称：

> Adaptive Moment Estimation

简单理解：

> Momentum + 自动调整每个参数学习率。

---

# 十、为什么需要不同学习率？

假设模型：

有两个参数：

```text
w1

w2
```

训练过程中：

参数1：

梯度：

```text
0.001
0.002
0.003
```

变化很慢。

参数2：

梯度：

```text
10
20
30
```

变化巨大。

如果统一：

```python
lr=0.001
```

可能：

参数1：

```text
几乎不动
```

参数2：

```text
跳来跳去
```

所以 Adam 想：

“每个参数应该有自己的步长。”

---

# 十一、Adam 的核心思想

Adam 保存两个东西。

## 1. 一阶矩

类似 Momentum：

记录：

> 梯度平均方向

叫：

\[
m_t
\]

可以理解：

```text
过去梯度趋势
```

---

## 2. 二阶矩

记录：

> 梯度大小变化

叫：

\[
v_t
\]

可以理解：

```text
这个参数最近波动大不大
```

---

然后：

如果某个参数：

梯度一直很大：

Adam：

```text
降低步子
```

如果某个参数：

梯度很小：

Adam：

```text
适当放大
```

所以：

> Adam 会给不同参数不同有效学习率。

---

# 十二、SGD vs Momentum vs Adam

教授画表：

|优化器|特点|
|-|-|
|SGD|简单，根据当前梯度走|
|Momentum|增加惯性，减少震荡|
|Adam|自适应学习率，训练方便|

---

实际经验：

很多深度学习任务：

```text
Adam / AdamW

↓

默认选择
```

尤其：

```text
Transformer
LLM
RAG模型微调
```

大量使用：

```text
AdamW
```

---

# 十三、AdamW 为什么多一个 W？

林默问：

“AdamW 和 Adam 差在哪？”

教授写：

```text
Weight Decay
```

中文：

> 权重衰减。

---

先说为什么需要它。

假设模型参数：

```text
w=100
```

模型可能：

过度依赖这个参数。

导致：

> 过拟合。

希望：

参数不要太大。

于是加入：

\[
Loss'=Loss+\lambda ||w||^2
\]


意思：

除了让预测正确。

还要求：

> 参数不要乱膨胀。

---

# 十四、AdamW 的改进

Adam：

把 weight decay 混进梯度计算。

AdamW：

把：

```text
参数更新

和

权重衰减
```

分开。

实际效果通常更好。

所以：

现在训练 Transformer：

经常看到：

```python
optimizer = AdamW(...)
```

---

# 十五、Learning Rate Scheduler

林默问：

“那学习率固定不行吗？”

教授摇头。

训练不同阶段：

需求不同。

---

刚开始：

模型很乱：

```text
猫不像猫
狗不像狗
```

希望：

快速调整。

需要：

```text
大一点 lr
```

---

后期：

已经接近答案：

```text
只差一点
```

希望：

精细调整。

需要：

```text
小一点 lr
```

于是：

学习率变化：

```text
lr

0.001
 |
 |
 |
0.0005
 |
 |
 |
0.0001
```

这就是：

> Learning Rate Scheduler

---

# 十六、常见 Scheduler

## StepLR

每隔几个 Epoch 降低：

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)
```

意思：

每 10 个 epoch：

学习率 ×0.1。

---

## Cosine Annealing

余弦下降：

```text
lr

高
 \
  \
   \
    \
     低
```

大模型训练非常常见。

---

## Warmup

Transformer 常见。

刚开始：

不要直接大 lr。

例如：

```text
epoch

0:
0.00001

10:
0.001
```

为什么？

因为模型刚开始参数随机。

太大的更新可能破坏训练。

所以：

先热身。

---

# 十七、一个真实训练配置

教授写：

```python
model = MyModel()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)
```

训练：

```python
for epoch in range(100):

    for x,y in loader:

        optimizer.zero_grad()

        pred=model(x)

        loss=loss_fn(pred,y)

        loss.backward()

        optimizer.step()


    scheduler.step()
```

---

# 十八、为什么 scheduler.step() 在后面？

林默注意：

```python
optimizer.step()

scheduler.step()
```

顺序。

教授说：

因为：

一次训练：

```text
计算梯度

↓

更新参数

↓

调整下一轮学习率
```

通常：

参数更新使用当前 lr。

然后：

准备下一轮。

---

# 十九、梯度爆炸与梯度裁剪

教授突然写：

```text
Gradient Explosion
```

林默皱眉：

“不是刚说梯度消失？”

“深度学习两个极端都有。”

---

梯度太大：

```text
gradient:

1000000
```

更新：

\[
w=w-\eta g
\]

可能：

参数直接飞走。

Loss：

```text
10
100
10000
∞
```

---

解决：

Gradient Clipping。

PyTorch：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

意思：

如果梯度太大：

限制一下。

---

训练：

```python
loss.backward()


torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0
)


optimizer.step()
```

---

# 二十、现在重新理解 optimizer.step()

第一章：

我们写：

```python
w -= lr*w.grad
```

现在：

真实世界：

```python
optimizer.step()
```

内部可能做：

AdamW:

```text
读取 grad

↓

更新一阶矩

↓

更新二阶矩

↓

计算自适应学习率

↓

weight decay

↓

修改参数
```

所以：

它远比简单减法复杂。

---

# 二十一、整个训练过程现在完整了

教授画最后一张图：

```text
Dataset

↓

DataLoader

↓

Batch

↓

Forward

↓

Prediction

↓

Loss

↓

Backward

↓

Gradient

↓

Optimizer

    |
    |
    ├── SGD
    |
    ├── Momentum
    |
    └── AdamW

↓

Update Parameters

↓

下一轮
```

---

# 二十二、林默终于理解了一个面试问题

面试官问：

> “为什么 Adam 比 SGD 常用？”

以前他可能回答：

“Adam 更先进。”

现在他会回答：

“Adam 在 SGD 的基础上引入了一阶动量和二阶矩估计，根据历史梯度信息调整更新方向和有效学习率，使训练过程通常更稳定，对不同参数尺度差异更友好。”

---

教授收起笔记。

“现在你已经掌握了训练一个普通神经网络需要的核心组件。”

林默看着：

```text
Tensor
Autograd
Module
Loss
Dataset
DataLoader
GPU
Optimizer
```

突然问：

“那如果数据不是图片，也不是表格。”

“比如……”

“文本呢？”

教授停了一下。

“很好。”

“因为下一步，你会进入大模型真正的入口。”

黑板上出现：

```text
文字
 ↓
Token
 ↓
Embedding
 ↓
向量
 ↓
神经网络
```

---

# 下一章

## 《文字如何进入神经网络——Token、Embedding、词表与 NLP 的第一步》

下一章会解释：

```text
为什么计算机不能直接理解文字？
        ↓
Token 到底是什么？
        ↓
Tokenizer 做了什么？
        ↓
Word Embedding 为什么能表示语义？
        ↓
为什么“国王-男人+女人≈女王”这种现象会出现？
        ↓
Embedding 和 RAG 向量检索有什么关系？
```

这一章开始，就会正式连接到你之前做的 **RAG、Embedding、向量数据库、LLM 应用开发**。


下一章


# 《PyTorch：炼金术士的学徒》

## 第八章：文字如何进入神经网络——Token、Embedding 与 NLP 的第一扇门

第八天。

林默刚进实验室，就看到黑板上写着一句话：

> 今天天气不错。

教授问：

“把这句话喂给神经网络。”

林默愣了一下。

“直接这样？”

```python
model("今天天气不错")
```

教授摇头。

“神经网络不会读中文。”

“它只认数字。”

于是，今天真正的问题出现了：

> **文字，怎么变成数字？**

---

# 一、神经网络只认识数值

前面我们训练模型时，输入都是：

```python
x = torch.tensor([
    [1.2, 3.4],
    [2.1, 5.7]
])
```

本质都是数字。

可文本是：

```text
我喜欢 PyTorch
```

神经网络不能直接对字符做矩阵乘法：

\[
W \times “我”
\]

没有意义。

所以必须先做转换：

```text
文字
 ↓
数字
 ↓
Tensor
 ↓
神经网络
```

教授在黑板上写下第一个词：

# Token

---

# 二、Token 到底是什么

林默问：

“Token 就是一个字吗？”

“有时候是，有时候不是。”

Token 可以是：

```text
一个字
一个词
一个词的一部分
标点
特殊符号
```

比如一句英文：

```text
I love PyTorch
```

一种最简单的切法：

```text
["I", "love", "PyTorch"]
```

三个 Token。

中文：

```text
我喜欢PyTorch
```

可能切成：

```text
["我", "喜欢", "PyTorch"]
```

也可能：

```text
["我", "喜", "欢", "Py", "Torch"]
```

不同 tokenizer 规则不同。

---

# 三、为什么不能直接“一个词一个 Token”

林默说：

“那按照词切不就行了吗？”

教授写：

```text
北京
上海
程序员
神经网络
大模型
反向传播
```

看起来没问题。

但语言中的词太多了。

比如：

```text
ChatGPT
ChatGPT-6
PyTorch
PyTorch2
foobar123
某个新出现的名字
```

如果每个完整词都必须提前放进词表，那么：

> 新词怎么办？

词表会无限膨胀。

所以现代大模型常用的是：

> **Subword Tokenization——子词切分。**

也就是：

```text
unbelievable
```

可能拆成：

```text
un
believ
able
```

这样模型不需要记住世界上所有完整单词。

---

# 四、Tokenizer 的两个核心工作

教授写：

```text
文本
 ↓
Tokenizer
 ↓
Token
 ↓
Token ID
```

Tokenizer 主要干两件事。

第一：

> 把文本切成 Token。

第二：

> 给每个 Token 一个整数编号。

例如词表：

```text
0  <pad>
1  <unk>
2  我
3  喜欢
4  PyTorch
5  Java
```

那么：

```text
我 喜欢 PyTorch
```

就变成：

```text
[2, 3, 4]
```

于是 PyTorch 终于能处理：

```python
token_ids = torch.tensor([2, 3, 4])
```

林默点头。

“所以 Token ID 就是 Token 的编号。”

“对。”

但教授马上补充：

> **Token ID 本身没有语义。**

---

# 五、为什么 Token ID 不能直接输入神经网络

假设：

```text
猫 = 10
狗 = 11
飞机 = 100
```

如果直接把 ID 当数值：

```text
猫 = 10
狗 = 11
飞机 = 100
```

模型可能误以为：

```text
狗比猫大 1
飞机比猫大 90
```

但这个数字差没有语义。

Token ID 只是数据库主键一样的东西。

就像：

```text
user_id = 10001
user_id = 10002
```

不能说明第二个人比第一个人大一点。

所以需要下一步。

---

# 六、Embedding 登场

教授写：

```python
embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=128
)
```

林默看着这个 API。

“这是什么？”

教授说：

“它可以理解成一张巨大的向量表。”

假设词表只有 5 个 Token，每个 Token 用 3 维向量表示：

```text
Token 0 → [ 0.2, -0.1,  0.7]
Token 1 → [-0.4,  0.9,  0.3]
Token 2 → [ 0.8,  0.1, -0.2]
Token 3 → [ 0.5, -0.7,  0.6]
Token 4 → [-0.2,  0.4,  0.9]
```

这张表就是：

\[
Embedding\ Matrix
\]

形状：

```text
[vocab_size, embedding_dim]
```

---

# 七、`nn.Embedding` 本质上就是查表

假设：

```python
embedding = nn.Embedding(10000, 128)
```

内部参数形状：

```text
[10000, 128]
```

也就是说：

```text
10000 个 Token
每个 Token 128 维
```

输入：

```python
token_ids = torch.tensor([2, 5, 9])
```

执行：

```python
x = embedding(token_ids)
```

得到：

```text
shape = [3, 128]
```

本质上就是：

```text
取第2行
取第5行
取第9行
```

所以 `Embedding` 可以粗略理解成：

> **用 Token ID 去查对应向量。**

---

# 八、那这些向量是谁定的？

林默问：

“难道人工规定？”

教授摇头。

“最重要的地方来了。”

这些 Embedding：

> **也是模型参数。**

也就是：

```python
embedding.weight
```

本身就是：

```text
Parameter
```

训练开始时可能随机：

```text
猫 → [0.12, -0.54, 0.33, ...]
狗 → [-0.72, 0.19, 0.05, ...]
```

然后通过：

```text
Loss
 ↓
backward
 ↓
optimizer.step()
```

不断调整。

所以：

> **词向量不是人写进去的，而是训练出来的。**

---

# 九、语义为什么会出现在向量里

教授写两句话：

```text
我养了一只猫。

我养了一只狗。
```

在大量语料中：

```text
猫
狗
```

经常出现在相似上下文。

比如：

```text
喂__
抱__
宠物__
可爱的__
```

那么为了完成语言预测任务，模型会逐渐学到：

> 猫和狗在很多语境中有相似功能。

于是它们的向量可能越来越接近。

比如：

```text
猫 → [0.7, 0.3, -0.2, ...]
狗 → [0.68, 0.29, -0.18, ...]
```

而：

```text
数据库
```

可能离得很远。

这就是：

> **语义进入向量空间。**

---

# 十、向量距离开始有意义了

假设二维世界：

```text
          猫 ●
              狗 ●


数据库 ●


                         飞机 ●
```

猫和狗距离近：

```text
相似
```

数据库较远：

```text
不太相似
```

于是可以用：

```text
Cosine Similarity
```

等指标衡量语义相似程度。

林默突然反应过来。

“这不就是我做 RAG 的向量检索？”

教授点头。

---

# 十一、RAG 的 Embedding 和 LLM 的 Token Embedding 是一回事吗？

这是一个非常容易混淆的问题。

教授写：

```text
LLM 内部 Token Embedding
```

和：

```text
RAG 文本 Embedding
```

它们都叫 Embedding。

但用途不同。

---

## LLM 的 Token Embedding

输入：

```text
Token ID
```

输出：

```text
一个 Token 的向量
```

比如：

```text
[batch, seq_len]
```

经过：

```python
nn.Embedding(...)
```

得到：

```text
[batch, seq_len, hidden_dim]
```

它是：

> Transformer 内部表示的起点。

---

## RAG Embedding

输入通常是：

```text
一句话
一个段落
一个 chunk
```

输出：

```text
一个整体语义向量
```

例如：

```text
“Redis 为什么快？”
 ↓
[0.123, -0.312, ..., 0.88]
```

然后存进：

```text
pgvector
Milvus
FAISS
```

做相似度检索。

所以：

```text
Token Embedding
→ token 级


RAG Embedding
→ 文本片段级
```

虽然背后可能都用神经网络产生向量，但不能简单当成同一个东西。

---

# 十二、Embedding 为什么不是 One-Hot

教授又画出旧时代的方法。

假设词表：

```text
猫
狗
鸟
车
```

One-Hot：

```text
猫 = [1,0,0,0]
狗 = [0,1,0,0]
鸟 = [0,0,1,0]
车 = [0,0,0,1]
```

问题来了。

猫和狗：

距离？

猫和车：

距离？

从这种表示本身看：

> 都一样远。

One-Hot 不携带语义相似性。

而 Embedding：

```text
猫 = [0.81, 0.23, -0.11]
狗 = [0.79, 0.25, -0.09]
车 = [-0.42, 0.61, 0.77]
```

猫狗就可以自然靠近。

---

# 十三、Embedding 其实是降维

假设词表：

```text
50000
```

One-Hot 每个词：

```text
50000 维
```

但 Embedding 可以：

```text
768 维
```

所以：

```text
50000维稀疏表示
 ↓
768维稠密表示
```

而且这 768 维是：

> **可学习的分布式表示。**

这也是深度学习非常关键的思想。

---

# 十四、一个小型 PyTorch Embedding 实验

教授写：

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=10,
    embedding_dim=4
)

tokens = torch.tensor([
    2, 5, 7
])

x = embedding(tokens)

print(x.shape)
```

输出：

```text
torch.Size([3, 4])
```

为什么？

```text
3 个 Token
×
每个 Token 4 维
```

所以：

```text
[3,4]
```

---

# 十五、如果有 Batch 呢？

输入：

```python
tokens = torch.tensor([
    [2, 5, 7],
    [1, 3, 9]
])
```

shape：

```text
[2,3]
```

意思：

```text
2 个样本
每个样本 3 个 Token
```

经过：

```python
x = embedding(tokens)
```

得到：

```text
[2,3,4]
```

也就是：

```text
batch_size = 2
seq_len = 3
embedding_dim = 4
```

这组 shape 极其重要。

因为进入 Transformer 后，你会天天看到：

```text
[batch_size, seq_len, hidden_size]
```

---

# 十六、序列长度 Seq Len 出现了

一句话：

```text
我 喜欢 PyTorch
```

三个 Token：

```text
seq_len = 3
```

另一句话：

```text
我 非常 喜欢 使用 PyTorch 训练 模型
```

可能 7 个 Token：

```text
seq_len = 7
```

但是一个 Batch 需要组织成 Tensor：

```text
[batch_size, seq_len]
```

问题来了。

不同句子长度不一样怎么办？

教授写：

# Padding

---

# 十七、Padding：把队伍补齐

句子 A：

```text
我 喜欢 PyTorch
```

3 Token。

句子 B：

```text
我 学习
```

2 Token。

如果词表定义：

```text
<PAD> = 0
```

那么：

```text
A：
[我, 喜欢, PyTorch]

B：
[我, 学习, PAD]
```

于是长度统一：

```text
3
```

形成：

```text
[batch_size, seq_len]
```

---

# 十八、但 PAD 不能让模型当正常词处理

林默马上发现：

“那模型不会以为 PAD 也是内容吗？”

“所以需要 Mask。”

比如：

```text
Token IDs:
[12, 35, 98, 0, 0]
```

对应：

```text
Attention Mask:
[1, 1, 1, 0, 0]
```

意思：

```text
1 → 真正 Token
0 → Padding
```

以后 Attention 计算时：

> 不应该让 PAD 影响正常内容。

这就是：

```text
attention_mask
```

的一个重要来源。

---

# 十九、特殊 Token

教授又写：

```text
[CLS]
[SEP]
[BOS]
[EOS]
[PAD]
[UNK]
```

不同模型会有不同特殊 Token。

常见含义：

```text
BOS
→ Begin Of Sequence


EOS
→ End Of Sequence


PAD
→ 补齐


UNK
→ 未知 Token
```

例如：

```text
<BOS> 我 喜欢 PyTorch <EOS>
```

模型不仅看到内容。

还知道：

```text
句子从哪里开始
在哪里结束
```

---

# 二十、Embedding 有一个致命问题

教授突然写：

```text
我 喜欢 狗
```

经过 Tokenizer：

```text
[12, 35, 91]
```

经过 Embedding：

```text
[
  vector(我),
  vector(喜欢),
  vector(狗)
]
```

林默点头。

教授又写：

```text
狗 喜欢 我
```

Tokenizer：

```text
[91, 35, 12]
```

Embedding：

还是：

```text
vector(狗)
vector(喜欢)
vector(我)
```

教授问：

“单看每个 Token 的 Embedding，模型知道谁在第一位吗？”

林默停住了。

“Embedding 自己不知道位置。”

“对。”

> `nn.Embedding` 只告诉模型“这是什么 Token”。

却没有告诉模型：

> “它在第几个位置。”

于是又出现了一个新问题：

# Position

---

# 二十一、词序非常重要

这两句话：

```text
狗咬人
```

和：

```text
人咬狗
```

Token 基本一样。

但顺序不同，意义完全不同。

所以：

```text
Token Embedding
```

还不够。

需要加入：

```text
Position Information
```

形成：

```text
Token Embedding
+
Position Embedding
```

例如：

```text
我
=
“我”的语义向量
+
“第1个位置”的向量
```

```text
喜欢
=
“喜欢”的语义向量
+
“第2个位置”的向量
```

```text
PyTorch
=
“PyTorch”的语义向量
+
“第3个位置”的向量
```

---

# 二十二、这就是 Transformer 的入口

教授终于画出：

```text
文本
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
Token Vectors
 ↓
加入位置信息
 ↓
Transformer
 ↓
上下文表示
```

林默看着这张图。

突然发现：

前七章所有东西都没有白学。

因为这里仍然是：

```text
Tensor
Parameter
Module
Forward
Loss
Backward
Optimizer
```

只不过输入从：

```text
普通数值
```

变成了：

```text
Token
```

---

# 二十三、“国王 - 男人 + 女人 = 女王”到底是真的吗？

林默突然想起经典例子：

```text
king - man + woman ≈ queen
```

教授说：

“这是早期词向量模型里一个非常经典的现象，但不要把它理解成所有现代 Embedding 都严格满足这种算术。”

核心想表达的是：

> 向量空间可能学习到某些语义关系方向。

比如：

```text
男性 → 女性
```

可能对应某种相对稳定的向量变化。

于是：

```text
king
 ↓
减去某些“男性”关系
 ↓
加上某些“女性”关系
 ↓
靠近 queen
```

但现代 Transformer 的表示更加上下文化。

同一个词：

```text
bank
```

在：

```text
river bank
```

和：

```text
bank account
```

里面，最终上下文表示可能完全不同。

---

# 二十四、静态 Embedding 和上下文 Embedding

教授写：

## 静态词向量

例如传统 Word2Vec：

```text
“苹果”
永远一个向量
```

不管：

```text
我吃了一个苹果
```

还是：

```text
苹果发布了新手机
```

“苹果”的基础词向量是同一个。

---

## 上下文表示

Transformer：

```text
Token Embedding
      ↓
Transformer
      ↓
Contextual Representation
```

于是：

```text
苹果 + 吃
```

可能更偏水果意义。

```text
苹果 + iPhone
```

可能更偏公司意义。

这也是 Transformer 强大的关键之一：

> **一个 Token 最终的表示，会根据上下文改变。**

---

# 二十五、PyTorch 中的一个微型 NLP 模型

教授说：

“先不学 Transformer，我们做一个极简文本分类。”

词表：

```text
0 PAD
1 我
2 喜欢
3 PyTorch
4 Java
5 讨厌
```

模型：

```python
import torch
import torch.nn as nn


class TextClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.fc = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, token_ids):

        x = self.embedding(token_ids)

        x = x.mean(dim=1)

        logits = self.fc(x)

        return logits
```

林默盯住：

```python
x = x.mean(dim=1)
```

“这是干什么？”

教授说：

“先把所有 Token 向量取平均。”

例如：

```text
[batch, seq_len, embed_dim]
```

变成：

```text
[batch, embed_dim]
```

相当于把一句话压成一个向量。

---

# 二十六、为什么这个模型很弱

模型：

```text
我 喜欢 PyTorch
```

和：

```text
PyTorch 喜欢 我
```

平均之后：

可能得到一样的结果。

因为：

\[
(a+b+c)/3
\]

和：

\[
(c+b+a)/3
\]

相同。

所以它完全不理解顺序。

林默马上说：

“这就需要位置信息和 Transformer 了。”

教授点头。

“而 Transformer 里面最关键的第一道门，就是——”

# Attention

---

# 二十七、Attention 为什么出现

教授写：

```text
小明把书给了小红，因为她明天要考试。
```

问：

“她是谁？”

林默：

“小红。”

“为什么？”

“因为我要看上下文。”

教授点头。

处理：

```text
她
```

的时候，模型需要关注：

```text
小红
考试
给
书
```

不同 Token 的重要程度不同。

这就是 Attention 最直观的思想：

> **当处理一个 Token 时，让它主动去查看序列中其他 Token，并决定应该关注谁。**

---

夜里十二点。

教授把今天所有概念连成一条线：

```text
文本
 ↓
Tokenizer
 ↓
Tokens
 ↓
Token IDs
 ↓
nn.Embedding
 ↓
Token Vectors
 ↓
Position
 ↓
Attention
 ↓
上下文化表示
 ↓
Transformer
 ↓
LLM
```

林默盯着最后四行。

他已经知道：

下一章之后，真正的大模型世界就要打开了。

---

# 下一章

## 《所有词都在互相打量——Self-Attention、Q/K/V 与 Transformer 的核心》

下一章我们会从一句话开始：

```text
“我把苹果手机放在桌子上，因为它没电了。”
```

然后一点点解释：

```text
为什么每个 Token 要生成 Q、K、V？
Q 到底在问什么？
K 到底代表什么？
V 为什么又是内容？
QKᵀ 为什么能算相关性？
为什么要除以 √d？
Softmax 在 Attention 里干什么？
Multi-Head Attention 为什么需要多个头？
Self-Attention 的复杂度为什么是 O(n²)？
```

等这一章打通，你就真正跨进 **Transformer / LLM 的核心结构**了。

下一章

# 《PyTorch：炼金术士的学徒》

## 第九章：所有词都在互相打量——Self-Attention、Q/K/V 与 Transformer 的核心

第九天。

教授没有打开代码。

他只在黑板上写了一句话：

> 我把苹果手机放在桌子上，因为它没电了。

然后问：

“这里的‘它’指什么？”

林默几乎没思考：

“苹果手机。”

“为什么不是桌子？”

“因为‘没电’和手机更相关。”

教授点头。

“很好。”

“人类读一句话时，不是机械地一个词一个词孤立理解。”

“你会让当前这个词，去参考上下文里其他词。”

他在黑板上写下：

# Attention

---

# 一、Attention 最朴素的想法

假设当前正在处理：

```text
它
```

那么模型可能会看看：

```text
我
把
苹果手机
放在
桌子
上
因为
它
没电
了
```

并给不同词不同关注程度。

比如：

```text
苹果手机  0.55
没电      0.25
桌子      0.03
我        0.02
...
```

于是：

> 当前 Token 的新表示，不再只由自己决定，而是会融合上下文信息。

这就是 Attention 的核心。

---

# 二、Self-Attention 为什么叫“Self”

教授问：

“谁在关注谁？”

林默回答：

“句子里的 Token 互相关注？”

“对。”

输入序列内部：

```text
Token 1
Token 2
Token 3
...
Token n
```

每个 Token 都可以去关注同一序列里的其他 Token。

所以叫：

> Self-Attention

也就是：

> 序列自己看自己。

---

# 三、但模型怎么知道“该关注谁”？

教授在黑板上写出三个字母：

```text
Q

K

V
```

林默皱眉。

“终于来了。”

教授笑了。

“很多人就是从这里开始把 Attention 背成公式。”

“今天别背。”

---

# 四、先把 Q、K、V 想成搜索系统

教授说：

“你做过 RAG，对吧？”

林默点头。

“假设你现在有一个 query：”

```text
Java HashMap 为什么线程不安全？
```

你会拿 query 去跟文档里的内容匹配。

比如：

```text
文档A：HashMap 扩容机制
文档B：ConcurrentHashMap
文档C：Spring IOC
```

然后判断：

```text
Query 和 A 相关程度？
Query 和 B 相关程度？
Query 和 C 相关程度？
```

Attention 也有点类似。

对于当前 Token：

> 我现在想找什么？

这就是：

# Query

简称：

```text
Q
```

---

# 五、Key 是什么？

每个被搜索的 Token 都提供一个：

> “我有什么特征，可以被别人匹配？”

这就是：

# Key

简称：

```text
K
```

比如当前：

```text
它
```

生成的 Q 可能表达：

```text
“我现在需要找一个可能被‘没电’修饰的对象。”
```

而：

```text
苹果手机
```

的 K 可能表达：

```text
“我是电子设备相关实体。”
```

桌子对应的 K：

```text
“我是家具。”
```

那么：

```text
Q(它)
```

和：

```text
K(苹果手机)
```

匹配度很高。

跟：

```text
K(桌子)
```

匹配度很低。

---

# 六、那 V 又是什么？

教授问：

“如果已经发现苹果手机最相关，接下来怎么办？”

林默说：

“把苹果手机的信息拿过来。”

“对。”

每个 Token 不仅有：

```text
Key
```

还带着真正想传递给别人的：

# Value

简称：

```text
V
```

所以：

```text
Q
→ 我想找谁？

K
→ 我适不适合被你找到？

V
→ 如果你关注我，我真正给你什么信息？
```

---

# 七、一个生活化比喻

教授画了一座图书馆。

你手里拿着问题：

```text
我要找 PyTorch 自动求导相关资料
```

这就是：

```text
Q
```

每本书封面上的关键词：

```text
深度学习
PyTorch
Java
数据库
Autograd
```

类似：

```text
K
```

而书里面真正的正文：

```text
V
```

搜索流程：

```text
Query
 ↓
和各个 Key 比较
 ↓
得到相关度
 ↓
按照相关度
 ↓
读取不同 Value
 ↓
综合得到结果
```

林默马上说：

“这和向量检索确实很像。”

“对，但 Attention 是在模型内部动态完成的。”

---

# 八、Q、K、V 是怎么来的？

教授把上章输出拿出来。

经过 Embedding 后：

```text
x
shape:
[batch, seq_len, hidden_dim]
```

例如：

```text
[1, 5, 512]
```

意思：

```text
1 个句子
5 个 Token
每个 Token 512 维
```

然后模型拥有三组可学习矩阵：

\[
W_Q,\ W_K,\ W_V
\]

计算：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

林默问：

“又是 Linear？”

“对。”

PyTorch 里完全可以理解成：

```python
self.q_proj = nn.Linear(hidden_dim, head_dim)
self.k_proj = nn.Linear(hidden_dim, head_dim)
self.v_proj = nn.Linear(hidden_dim, head_dim)
```

所以：

> Q、K、V 并不是凭空出现的。

它们是：

> 同一个输入表示，经过三种不同的可学习线性投影得到的。

---

# 九、为什么同一个 Token 要生成三个版本？

林默问：

“直接用原始 Embedding 比不行吗？”

教授说：

“因为同一个 Token 在 Attention 中承担三种角色。”

比如：

```text
苹果手机
```

作为 Query：

```text
我现在想找什么信息？
```

作为 Key：

```text
别人为什么应该关注我？
```

作为 Value：

```text
如果别人关注我，我提供什么信息？
```

这三种角色不一定需要使用同一个向量空间。

所以模型学习：

```text
W_Q
W_K
W_V
```

把原始表示变成三套不同表示。

---

# 十、Q 和 K 怎么算相关性？

教授写出最核心的一步：

\[
QK^T
\]

林默：

“矩阵乘法？”

“对。”

先看一个 Token。

假设：

```text
Q = [1, 2]
```

苹果手机的：

```text
K1 = [1, 3]
```

桌子的：

```text
K2 = [-1, 0]
```

点积：

\[
Q\cdot K_1
=
1\times1+2\times3
=
7
\]

而：

\[
Q\cdot K_2
=
1\times(-1)+2\times0
=
-1
\]

于是：

```text
苹果手机：7
桌子：-1
```

很自然：

> 苹果手机相关性更高。

---

# 十一、点积到底代表什么？

两个向量：

\[
a\cdot b
\]

可以写成：

\[
|a||b|\cos\theta
\]

所以当方向更相似：

```text
点积通常更大
```

在 Attention 里：

> Query 和 Key 点积越大，可以理解为它们越匹配。

因此：

```text
Q × K^T
```

会一次性算出：

> 每个 Query 和所有 Key 的匹配分数。

---

# 十二、整个 Attention Score 矩阵

假设有 4 个 Token：

```text
A
B
C
D
```

每个 Token 都生成：

```text
Q
K
```

那么：

\[
QK^T
\]

会得到一个：

```text
4 × 4
```

矩阵：

```text
          K_A   K_B   K_C   K_D

Q_A       2.1   0.3   1.7   0.1

Q_B       0.2   3.4   0.8   1.1

Q_C       1.9   0.7   2.5   0.4

Q_D       0.1   1.2   0.5   3.1
```

每一行：

> 一个 Token 在看所有 Token。

例如：

```text
Q_C
```

这一行：

```text
C 对 A 的关注分数
C 对 B 的关注分数
C 对 C 的关注分数
C 对 D 的关注分数
```

---

# 十三、为什么还要除以 √d？

教授终于写完整：

\[
\frac{QK^T}{\sqrt{d_k}}
\]

林默问：

“这个 `√d` 到底是干嘛？”

教授说：

“如果向量维度很大，点积容易变得很大。”

比如：

维度只有 2：

```text
点积可能：
2
5
-3
```

维度变成 512：

很多项加起来。

结果可能：

```text
50
100
-70
```

然后 Softmax：

```text
[50, 100, -70]
```

会变得极端：

```text
≈ [0, 1, 0]
```

一旦 Softmax 太饱和：

梯度可能变得很小。

所以除以：

\[
\sqrt{d_k}
\]

把数值缩小到更稳定的范围。

这就是：

> Scaled Dot-Product Attention

里的：

# Scaled

---

# 十四、为什么要 Softmax？

现在得到：

```text
scores:

[2.1, 5.7, 0.3, -1.2]
```

这还不是权重。

于是：

```python
weights = torch.softmax(scores, dim=-1)
```

变成：

```text
[0.026, 0.934, 0.037, 0.003]
```

满足：

```text
所有权重 ≥ 0

并且

总和 = 1
```

于是这些数字可以解释成：

> 当前 Token 应该从不同 Token 那里拿多少信息。

---

# 十五、最后为什么乘 V？

教授写：

\[
Attention(Q,K,V)
=
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

林默终于能拆了。

第一部分：

\[
QK^T
\]

回答：

> 谁和谁相关？

第二部分：

```text
Softmax
```

回答：

> 每个人应该占多少权重？

第三部分：

```text
× V
```

回答：

> 按照这些权重，把真正的信息混合起来。

---

# 十六、一个极简例子

假设三个 Token：

```text
苹果手机
桌子
没电
```

对于：

```text
“它”
```

Attention 权重：

```text
苹果手机：0.7
桌子：0.1
没电：0.2
```

对应 Value：

```text
V_手机
V_桌子
V_没电
```

那么输出：

\[
0.7V_{\text{手机}}
+
0.1V_{\text{桌子}}
+
0.2V_{\text{没电}}
\]

所以：

> “它”的新表示中，包含了大量手机信息，也融合了一部分“没电”的上下文。

这就是：

# Contextual Representation

---

# 十七、Self-Attention 真正改变了 Token

进入 Attention 前：

```text
“苹果”
```

可能只有基础表示：

```text
苹果这个 Token 的初始向量
```

但是经过 Attention：

在：

```text
苹果很好吃
```

中：

会融合：

```text
吃
好吃
```

的信息。

在：

```text
苹果发布了新款 Mac
```

中：

会融合：

```text
发布
Mac
```

的信息。

所以最终：

```text
苹果
```

的表示会因上下文不同而不同。

这就是上一章说的：

> 上下文化表示。

---

# 十八、PyTorch 手写一个最小 Attention

教授终于打开代码：

```python
import torch
import torch.nn.functional as F

Q = torch.randn(4, 8)
K = torch.randn(4, 8)
V = torch.randn(4, 8)

scores = Q @ K.T

scores = scores / (8 ** 0.5)

weights = F.softmax(scores, dim=-1)

output = weights @ V
```

林默问：

“shape 是什么？”

教授写：

```text
Q: [4, 8]
K: [4, 8]
V: [4, 8]
```

那么：

```text
K.T
→ [8, 4]
```

所以：

```text
Q @ K.T

[4,8] × [8,4]

→ [4,4]
```

也就是：

```text
Attention Score Matrix
```

再：

```text
weights @ V

[4,4] × [4,8]

→ [4,8]
```

最后：

> 每个 Token 仍然得到一个 8 维向量。

---

# 十九、加上 Batch 之后

真实模型不会只有一句话。

假设：

```text
batch_size = 32
seq_len = 128
hidden_dim = 512
```

输入：

```text
X:

[32, 128, 512]
```

生成 Q/K/V 后：

```text
Q:
[32, 128, 512]

K:
[32, 128, 512]

V:
[32, 128, 512]
```

计算：

```text
Q @ K^T
```

注意最后两个维度：

```text
[128,512]
×
[512,128]
```

得到：

```text
[32,128,128]
```

也就是说：

每个 batch 中：

```text
128 个 Token
×
128 个 Token
```

互相计算相关性。

---

# 二十、O(n²) 从哪来的？

教授在黑板上圈住：

```text
[seq_len, seq_len]
```

假设：

```text
seq_len = n
```

Attention Matrix：

```text
n × n
```

所以计算和内存开销的重要部分会随：

\[
n^2
\]

增长。

例如：

```text
n = 1,000

attention matrix:
1,000,000
```

如果：

```text
n = 10,000
```

变成：

```text
100,000,000
```

不是增长 10 倍。

而是：

> 100 倍。

这就是经典 Self-Attention 长上下文昂贵的重要原因。

---

# 二十一、为什么不是只看附近？

林默问：

“那为什么不只看前后几个 Token？”

教授说：

“那当然可以设计。”

比如 CNN 或局部 Attention 就会限制感受范围。

但标准 Self-Attention 的强大之处之一就是：

> 任意两个位置，可以直接建立联系。

例如：

```text
序列开头：
张三今天来到北京……

中间几千字……

最后：
他决定回家。
```

“他”可以直接关注很远之前的“张三”。

路径长度近似：

```text
1
```

而不是一层一层传过去。

这也是 Transformer 相比很多早期序列模型的重要优势。

---

# 二十二、但是一个 Attention 头够吗？

教授写一句：

> 小明去银行存了钱，回来的路上坐在河岸边。

“银行”和“河岸”的 bank 是不同含义。

模型处理语言时，可能同时需要关注：

```text
语法关系
语义关系
指代关系
位置关系
实体关系
长距离依赖
```

如果只有一套：

```text
Q/K/V
```

相当于只有一种关注视角。

于是 Transformer 做了一件事：

> 同时开多个观察窗口。

这就是：

# Multi-Head Attention

---

# 二十三、多头注意力是什么？

假设：

```text
hidden_dim = 512
num_heads = 8
```

那么：

```text
head_dim = 512 / 8 = 64
```

把表示拆成：

```text
Head 1 → 64维
Head 2 → 64维
Head 3 → 64维
...
Head 8 → 64维
```

每个 Head 都有自己的：

```text
Q
K
V
```

于是：

```text
Head 1
可能重点学习语法关系

Head 2
可能重点学习实体关系

Head 3
可能重点学习位置关系

...
```

注意：

这只是帮助理解。

现实中不能简单认定：

> 某个头一定就是“语法头”或“指代头”。

不同模型内部机制会更复杂。

---

# 二十四、多头怎么合回来？

每个 Head 输出：

```text
[batch, seq_len, head_dim]
```

8 个 Head：

```text
8 × 64 = 512
```

拼接：

```text
Concat
```

得到：

```text
[batch, seq_len, 512]
```

再经过：

```text
W_O
```

一个输出投影。

所以整体：

```text
X
 ↓
分成多个 Head
 ↓
各自 Attention
 ↓
Concat
 ↓
Linear
 ↓
输出
```

---

# 二十五、PyTorch 里的形状变换

教授写：

假设：

```text
X:
[B, N, D]
```

其中：

```text
B = batch
N = seq_len
D = hidden_dim
```

例如：

```text
[32,128,512]
```

8 头：

```text
H = 8
head_dim = 64
```

会 reshape：

```text
[32,128,512]
↓
[32,128,8,64]
```

然后通常 transpose：

```text
[32,8,128,64]
```

为什么？

因为我们希望：

```text
每个 Batch
里面
每个 Head
独立计算
128×128 的 Attention
```

所以：

```text
Q:
[B,H,N,d]

K:
[B,H,N,d]

V:
[B,H,N,d]
```

---

# 二十六、真正的多头 Attention

计算：

\[
QK^T
\]

shape：

```text
[B,H,N,d]
×
[B,H,d,N]

→

[B,H,N,N]
```

于是：

```text
每个 Batch
每个 Head

都有一个：

N × N

Attention Matrix
```

然后：

```text
softmax
```

再乘：

```text
V
```

得到：

```text
[B,H,N,d]
```

再把 Heads 合并回：

```text
[B,N,D]
```

---

# 二十七、那 Decoder 为什么不能看未来？

教授突然写一句：

```text
我 喜欢 吃 苹果
```

训练语言模型时：

当预测：

```text
喜欢
```

的时候，

如果让模型直接看到：

```text
吃 苹果
```

那就是作弊。

就像考试的时候：

> 答案直接印在后面。

所以 GPT 这种自回归语言模型还需要：

# Causal Mask

---

# 二十八、Causal Mask

假设有四个 Token：

```text
A B C D
```

标准 Self-Attention：

```text
A 可以看 A B C D
B 可以看 A B C D
C 可以看 A B C D
D 可以看 A B C D
```

但是 GPT：

```text
A 只能看 A

B 能看 A B

C 能看 A B C

D 能看 A B C D
```

形成：

```text
      A   B   C   D

A     ✓   ×   ×   ×
B     ✓   ✓   ×   ×
C     ✓   ✓   ✓   ×
D     ✓   ✓   ✓   ✓
```

这就是：

> 下三角 Mask。

---

# 二十九、Mask 是怎么实现的？

在 Softmax 之前。

原始 Score：

```text
[
 [2.1, 3.2, 1.4, 5.1],
 ...
]
```

对于未来位置：

加入：

```text
-inf
```

例如：

```text
[
 [2.1, -inf, -inf, -inf],
 [1.2,  3.1, -inf, -inf],
 ...
]
```

Softmax 后：

```text
exp(-inf) = 0
```

所以未来 Token 权重：

```text
0
```

模型就“看不见”未来。

---

# 三十、Padding Mask 和 Causal Mask 不一样

这是很容易混淆的。

Padding Mask：

```text
别看 PAD。
```

Causal Mask：

```text
别看未来。
```

一个解决：

```text
无效位置
```

一个解决：

```text
信息泄露
```

有时候两种 Mask 会同时存在。

---

# 三十一、一个真正接近 Transformer 的 Attention 模块

教授写：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.q_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(self, x):

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        d = Q.size(-1)

        scores = (
            Q @ K.transpose(-2, -1)
        ) / (d ** 0.5)

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = weights @ V

        return output
```

林默看着。

突然觉得那个经典公式没那么可怕了。

因为每一行他都知道为什么。

---

# 三十二、公式现在可以读成人话了

经典公式：

\[
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

林默现在会翻译：

```text
Q
↓
当前 Token 想找什么


K
↓
其他 Token 提供什么匹配特征


QKᵀ
↓
两两计算匹配程度


÷ √d
↓
防止数值过大


Softmax
↓
把匹配分数变成注意力权重


× V
↓
按照权重把信息聚合回来
```

教授点头。

“到这里，你才是真的理解了 Attention，而不是会背公式。”

---

# 三十三、但 Transformer 还没完成

林默问：

“所以 Transformer 就是 Attention？”

教授摇头。

“这是很多人的第二个误区。”

标准 Transformer Block 远不止：

```text
Attention
```

还包括：

```text
LayerNorm
Residual Connection
MLP / FFN
```

结构大致：

```text
Input
 │
 ▼
LayerNorm
 │
 ▼
Multi-Head Attention
 │
 ▼
 + Residual
 │
 ▼
LayerNorm
 │
 ▼
MLP / FFN
 │
 ▼
 + Residual
 │
 ▼
Output
```

林默看到：

```text
Residual
LayerNorm
FFN
```

三个陌生东西。

教授说：

“Attention 只负责让 Token 之间交换信息。”

“但是一个深层 Transformer 要堆几十层甚至上百层。”

“如果没有另外几样东西，训练会变得非常困难。”

---

夜已经深了。

教授在黑板上留下三句话：

```text
Attention
负责：

Token 和 Token 交流


MLP
负责：

每个 Token 自己进行非线性变换


Residual + LayerNorm
负责：

让深层网络更容易训练
```

林默看着整张图：

```text
Token
 ↓
Embedding
 ↓
Position
 ↓
Attention
 ↓
Residual
 ↓
LayerNorm
 ↓
MLP
 ↓
Residual
 ↓
LayerNorm
 ↓
重复几十层
 ↓
Transformer
```

真正的大模型轮廓第一次完整出现了。

# 下一章

## 《一座 Transformer Block 是如何建成的——Residual、LayerNorm、FFN 与 GELU》

下一章我们会把 Transformer Block 真正拼起来，重点解决：

```text
为什么 Attention 后面还要一个 MLP？

FFN 为什么经常先升维再降维？

Residual Connection 为什么直接写 x + attention(x)？

没有残差连接为什么深层网络难训练？

LayerNorm 和 BatchNorm 到底有什么区别？

为什么 Transformer 更喜欢 LayerNorm？

Pre-Norm 和 Post-Norm 又是什么？

GELU / SiLU / SwiGLU 为什么逐渐取代简单 ReLU？
```

等这一章讲完，我们就可以用 PyTorch **手写一个简化版 Transformer Block**，再往后就能继续走到 GPT 的训练与生成过程。

下一章

# 《PyTorch：炼金术士的学徒》

## 第十章：深渊中的通天塔——Residual、LayerNorm、FFN 与 Transformer Block

第十天。

凌晨两点，实验室的 GPU 服务器还在运转。

林默已经连续训练了六个小时。

屏幕上，一串数字正在缓慢变化。

```
Epoch 1   Loss: 4.821
Epoch 2   Loss: 4.163
Epoch 3   Loss: 3.892
Epoch 4   Loss: 3.761
Epoch 5   Loss: 3.758
Epoch 6   Loss: 3.759
Epoch 7   Loss: 3.758
```

“奇怪。”

林默皱起眉头。

“为什么 Loss 不下降了？”

陈教授走到他身后。

“你的模型有多少层？”

“我把昨天的 Attention 堆了起来。”

“多少层？”

“96 层。”

教授沉默了几秒。

“你为什么觉得，把 96 个 Attention 堆起来，就能得到一个优秀的 Transformer？”

林默愣住。

“难道不是吗？”

教授拿起粉笔，在黑板上画了一座高塔。

```
              第 96 层
                 │
              第 95 层
                 │
              第 94 层
                 │
                ...
                 │
               第 3 层
                 │
               第 2 层
                 │
               第 1 层
                 │
                输入
```

“你现在遇到的，是深度学习发展过程中一个极其重要的问题。”

“网络可以堆得很深。”

“但——”

> 网络越深，并不意味着它就越容易训练。

“今天，我们要学会建造一座能够不断加高、却不容易在训练中崩塌的塔。”

# 一、第一场灾难：梯度在深层网络中消失

教授没有立即讨论 Transformer。

而是重新写下第一章的公式：

\\[ y=wx \\]

然后：

\\[ L=(y-target)^2 \\]

“你还记得 `backward()` 的本质吗？”

林默回答：

“链式法则。”

“很好。”

教授写下：

\\[ x\rightarrow f_1\rightarrow f_2\rightarrow f_3\rightarrow L \\]

如果求：

\\[ \frac{\partial L}{\partial x} \\]

需要：

\\[ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x} \\]

林默点头。

“这不是之前讲过的吗？”

教授说：

“假设每一层的局部导数都是 0.5。”

那么经过一层：

\\[ 0.5 \\]

经过两层：

\\[ 0.5^2=0.25 \\]

经过十层：

\\[ 0.5^{10}\approx0.000977 \\]

经过一百层：

\\[ 0.5^{100}\approx7.89\times10^{-31} \\]

林默愣住。

“一百层之后，几乎是零？”

“对。”

这是一个简化的标量例子。真实神经网络的梯度涉及矩阵、激活函数和复杂的计算路径，不是简单把每层梯度相乘成一个数字，但它能说明问题。

如果反向传播中的许多局部变换持续压缩梯度，就可能出现：

> Vanishing Gradient——梯度消失。

反过来，如果连续放大，也可能出现：

> Exploding Gradient——梯度爆炸。

于是林默终于意识到：

“我堆的 96 层 Attention，可能根本没有顺利学起来。”

教授点头。

“这正是残差连接想解决的重要问题之一。”

# 二、残差连接：给梯度修一条高速公路

教授在黑板上画：

普通网络：

```
       x
       │
       ▼
   Attention
       │
       ▼
       y
```

然后画出另一种结构：

```
       x ──────────────┐
       │               │
       ▼               │
   Attention           │
       │               │
       ▼               │
       + ◄─────────────┘
       │
       ▼
       y
```

林默指着旁边那条线。

“这是什么？”

“捷径。”

“什么意思？”

教授写下：

普通网络：

\\[ y=F(x) \\]

残差网络：

\\[ \boxed{y=x+F(x)} \\]

林默看着公式。

“就多加了一个 x？”

“对。”

“这也太简单了吧？”

教授笑了。

“恰恰是这一个加法，改变了深度神经网络的训练方式。”

# 三、为什么 `x + F(x)` 有用？

教授写：

\\[ y=x+F(x) \\]

现在对 \\(x\\) 求导：

\\[ \frac{\partial y}{\partial x} = I+\frac{\partial F(x)}{\partial x} \\]

这里的 \\(I\\) 是恒等映射对应的单位矩阵。

如果暂时只考虑一维情况：

\\[ \frac{dy}{dx} = 1+F'(x) \\]

林默突然坐直。

“等等。”

“以前梯度只能经过 F。”

“现在多了一条导数为 1 的路径！”

教授点头。

```
普通网络反向传播：

Loss
  │
  ▼
F 的梯度
  │
  ▼
前一层


残差网络反向传播：

       Loss
         │
         ▼
       加法
       /  \
      /    \
     ▼      ▼
   F路径   恒等路径
     \      /
      \    /
       ▼
      前一层
```

恒等路径为梯度传播提供了一条额外通道。

不过教授马上补充：

“不能因此认为残差网络绝对不会出现梯度消失或爆炸。不同路径的梯度仍可能相互抵消或放大。”

“残差连接的价值在于，它通常能明显改善深层网络的优化难度。”

# 四、残差连接的第二个秘密：学习变化，而不是从头重建

教授换了一个角度。

“假设一个 Token 已经拥有不错的表示。”

例如：

```
x = “苹果手机”的上下文表示
```

经过某个 Attention 层后，模型只是希望：

```
增加一点“没电”的语义信息
```

普通网络：

\\[ y=F(x) \\]

意味着 F 必须产生完整的新表示。

但残差网络：

\\[ y=x+F(x) \\]

可以让：

```
x
│
├── 保留原来的信息
│
└── F(x) 负责补充或修正信息
```

例如：

```
x：

已经知道这是“手机”


F(x)：

补充“没电”相关信息


x + F(x)：

得到更丰富的手机表示
```

这只是帮助理解的类比，真实表示并不是逐项标注语义。

但设计思想非常重要：

> 模型不一定需要每层都重新生成全部信息，它可以在已有表示的基础上学习增量修正。

# 五、PyTorch 里的残差连接

教授打开编辑器：

```
import torchimport torch.nn as nnclass ResidualBlock(nn.Module):    def __init__(self, dim):        super().__init__()        self.net = nn.Sequential(            nn.Linear(dim, dim),            nn.ReLU(),            nn.Linear(dim, dim)        )    def forward(self, x):        return x + self.net(x)
```

林默看着：

```
return x + self.net(x)
```

“就这一句？”

“核心就是这一句。”

假设输入：

```
x = torch.randn(32, 128)
```

形状：

```
[32, 128]
```

那么：

```
self.net(x)
```

也必须输出：

```
[32, 128]
```

才能相加。

教授提醒：

> 最普通的恒等残差连接要求两条路径的输出形状兼容。

如果：

```
x:
[32, 128]

F(x):
[32, 256]
```

就不能直接进行这种残差相加。

某些网络会使用投影层调整维度，但 Transformer Block 通常让外部隐藏维度保持不变。

# 六、第二场灾难：网络里的数值开始失控

林默以为问题解决了。

他给 96 层模型加入残差连接。

然后重新训练。

但这一次，Loss 依旧不够稳定。

教授看着日志。

“你解决了梯度传播的一部分问题。”

“但是还有另外一件事。”

“你的每一层，拿到的输入数值是什么分布？”

林默一愣。

“数值分布？”

教授写：

第一层输出：

```
[0.1, 0.3, -0.2, 0.5]
```

中间某层输出可能变成：

```
[12.3, -18.7, 25.1, 8.4]
```

更深的某层：

```
[103.2, -87.4, 204.7, -51.8]
```

“这些数字只是示意，不代表所有网络都会如此。”

教授继续：

“如果不同层的激活尺度变化剧烈，优化就可能更加困难。”

“所以我们需要某种机制，控制数值尺度。”

黑板上出现：

# Normalization

# 七、LayerNorm：让每个 Token 整理自己的数值

教授写：

```
nn.LayerNorm(512)
```

林默问：

“又一个新层？”

“对。”

“但是理解它之前，我们先做一个很简单的数学计算。”

假设一个 Token 的隐藏向量：

\\[ x=[1,2,3] \\]

计算平均值：

\\[ \mu=\frac{1+2+3}{3}=2 \\]

然后计算方差：

\\[ \sigma^2 = \frac{(1-2)^2+(2-2)^2+(3-2)^2}{3} = \frac23 \\]

标准化：

\\[ \hat{x}\_i= \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} \\]

忽略很小的 \\(\epsilon\\)，得到近似：

\\[ \hat{x}=[-1.225,0,1.225] \\]

林默看着结果。

“平均值变成零了。”

“对。”

“方差也变成接近一。”

“对。”

# 八、LayerNorm 不是简单把数值固定住

教授继续写：

\\[ y=\gamma\hat{x}+\beta \\]

林默发现两个新变量：

```
γ gamma
β beta
```

教授解释：

“标准化之后，模型还可以通过可学习参数调整每个特征维度的缩放和平移。”

所以 LayerNorm 内部通常也有：

```
weight → gamma

bias → beta
```

在 PyTorch 里：

```
ln = nn.LayerNorm(512)
```

可以看到：

```
print(ln.weight.shape)print(ln.bias.shape)
```

输出：

```
torch.Size([512])
torch.Size([512])
```

林默突然想起第二章。

“它们也是 Parameter？”

“没错。”

所以：

```
model.parameters()
```

不仅包含：

```
Linear 的 weight、bias
```

还可能包含：

```
LayerNorm 的 weight、bias
Embedding 的 weight
Attention 的投影参数
```

优化器统一管理这些参数。

# 九、LayerNorm 究竟沿哪个维度计算？

教授写出 Transformer 中最经典的 Tensor 形状：

```
X.shape = [B, N, D]
```

例如：

```
[32, 128, 512]
```

代表：

```
32   → Batch Size
128  → Sequence Length
512  → Hidden Dimension
```

然后：

```
ln = nn.LayerNorm(512)y = ln(x)
```

问题来了。

“它对哪一维计算均值和方差？”

林默思考片刻。

“512？”

“对。”

LayerNorm 的归一化范围

示意图：每个 Token 独立地沿隐藏特征维度计算统计量，不是把整个 Batch 混在一起。

教授说：

“对于这种输入，`LayerNorm(512)` 会分别处理每个样本、每个 Token 的 512 维向量。”

也就是：

```
Batch 1
  Token 1 → 对512维归一化
  Token 2 → 对512维归一化
  ...

Batch 2
  Token 1 → 对512维归一化
  Token 2 → 对512维归一化
  ...
```

不会把所有 Token 和 Batch 的数值混在一起求同一组均值、方差。

# 十、为什么 Transformer 常用 LayerNorm，而不是 BatchNorm？

林默突然想起第六章。

“教授，BatchNorm 不也能归一化吗？”

“当然。”

“那为什么 Transformer 里常见 LayerNorm？”

教授画出两个概念。

| 对比          | BatchNorm          | LayerNorm         |
| ----------- | ------------------ | ----------------- |
| 典型归一化范围     | 跨 Batch 等维度，对各通道统计 | 对每个样本的指定特征维度统计    |
| 是否受同批其他样本影响 | 训练时通常会             | 通常不会              |
| 训练与推理统计方式   | 常见实现存在差异           | 通常一致              |
| 常见使用场景      | CNN 等              | Transformer、LLM 等 |

教授说：

“语言模型的输入长度、批次组成可能不断变化。”

“尤其在自回归生成时，每次可能只处理一个新 Token。”

“LayerNorm 不依赖跨样本统计，因此非常适合这种场景。”

不过他又补充：

“现代大语言模型也大量使用 RMSNorm 等变体，不能简单认为所有 Transformer 都一定使用 LayerNorm。”

林默在笔记上写下：

> BatchNorm 更多考虑批次等维度上的统计，LayerNorm 通常独立处理每个 Token 的特征向量。

# 十一、第三个问题：Attention 为什么还不够？

教授终于把上一章的 Attention 搬了出来。

```
Token 1
Token 2
Token 3
Token 4
    │
    ▼
Self-Attention
    │
    ▼
新的 Token 表示
```

“Attention 干什么？”

林默回答：

“让 Token 根据 Q、K、V 从其他 Token 聚合信息。”

“很好。”

“那它是否已经足够？”

林默犹豫了。

“好像还缺非线性处理？”

教授点头。

“这就是今天第三个核心模块。”

# FFN

全称：

Feed-Forward Network

在 Transformer 中通常也称：

Position-wise Feed-Forward Network。

# 十二、FFN：每个 Token 自己的加工车间

教授画了两座工厂。

第一座：

```
Self-Attention

Token A ───┐
Token B ───┼──► 信息交换
Token C ───┼──► 上下文融合
Token D ───┘
```

第二座：

```
FFN

Token A → 自己加工

Token B → 自己加工

Token C → 自己加工

Token D → 自己加工
```

林默问：

“什么意思？”

教授解释：

“Self-Attention 的一个重要作用，是在不同 Token 之间交换信息。”

“FFN 则通常对每个 Token 的隐藏向量，独立应用同一套非线性变换。”

也就是说：

> Attention 负责跨 Token 的信息交互，FFN 负责对每个 Token 的特征进行非线性加工。

这是一种非常实用的理解方式，并非两者功能的绝对边界。

# 十三、FFN 为什么要先升维，再降维？

教授写下：

```
nn.Linear(512, 2048)
```

再写：

```
nn.Linear(2048, 512)
```

林默看着数字。

“512 变 2048，再变回 512？”

“对。”

经典 FFN 的结构：

```
输入

512维
  │
  ▼
Linear(512, 2048)
  │
  ▼
2048维
  │
  ▼
激活函数
  │
  ▼
Linear(2048, 512)
  │
  ▼
512维
```

对应公式：

\\[ FFN(x)=W_2\phi(W_1x+b_1)+b_2 \\]

其中：

\\[ \phi \\]

代表激活函数。

教授问：

“为什么不是从头到尾都保持 512 维？”

林默思考。

“因为扩展中间维度，可以提供更丰富的中间特征表达？”

“对。”

更大的中间维度给非线性变换提供了更多表达容量。

但教授强调：

> 升维并不自动保证模型更好，它也会增加参数量和计算量。

# 十四、FFN 的参数量究竟有多大？

教授写：

```
fc1 = nn.Linear(512, 2048)fc2 = nn.Linear(2048, 512)
```

先算第一层。

Weight：

\\[ 512\times2048=1,048,576 \\]

Bias：

\\[ 2048 \\]

第一层总计：

\\[ 1,050,624 \\]

第二层：

Weight：

\\[ 2048\times512=1,048,576 \\]

Bias：

\\[ 512 \\]

第二层总计：

\\[ 1,049,088 \\]

整个 FFN：

\\[ \boxed{2,099,712\text{ 个参数}} \\]

林默愣住。

“一个小小的 FFN 就两百多万参数？”

教授点头。

“所以千万不要觉得 Transformer 里的参数主要都在 Attention。”

“在许多常见架构中，FFN/MLP 占据了相当大比例的参数。”

# 十五、为什么激活函数变成 GELU？

林默想起第三章：

```
nn.ReLU()
```

于是问：

“这里直接用 ReLU 不就可以了吗？”

教授回答：

“当然可以。早期 Transformer 也使用过 ReLU。”

“但是后来出现了许多更常用的选择。”

例如：

```
nn.GELU()
```

GELU：

Gaussian Error Linear Unit

公式：

\\[ GELU(x)=x\Phi(x) \\]

其中：

\\[ \Phi(x) \\]

是标准正态分布的累积分布函数。

林默看着公式。

“这又是什么意思？”

教授画出 ReLU 和 GELU 的曲线。

ReLU

负数直接截断为零，零点处有折角

GELU

平滑过渡，部分负输入也有非零输出

教授解释：

ReLU：

\\[ ReLU(x)=\max(0,x) \\]

对于负数：

```
全部输出 0
```

而 GELU 不会简单地把所有负数直接截断到零。

例如：

```
输入 -1

ReLU(-1) = 0

GELU(-1) ≈ -0.159
```

林默点头。

“所以 GELU 的变化更加平滑。”

“对。”

“但不要记成 GELU 永远比 ReLU 更好。不同任务、模型架构和硬件实现会影响最终选择。”

# 十六、SiLU 与 SwiGLU：大模型时代的新车间

教授又写出两个名字：

```
SiLU

SwiGLU
```

SiLU：

\\[ SiLU(x)=x\sigma(x) \\]

其中：

\\[ \sigma(x) \\]

是 Sigmoid。

PyTorch：

```
activation = nn.SiLU()
```

而 SwiGLU 则更特别。

它不是简单地把：

```
Linear
↓
ReLU
↓
Linear
```

换掉一个激活函数。

它引入了门控结构。

一种常见形式：

\\[ SwiGLU(x)= \left( SiLU(xW_g)\odot(xW_u) \right)W_d \\]

忽略偏置。

其中：

\\[ \odot \\]

表示逐元素相乘。

教授画：

```
                  x
                 / \
                /   \
               ▼     ▼
             Linear Linear
               │      │
               ▼      │
              SiLU    │
               │      │
               └──┬───┘
                  │
                  ▼
                逐元素乘
                  │
                  ▼
                Linear
                  │
                  ▼
                 输出
```

林默突然发现：

“这里有两条路径。”

“一条经过 SiLU，另一条保留另一种投影。”

“然后两个结果相乘？”

“对。”

教授说：

“你可以把它理解为一种可学习的门控机制，让不同特征之间产生更灵活的交互。”

现代不少 LLM 使用类似 SwiGLU 的 FFN 变体。

注意，不同模型的隐藏维度设置可能不同，并不一定继续沿用经典 FFN 的四倍扩展比例。

# 十七、终于开始组装 Transformer Block

教授擦干净黑板。

“现在你已经拥有三种零件。”

```
第一种：

Self-Attention
负责跨 Token 信息交互


第二种：

FFN / MLP
负责每个 Token 的非线性变换


第三种：

Residual + Normalization
改善深层网络的优化和数值行为
```

“我们现在开始组装。”

# 十八、Pre-Norm Transformer Block

教授首先画出一种常见的现代结构：

Pre-Norm Transformer Block

林默盯着这张图。

“先 LayerNorm。”

“然后 Attention。”

“再和最初的 x 相加。”

“接着又 LayerNorm。”

“再 FFN。”

“最后再加一次残差？”

教授点头。

数学上：

\\[ h=x+Attention(LN(x)) \\]

然后：

\\[ \boxed{y=h+FFN(LN(h))} \\]

这就是一种标准的 Pre-Norm Block 写法。

# 十九、为什么叫 Pre-Norm？

教授问：

“Norm 在哪里？”

林默回答：

“Attention 和 FFN 前面。”

“所以叫 Pre-Norm。”

对应的另一种结构：

Post-Norm：

\\[ h=LN(x+Attention(x)) \\]

\\[ y=LN(h+FFN(h)) \\]

两种设计的区别：

|              | Pre-Norm    | Post-Norm            |
| ------------ | ----------- | -------------------- |
| LayerNorm 位置 | 子层前         | 残差相加后                |
| 残差主路径        | 更直接         | 经过后置归一化              |
| 深层优化         | 通常更容易稳定训练   | 可能更依赖初始化和训练配置        |
| 使用情况         | 现代 LLM 中很常见 | 原始 Transformer 等架构使用 |

教授补充：

“原始的 Transformer 采用的是 Post-Norm 结构。后来许多深层语言模型采用 Pre-Norm 或其变体。”

“不要把 Pre-Norm 理解成所有指标上都绝对更好，它主要是一个重要的优化与架构权衡。”

# 二十、用 PyTorch 亲手造一个 Transformer Block

林默已经迫不及待。

“教授，我们自己实现吧。”

教授打开编辑器。

“这次我们不只画结构。”

“我们真正写一个能执行的 Transformer Block。”

先做一个简化实现，使用 PyTorch 已有的多头 Attention。

```
import torchimport torch.nn as nnclass TransformerBlock(nn.Module):    def __init__(self, d_model=128, num_heads=4):        super().__init__()        # 1. Attention 前的 LayerNorm        self.ln1 = nn.LayerNorm(d_model)        # 2. 多头注意力        self.attn = nn.MultiheadAttention(            embed_dim=d_model,            num_heads=num_heads,            batch_first=True        )        # 3. FFN 前的 LayerNorm        self.ln2 = nn.LayerNorm(d_model)        # 4. FFN        self.ffn = nn.Sequential(            nn.Linear(d_model, d_model * 4),            nn.GELU(),            nn.Linear(d_model * 4, d_model)        )    def forward(self, x):        # ---- Attention 子层 ----        h = self.ln1(x)        attn_out, _ = self.attn(            h, h, h,            need_weights=False        )        x = x + attn_out        # ---- FFN 子层 ----        h = self.ln2(x)        ffn_out = self.ffn(h)        x = x + ffn_out        return x
```

林默看完，几乎立刻指出：

```
attn_out, _ = self.attn(h, h, h)
```

“为什么输入三个 h？”

教授回答：

“因为这是 Self-Attention。”

对应：

```
Query 来源：h

Key 来源：h

Value 来源：h
```

注意，这不是说 Q、K、V 三个投影后的向量完全相同。

`nn.MultiheadAttention` 内部依然会执行各自的可学习投影。

# 二十一、为什么 `batch_first=True`？

教授又指向：

```
batch_first=True
```

之前我们一直使用：

```
[B, N, D]
```

例如：

```
[32, 128, 512]
```

但 PyTorch 的 `nn.MultiheadAttention` 在 `batch_first=False` 时，三维输入通常使用：

```
[N, B, D]
```

设置：

```
batch_first=True
```

后，输入就可以使用：

```
[batch_size, seq_len, hidden_dim]
```

也就是：

```
[B, N, D]
```

这与我们前面学习的 Tensor 形状保持一致。

# 二十二、第一次运行 Transformer Block

教授写：

```
block = TransformerBlock(    d_model=128,    num_heads=4)x = torch.randn(2, 10, 128)y = block(x)print(y.shape)
```

输出：

```
torch.Size([2, 10, 128])
```

林默问：

“为什么维度没变？”

教授指向两次残差连接。

“因为这个 Block 的输入和输出，都需要保持相同的模型隐藏维度。”

具体：

```
输入：

[B, N, D]
[2, 10, 128]

       ↓

LayerNorm

[2, 10, 128]

       ↓

Multi-Head Attention

[2, 10, 128]

       ↓

Residual Add

[2, 10, 128]

       ↓

FFN

[2, 10, 512]
       ↓
[2, 10, 128]

       ↓

Residual Add

[2, 10, 128]
```

所以每个 Block 都能保持：

\\[ [B,N,D] \\]

不变。

这正是它能够被反复堆叠的重要原因。

# 二十三、但是这个 Block 还不能直接当 GPT

林默突然说：

“那我只要把它堆 96 层，不就做出了 GPT？”

教授摇头。

“还差一个极其关键的约束。”

“上一章我们刚讲过。”

林默很快反应过来。

“Causal Mask！”

“没错。”

刚才我们写：

```
attn_out, _ = self.attn(    h, h, h,    need_weights=False)
```

没有传入因果 Mask。

因此：

> 这个实现默认允许每个 Token 关注序列中其他所有 Token。

它更接近一个不带 Padding Mask 的双向 Self-Attention 模块，而不是完整的 GPT 自回归注意力层。

如果想做 GPT，需要限制：

```
Token 1 不能看到 2、3、4……

Token 2 不能看到 3、4……

Token 3 不能看到 4……
```

可以构造：

```
seq_len = x.size(1)causal_mask = torch.triu(    torch.ones(        seq_len,        seq_len,        device=x.device,        dtype=torch.bool    ),    diagonal=1)
```

然后：

```
attn_out, _ = self.attn(    h, h, h,    attn_mask=causal_mask,    need_weights=False)
```

这里布尔 `attn_mask=True` 的位置，在 `nn.MultiheadAttention` 中表示禁止关注。

教授提醒：

“不同 Attention API 的布尔 Mask 约定可能不同。以后使用 `scaled_dot_product_attention` 时，不能想当然地照搬这个规则。”

# 二十四、现在真正理解一层 Transformer 在做什么

教授指着输入句子：

> 我把苹果手机放在桌子上，因为它没电了。

经过 Tokenizer：

```
[我, 把, 苹果手机, 放在, 桌子, 上, ...]
```

Embedding：

```
[B, N, D]
```

进入一个 Transformer Block。

第一步：

```
LayerNorm
```

对输入特征做归一化。

第二步：

```
Self-Attention
```

Token 之间交换上下文信息。

例如“它”可能融合与“苹果手机”“没电”有关的信息。

第三步：

```
Residual
```

把原有表示与 Attention 输出相加。

第四步：

```
LayerNorm
```

对更新后的表示再次归一化。

第五步：

```
FFN
```

每个 Token 独立经过非线性特征变换。

第六步：

```
Residual
```

保留主路径并叠加本层的 FFN 输出。

于是：

> 一个 Transformer Block 完成了一轮信息交互和特征加工。

而接下来：

```
Block 1
   ↓
Block 2
   ↓
Block 3
   ↓
...
   ↓
Block 96
```

模型就能够不断构建更复杂的上下文表示。

# 二十五、为什么 Transformer 可以并行处理 Token？

林默想起之前学过的 RNN。

“教授，以前 RNN 不是要一个 Token 一个 Token 处理吗？”

教授点头。

“经典 RNN 的隐藏状态依赖前一个时间步，因此序列维度上的计算有较强的顺序依赖。”

```
RNN：

Token 1
  ↓
Hidden 1
  ↓
Token 2
  ↓
Hidden 2
  ↓
Token 3
  ↓
...
```

而 Transformer 在训练时可以让一整个序列共同进入 Attention：

```
Token 1 ─┐
Token 2 ─┤
Token 3 ─┼──► 矩阵运算
Token 4 ─┤
Token 5 ─┘
```

即使使用 Causal Mask，训练期间也可以并行计算多个位置的预测。

林默问：

“那为什么 GPT 推理还是一个 Token 一个 Token 生成？”

教授回答：

“因为生成时，下一个 Token 还没有出现。”

“当前 Token 需要根据之前已经生成的内容进行预测。”

这就涉及：

> Autoregressive Generation——自回归生成。

不过这是下一阶段的内容。

# 二十六、Transformer Block 里的参数到底有多少？

教授突然问：

“我们现在使用：”

```
d_model = 512

num_heads = 8

FFN hidden = 2048
```

“那么一个 Block 大约有多少参数？”

林默开始计算。

Attention 中，经典多头注意力通常存在：

```
W_Q

W_K

W_V

W_O
```

每个主要投影矩阵的大小：

\\[ 512\times512 \\]

四个：

\\[ 4\times512^2=1,048,576 \\]

FFN：

\\[ 512\times2048+2048\times512 \\]

等于：

\\[ 2,097,152 \\]

两个 LayerNorm：

```
gamma + beta
```

总计：

\\[ 2\times(512+512)=2048 \\]

如果加上四个 Attention 线性投影和两个 FFN 线性层的偏置：

\\[ 2048+2560=4608 \\]

那么一个这种具体配置的 Block，总计：

\\[ 1,048,576 + 2,097,152 + 2048 + 4608 \\]

\\[ \boxed{3,152,384\text{ 个参数}} \\]

林默愣住。

“一个 Block 就三百多万？”

“对。”

如果有 96 层：

\\[ 3,152,384\times96 \\]

大约：

\\[ 3.03亿 \\]

这还没有加：

```
Token Embedding

位置相关参数（如果有）

最终输出层

其他可能的模块
```

教授说：

“而且更大的模型会把 `d_model` 扩展得更宽，参数量增长会非常快。”

# 二十七、为什么大模型越宽，参数增加得越快？

教授在黑板上写：

\\[ D=d\_{model} \\]

如果 FFN 扩展比例固定为 4：

Attention 主要权重：

\\[ 4D^2 \\]

FFN 主要权重：

\\[ 8D^2 \\]

合起来：

\\[ 12D^2 \\]

忽略偏置和归一化参数：

\\[ \boxed{BlockParams\approx12D^2} \\]

林默马上发现：

“所以如果 hidden_dim 翻倍——”

\\[ (2D)^2=4D^2 \\]

“单层主要参数量大约变成四倍！”

教授点头。

“这就是宽度扩展非常昂贵的原因之一。”

而层数翻倍，在其他配置不变时，Block 总参数量则大致翻倍。

因此模型规模可以通过：

```
增加深度

增加宽度

改变 FFN 扩展比例

改变 Attention 结构

改变词表规模

引入 MoE
```

等方式调整。

# 二十八、为什么现代模型不一定使用普通 Multi-Head Attention？

教授又在黑板上写下：

```
MHA

MQA

GQA
```

林默疑惑。

“刚理解多头，怎么又变了？”

教授说：

“因为我们刚实现的是一种经典形式。”

“但实际 LLM 工程需要考虑显存和推理速度。”

传统 MHA：

```
每个 Query Head
有对应的 Key Head 和 Value Head
```

MQA：

```
多个 Query Head
共享一组 K/V Head
```

GQA：

```
若干 Query Head
共享一组 K/V Head
```

主要目标之一：

> 减少自回归推理中 KV Cache 的内存占用和访存成本。

林默立刻联想到第六章。

“也就是说，有时候模型快不快，不仅仅取决于计算量？”

“对。”

“还取决于显存带宽、KV Cache、内存布局、算子实现和批处理方式。”

教授笑了。

“你已经开始从训练算法走向大模型系统工程了。”

# 二十九、一个隐藏的问题：深层 Transformer 是不是越多层越好？

林默看着自己最初的 96 层模型。

“那我现在有了 LayerNorm、残差和 FFN，是不是层数越多越好？”

教授摇头。

“不是。”

深度增加通常意味着：

```
更多参数

更多计算

更高训练成本

更长前向和反向传播路径

可能更高的推理延迟
```

但收益不一定线性增加。

真实训练还受到：

```
训练数据量

数据质量

优化器

学习率

初始化

Batch Size

模型宽度

训练 Token 数

计算预算
```

等因素影响。

甚至可能出现：

> 模型更大，但训练不足，最终效果反而不如配置合理的小模型。

所以现代模型设计不仅要考虑：

```
能不能训练
```

还要考虑：

```
能否在有限资源下有效训练

能否在目标硬件上高效推理
```

# 三十、现在可以把前十章全部连起来了

教授在黑板上画了一张完整架构图。

```
                   原始文本
                      │
                      ▼
                  Tokenizer
                      │
                      ▼
                   Token IDs
                      │
                      ▼
                 nn.Embedding
                      │
                      ▼
                  Token 向量
                      │
                      ▼
                 位置信息
                      │
                      ▼
              ┌─────────────────┐
              │ Transformer     │
              │ Block 1         │
              │                 │
              │ LayerNorm       │
              │ Attention       │
              │ Residual        │
              │ LayerNorm       │
              │ FFN             │
              │ Residual        │
              └─────────────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Transformer     │
              │ Block 2         │
              └─────────────────┘
                      │
                      ▼
                     ...
                      │
                      ▼
              ┌─────────────────┐
              │ Transformer     │
              │ Block N         │
              └─────────────────┘
                      │
                      ▼
              最终隐藏状态
                      │
                      ▼
                 输出投影
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
                 Optimizer
                      │
                      ▼
                  更新参数
```

林默盯着这张图。

他突然意识到，最早那个陌生的：

```
import torch
```

如今已经展开成了一整套体系。

```
Tensor
  ↓
Autograd
  ↓
Parameter
  ↓
nn.Module
  ↓
Linear / Activation
  ↓
Loss
  ↓
Dataset / DataLoader
  ↓
GPU / CUDA
  ↓
Optimizer
  ↓
Embedding
  ↓
Self-Attention
  ↓
Transformer Block
```

# 三十一、林默最后还有三个问题

教授准备离开。

林默突然叫住他。

“教授，我还想确认三个问题。”

“说。”

第一个问题：Residual 会不会增加模型参数？

“如果只是：”

```
x = x + f(x)
```

“残差相加本身没有可学习参数。”

“但如果需要额外的投影层来匹配维度，那么投影层会有参数。”

第二个问题：LayerNorm 会改变 Tensor 的 Shape 吗？

“正常情况下不会。”

```
输入：[B, N, D]

输出：[B, N, D]
```

“它主要改变数值，不改变整体形状。”

第三个问题：FFN 会不会让不同 Token 直接互相交流？

“在标准的 position-wise FFN 中，不会。”

“它对每个 Token 位置独立应用同一套变换。”

“跨 Token 的信息交互，主要发生在 Attention 等模块中。”

林默点头。

三个问题，终于完全清楚。

# 三十二、最后一道理解测试

教授在门口留下五道题。

Transformer Block 随堂测试

0/5 已回答

1\. 残差连接 x + F(x) 的核心作用是什么？

让模型参数量翻倍

为信息和梯度提供额外的恒等路径

自动消除所有梯度爆炸

2\. 对于 X.shape=[32,128,512]，nn.LayerNorm(512) 默认沿什么维度归一化？

Batch 维度 32

序列维度 128

每个 Token 的最后 512 维

3\. 标准 Transformer FFN 为什么先升维再降维？

增加中间非线性特征表达容量

为了跨 Token 交换信息

为了改变 Batch Size

4\. Pre-Norm 指的是哪种结构？

LN(x + Attention(x))

x + Attention(LN(x))

Attention(x) + FFN(x)

5\. 一个常规 Transformer Block 的输入输出维度通常如何变化？

[B,N,D] → [B,N,D]

[B,N,D] → [B,N,4D]

[B,N,D] → [B,D]

## 第十一章预告：《预测下一个词——从 Transformer 到 GPT 的诞生》

夜色渐深。

林默望着自己写好的 Transformer Block：

```
x = block(x)
```

他突然发现一个更根本的问题。

“教授。”

“我们费了这么大劲，终于把输入文本变成了上下文表示。”

“但——”

“它究竟是怎么学会说话的？”

教授在黑板上写下一句话：

> 我今天学习了 PyTorch。

然后擦掉最后一个词。

变成：

> 我今天学习了 \_\_\_\_\_\_

“如果让模型预测空白处的下一个 Token，会发生什么？”

林默想了想。

“它可能预测 PyTorch？”

“对。”

“如果让它预测完一个，再把预测结果放回输入呢？”

林默突然意识到了什么。

教授写：

```
输入：

我今天学习了

       ↓

预测下一个 Token：

PyTorch

       ↓

拼接：

我今天学习了 PyTorch

       ↓

再次预测：

。

       ↓

拼接：

我今天学习了 PyTorch。

       ↓

继续预测……
```

林默猛地抬头。

“所以大语言模型生成文本——”

“就是不断预测下一个 Token？”

教授点头。

“你终于走到了 GPT 的起点。”

下一章，我们会从零组装一个真正可以训练、预测并生成文本的 Mini-GPT，深入讲解：

- Decoder-Only Transformer 为什么适合文本生成。
- Next Token Prediction 如何构造训练标签。
- Causal Mask 如何保证不偷看答案。
- 为什么一次前向传播可以同时训练整个序列的多个预测位置。
- Logits、Temperature、Top-k、Top-p 如何决定生成结果。
- 为什么模型训练时可以并行处理序列，而推理时通常逐 Token 生成。
- 如何使用 PyTorch 完成一个最小可运行的 GPT 训练与生成程序。

下一章：《预测下一个词——从 Transformer 到 GPT 的诞生》。