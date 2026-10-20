from pathlib import Path
import html, json, re, subprocess, shutil
import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'docs'
OUT.mkdir(exist_ok=True)
NAMES = ['一块奇怪的石板', '神经网络诞生', '让神经网络学会拐弯', '审判模型的法官', '粮草与运输队', '跨越两界', '寻找最低山谷', '文字如何进入神经网络', '所有词都在互相打量', '深渊中的通天塔']
TOPICS = ['Tensor · Autograd · 第一次训练', 'nn.Module · Linear · Parameter', '激活函数 · ReLU · 非线性', 'Loss · MSE · Softmax · CrossEntropy', 'Dataset · DataLoader · Batch · Epoch', 'GPU · CUDA · train() · eval()', 'SGD · Momentum · AdamW · 学习率', 'Token · Embedding · 词表与位置', 'Self-Attention · Q/K/V · 多头注意力', 'Residual · LayerNorm · FFN · Transformer Block']
BRIDGES = ['从一个数字开始，看参数怎样迈出学习的第一步。', '把上一章的一个参数，组织成可以管理和训练的网络。', '网络已经成形，但要学会复杂规律，还得拥有“拐弯”的能力。', '有了表达能力，还需要一把衡量预测好坏的尺子。', '训练逻辑已经完整，现在让成千上万条数据有序进入模型。', '数据可以分批送达了，接下来跨越设备与运行模式的边界。', '梯度算出来以后，参数究竟应该怎样迈出下一步？', '把训练框架带到文字世界，先让词语拥有数值表示。', '有了词向量，还需要让每个词看见自己的上下文。', '把信息交互与特征加工组装起来，让深层网络站稳。']

def page(title, body, chapter=0):
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><meta name="description" content="跟随林默与陈教授，用十章渐进式技术小说理解 PyTorch：从 Tensor 到 Transformer。"><title>{html.escape(title)} · PyTorch 炼金术士的学徒</title><script>try{{document.documentElement.dataset.theme=localStorage.getItem('pa-theme')||'light';document.documentElement.style.setProperty('--reader-size',(Number(localStorage.getItem('pa-size'))||18)+'px')}}catch(e){{}}</script><link rel="stylesheet" href="style.css"><script src="app.js" defer></script></head>
<body data-chapter="{chapter}"><a class="skip" href="#main">跳到正文</a><div id="progress" aria-hidden="true"></div><header class="topbar"><a class="brand" href="index.html"><span class="emblem" aria-hidden="true">P<span>↗</span></span><span>炼金术士的学徒<small>A PYTORCH STORY</small></span></a><div class="controls"><button id="search-open" aria-label="搜索全书">搜索 <kbd>/</kbd></button><button id="smaller" aria-label="缩小字号">A−</button><button id="larger" aria-label="增大字号">A＋</button><button id="theme" aria-label="切换深浅色主题">◐</button></div></header>
{body}<footer><a href="index.html">PyTorch：炼金术士的学徒</a><p>故事引路，代码落地。<span>现有十章 · 第十一章待续</span></p><a href="https://github.com/wupzl/pytorch-apprentice">项目源码 ↗</a></footer>
<dialog id="search-dialog" aria-labelledby="search-title"><div class="search-top"><h2 id="search-title">在故事里寻找答案</h2><button id="search-close" aria-label="关闭搜索">关闭 ×</button></div><label for="query">关键词或代码，例如 Tensor、梯度、zero_grad</label><input id="query" type="search" autocomplete="off" placeholder="搜索全书正文…"><p id="search-status" role="status">输入关键词，搜索十章内容</p><div id="search-results"></div></dialog></body></html>'''

search = []
metadata = []
for i in range(10):
    source = (ROOT / 'chapters' / f'{i+1:02}.md').read_text(encoding='utf-8')
    math = []
    def stash(m):
        math.append({'tex':m.group(1) if m.group(1) is not None else m.group(2),'display':m.group(1) is not None})
        return f'MATHTOKEN{len(math)-1}END'
    chunks = re.split(r'(```.*?```|`[^`\n]+`)', source, flags=re.S)
    for j in range(0,len(chunks),2):
        chunks[j] = re.sub(r'\\\[(.*?)\\\]|\\\((.*?)\\\)',stash,chunks[j],flags=re.S)
    md = markdown.Markdown(extensions=['fenced_code','tables','toc','sane_lists','codehilite'], extension_configs={'toc': {'toc_depth':'2-2'}, 'codehilite':{'guess_lang':False}})
    rendered = md.convert(''.join(chunks))
    if math:
        result = subprocess.run(['node', str(ROOT / 'render-math.cjs')], input=json.dumps(math), text=True, encoding='utf-8',capture_output=True)
        if result.returncode:
            raise RuntimeError(f'Chapter {i+1}: {result.stderr}')
        for j, value in enumerate(json.loads(result.stdout)):
            rendered = rendered.replace(f'MATHTOKEN{j}END', value)
    rendered = re.sub(r'<p>(<math.*?</math>)</p>',r'\1',rendered,flags=re.S)
    minutes = max(5,round(len(source)/450))
    metadata.append({'number':i+1,'title':NAMES[i],'topics':TOPICS[i],'minutes':minutes})
    booknav = ''.join(f'<a class="{"current" if k==i else ""}" href="chapter-{k+1:02}.html" {"aria-current=page" if k==i else ""}><span>{k+1:02}</span>{NAMES[k]}</a>' for k in range(10))
    nav = (f'<a href="chapter-{i:02}.html">← 上一章<span>{NAMES[i-1]}</span></a>' if i else '<a href="index.html">← 全书目录<span>回到旅程起点</span></a>')
    nav += (f'<a href="chapter-{i+2:02}.html">下一章 →<span>{NAMES[i+1]}</span></a>' if i<9 else '<a href="index.html">本卷读毕 →<span>第十一章待续 · 返回目录</span></a>')
    body = f'''<div class="reader-layout"><aside class="sidebar"><p class="eyebrow">THE APPRENTICE’S PATH</p><nav aria-label="全书章节">{booknav}</nav><details class="local-toc" open><summary>本章线索</summary>{md.toc}</details></aside><main id="main" class="chapter-main"><div class="chapter-heading"><p class="eyebrow">CHAPTER {i+1:02} <span>/ 10 · 约 {minutes} 分钟</span></p><h1>{NAMES[i]}</h1><p class="topics">{TOPICS[i]}</p><div class="bridge">{BRIDGES[i]}</div></div><details class="mobile-toc"><summary>章节与本章目录</summary><nav aria-label="移动端章节">{booknav}</nav>{md.toc}</details><article>{rendered}</article><section class="practice"><p class="eyebrow">FROM STORY TO CODE</p><h2>把理解带回编辑器</h2><p>小片段用于解释概念，需结合本章上下文。完整示例与运行步骤放在独立页面，CPU 即可起步。</p><a href="practice.html">查看完整示例与运行说明 →</a></section><nav class="chapter-nav" aria-label="上下章">{nav}</nav></main></div>'''
    (OUT / f'chapter-{i+1:02}.html').write_text(page(NAMES[i],body,i+1),encoding='utf-8')
    # Search by section; include the body and code rather than headings alone.
    sections = re.split(r'(?m)^## ',source)
    headings = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',rendered)
    for j,section in enumerate(sections):
        text = re.sub(r'[`#*>\[\]]','',section)
        search.append({'chapter':i+1,'title':NAMES[i], 'section':html.unescape(re.sub('<[^>]+>','',headings[j-1][1])) if j and j<=len(headings) else '故事开始','url':f'chapter-{i+1:02}.html'+('#'+headings[j-1][0] if j and j<=len(headings) else ''),'text':text})

cards = ''
stages = ['01 / 建立直觉','02 / 训练成环','03 / 走向语言模型']
for i in range(10):
    if i in (0,3,7):
        cards += f'<div class="stage"><span>{stages[(0,3,7).index(i)]}</span><i></i></div>'
    cards += f'<a class="chapter-card" href="chapter-{i+1:02}.html"><span class="chapter-number">{i+1:02}</span><div><h3>{NAMES[i]}</h3><p>{TOPICS[i]}</p></div><span class="card-meta">约 {metadata[i]["minutes"]} 分钟 <b>↗</b></span></a>'
home = f'''<main id="main" class="home"><section class="hero"><div class="hero-copy"><p class="eyebrow">一部循序渐进的 PYTORCH 技术小说</p><h1>每一个模型，<br>都从<span>好奇</span>开始。</h1><p class="hero-subtitle">PyTorch：炼金术士的学徒</p><p class="intro">凌晨两点，林默敲下 <code>import torch</code>。<br>一个数字，一次求导，一座逐渐成形的 Transformer。<br>跟随他与陈教授，把原理一步步写成可以理解的代码。</p><div class="hero-actions"><a class="primary" href="chapter-01.html">翻开第一章 <span>→</span></a><a id="resume" class="resume" href="chapter-01.html" hidden>继续上次阅读 →</a></div><p class="hero-note">10 个章节 <span>·</span> 无需登录 <span>·</span> 从 Tensor 到 Transformer</p></div><div class="book-art" aria-hidden="true"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><div class="book"><p>THE APPRENTICE<br>OF PYTORCH</p><div class="book-title">炼金术士<br>的学徒<span>从数字，到智能。</span></div><div class="book-symbol">∂<span>从一次微小的变化开始</span></div><div class="book-bottom">TENSOR → TRANSFORMER<span>VOL. 01</span></div></div><span class="art-note">一行代码，打开一个世界。</span></div></section><section class="journey"><p class="eyebrow">A PATH, NOT A PILE OF APIS</p><h2>先遇见问题，再认识答案。</h2><p>不急着堆满术语。从林默的一个疑问出发，让下一步知识成为上一步困惑的答案。</p><div class="path"><span>数字与梯度</span><b>→</b><span>网络与训练</span><b>→</b><span>文字与注意力</span><b>→</b><span>Transformer</span></div></section><section class="catalog"><div class="catalog-heading"><div><p class="eyebrow">TEN CHAPTERS · ONE CONNECTED STORY</p><h2>学徒的十次进阶</h2></div><a href="practice.html">准备好运行代码了吗？ ↗</a></div><div class="chapter-list">{cards}</div><div class="coming"><span>11</span><div><h3>预测下一个词</h3><p>从 Transformer 到 GPT 的诞生 · 待续</p></div><span class="tag">尚未发布</span></div></section><section class="reading-note"><h2>怎样读这本书</h2><p>建议按章节顺序阅读，跟着问题往前走。你不需要预先掌握深度学习，但最好了解基础 Python；前两章会从最小例子搭起训练直觉。</p><p>代码片段帮助理解概念，完整示例用于动手验证。阅读位置只保存在当前浏览器中。<a href="editorial.html">查看内容整理说明 →</a></p></section></main>'''
(OUT/'index.html').write_text(page('全书目录',home),encoding='utf-8')
practice = '''# 从故事到编辑器

建议使用 Python 3.10 或更高版本。下面的示例不需要下载数据集，默认使用 CPU；检测到 CUDA 时，分类示例会自动使用 GPU。

## 第一步：准备环境

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\\Scripts\\Activate.ps1
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

CPU 安装与运行：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python examples/train.py
python examples/transformer.py
```

先从[项目仓库](https://github.com/wupzl/pytorch-apprentice)下载或克隆完整项目，并在项目根目录执行上述命令。GPU 安装请使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)生成适合设备的命令。

## 示例一：一条完整训练链路

对应第 1–7 章：用二维合成数据训练小型分类网络，包含 Dataset、DataLoader、训练模式、清梯度、前向、Loss、反向和参数更新。

[下载 train.py](examples/train.py)

```python
'''+(ROOT/'examples/train.py').read_text(encoding='utf-8')+'''\n```

## 示例二：可以运行的 Transformer Block

对应第 9–10 章：支持双向和因果模式，包含反向传播与“未来输入不影响过去输出”的因果性检查。运行后应输出 torch.Size([2, 10, 128]) 和检查通过提示。它仍不是完整 GPT。

[下载 transformer.py](examples/transformer.py)

```python
'''+(ROOT/'examples/transformer.py').read_text(encoding='utf-8')+'\n```\n'
for name,title,text in [('practice','运行代码',practice),('editorial','内容整理说明',(ROOT/'EDITORIAL.md').read_text(encoding='utf-8'))]:
    body = markdown.markdown(text,extensions=['fenced_code','tables','codehilite'],extension_configs={'codehilite':{'guess_lang':False}})
    (OUT/f'{name}.html').write_text(page(title,f'<main id="main" class="standalone"><a href="index.html">← 返回全书目录</a><article>{body}</article></main>'),encoding='utf-8')
(OUT/'search.json').write_text(json.dumps(search,ensure_ascii=False),encoding='utf-8')
(OUT/'chapters.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'syntax.css').write_text(HtmlFormatter().get_style_defs('.codehilite'),encoding='utf-8')
for name in ['style.css','app.js']:
    shutil.copyfile(ROOT/'assets'/name, OUT/name)
(OUT/'examples').mkdir(exist_ok=True)
for p in (ROOT/'examples').glob('*.py'):
    shutil.copyfile(p,OUT/'examples'/p.name)
(OUT/'.nojekyll').touch()
print('Built 13 pages, 10 chapters, section search and local MathML formulas.')
