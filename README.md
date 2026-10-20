# PyTorch：炼金术士的学徒

十章小说式 PyTorch 入门读物，从 Tensor 到 Transformer Block。现有十章已发布，第十一章待续。

- 网站：https://wupzl.github.io/pytorch-apprentice/
- 手机与桌面阅读，全书正文搜索、阅读位置记忆、字号调整、深浅色模式、代码高亮与复制。
- 纯静态站点，无账号与数据库；正文、高亮、公式不依赖外部 CDN。

## 本机预览

```bash
python -m pip install -r requirements.txt
python build.py
python -m http.server 4174 --directory docs
```

打开 http://localhost:4174 。构建还需要 Node.js，用于通过本地 KaTeX 将公式转换为 MathML。

## 内容维护

编辑 chapters/ 下的 Markdown，再运行 python build.py。请勿在日常构建中运行 prepare.py：它是一次性导入工具，会从原稿重新生成章节。

source/pytorch-original.md 完整保存原稿，EDITORIAL.md 记录整理与技术校正。examples/ 包含两份独立运行示例，详见网站“运行代码”页。PyTorch 是运行示例的独立依赖，不是生成或访问网站的依赖。

## 发布

GitHub Pages 使用 main 分支的 /docs 目录。所有内部资源使用相对路径，支持仓库子路径部署。提交内容变更时须同时提交构建后的 docs/。

## 隐私与限制

进度、主题、字号只存储在当前浏览器，不跨设备同步。网页不执行 Python。概念演示不代表模型对未知数据的泛化效果。

KaTeX 保留上游许可证，见 vendor/KaTeX-LICENSE.txt。尚未为原创正文与项目整体指定开源许可。
