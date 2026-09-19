# context-budget

把 LLM 对话历史裁到 token 预算内：**保留人设，丢掉最旧的轮次**。零依赖，不绑定 tokenizer。

```python
from context_budget import BudgetSettings, trim

history = trim(raw_history, settings=BudgetSettings(max_context_tokens=32000))
reply = model.chat(history)
```

## 它解决什么问题

上下文窗口是有限的，而对话历史只增不减。朴素的 `history[-20:]` 有两个毛病：
按**条**截断会把一问一答拆散（模型看到"助手回答了但用户没问过"），
而按固定条数也管不住单条超长消息。`context-budget` 按**轮**裁、按**预算**裁，
并且明确保证哪些东西永远不丢。

## 裁剪规则

输入被拆成三块，分别处理：

| 块 | 处理 |
| --- | --- |
| 常驻 `system`（人设、规则） | **永不丢弃** |
| 临时 `system`（"现在是14点30分"） | 只保留**最新一条** |
| 对话（user/assistant） | 先按轮数截断，再按 token 预算从最早的**一对**开始丢 |

两条底线：

- **最后一轮永远保留。** 预算小到装不下也留下最后一问一答 —— 否则模型不知道
  "我刚才问了什么"，会答非所问。所以 `max_turns=0` 的结果是最后一轮，不是空列表。
- **按对丢，不按条丢。** 从中间截断会让历史出现"无源之答"。

为什么不丢 `system`：那通常是人设与规则，丢了整个回答的基调就变了。
为什么时间播报要特殊处理：多轮对话里每轮插一条，不清理会堆积几十条，
既浪费预算又让模型看到互相矛盾的"现在几点"。

## 安装

```bash
pip install git+https://github.com/canyueY/context-budget.git
# 或从源码
git clone https://github.com/canyueY/context-budget.git
cd context-budget && pip install -e ".[dev]"
```

> 暂未发布到 PyPI。

## 为什么不绑定 tokenizer

本包的职责是「**决定丢哪些消息**」，不是「精确算出 token 数」——后者各家的分词器
不同，绑进来就把一个纯逻辑模块变成了重依赖。默认用字符数近似
（中英混合约 2 字符 ≈ 1 token），这是**故意保守**的：

> 宁可早一点开始丢历史，也不要因为低估而把请求撑爆上下文窗口。

需要精确时换成自己的计数器：

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
history = trim(raw, settings=st, counter=lambda s: len(enc.encode(s)))
```

## API

```python
from context_budget import (
    BudgetSettings, budget_settings,   # 配置
    trim, trim_auxiliary,               # 裁剪
    estimate_tokens, estimate_messages_tokens,
    coerce_messages, split_messages, is_time_system,
)

# 从 {"chat": {...}} 读配置；缺项用默认，越界夹紧而不是报错
st = budget_settings(config)

# 主链路：保留人设 + 时间 + 最近若干轮
trim(history, settings=st)
trim(history, max_turns=8, max_tokens=16000)   # 也可以临时覆盖

# 辅助链路（情感判定、立绘选择）：只留第一条 system + 最近几轮，
# 时间播报全部丢掉 —— 这些链路只要"最近发生了什么"
trim_auxiliary(history, settings=st)

# 规范化：content=None 变空串、role 缺失补 user、丢掉非映射
coerce_messages(raw_history)
```

`budget_settings` 的越界处理是**夹紧**（clamp），不是抛异常：

```python
budget_settings({"chat": {"max_history_turns": 99999}}).max_history_turns  # -> 200
budget_settings({"chat": {"max_history_turns": 1}}).max_history_turns      # -> 4
```

配置写错数量级时最多让行为不如预期，不会让程序起不来。但直接构造
`BudgetSettings(max_context_tokens=0)` 会 `ValueError` —— 那是程序员错误，
不是配置错误。

## 时间播报的识别

默认认 `^现在是.+点.+分$`（形如 `现在是14点30分`）。如果你的项目用了别的格式，
改 `context_budget.budget.TIME_SYSTEM_RE` 即可 —— 单独拎出来就是为了好改。

## 授权

**AGPL-3.0-or-later**，见 [LICENSE](LICENSE)。

本包从作者自己的桌宠项目 [MurasamePet](https://github.com/canyueY/MurasamePet)
（AGPL-3.0）中抽出。代码是那个项目自己新增的部分，但既然来自 AGPL 项目，
这里就继续沿用 AGPL，不做改许可。

## 来源与开发位置

> **本仓库是镜像；权威源码在 MurasamePet 仓库里。**
>
> 开发位置：`MurasamePet/packages/context-budget/`。
> 桌宠通过 **path 依赖**直接使用它 —— 不再保留任何副本，改这里立即可见。
> 这个独立仓库用于对外发布与展示，内容由 Monorepo 同步而来。
>
> **为什么不反过来（本仓库为源、桌宠用 `git` 依赖）？**
> 开发机上网关受限：`github.com` 必须走本地代理，而 `uv` 拉 git 依赖时
> 用不上该代理（实测 `git fetch` 失败）。path 依赖离线可用、不受代理开关影响。

## 测试

```bash
uv run --python 3.10 --extra dev python -m pytest tests -q
```

58 个用例，零依赖。重点覆盖那些"裁错了不会报错、但会让模型答非所问"的地方：
`system` 有没有被误丢、问答有没有被拆散、时间播报有没有堆积、
极小预算下最后一轮还在不在。
