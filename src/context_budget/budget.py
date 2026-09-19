# -*- coding: utf-8 -*-
"""对话上下文预算：轮数 / token 上限与历史裁剪。

Copyright (C) 2026 canyueY <https://github.com/canyueY>
SPDX-License-Identifier: AGPL-3.0-or-later

设计取舍
--------
**为什么不绑定 tokenizer。** 这个包的职责是"决定丢哪些消息"，不是"精确算出
token 数"——后者各家的分词器不同，绑进来就把一个纯逻辑模块变成了重依赖。
所以默认用 :func:`estimate_tokens` 的字符数近似（中英混合约 2 字符 ≈ 1 token），
需要精确时可以传 ``counter=`` 换成自己的实现。

**为什么按「对」丢而不是按「条」丢。** 对话历史里 user/assistant 是成对的；
从中间截断会让模型看到"助手回答了但用户没问过"这种上下文。所以裁剪以
两条（一问一答）为单位。

**为什么保留 system。** system 通常是人设与规则，丢了整个回答的基调就变了。
其中只有"当前时间"这类**临时** system 会被特殊处理——保留最新一条、丢掉旧的，
否则多轮对话里会堆积几十条时间戳，既浪费预算又让模型看到矛盾的"现在几点"。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "BudgetSettings",
    "MAX_TOKEN_ESTIMATE",
    "TIME_SYSTEM_RE",
    "budget_settings",
    "coerce_messages",
    "estimate_messages_tokens",
    "estimate_tokens",
    "is_time_system",
    "split_messages",
    "trim",
    "trim_auxiliary",
    "trim_conversation_history",
]

#: 默认的"当前时间"播报格式，用来识别**临时** system 消息。
#: 形如 ``现在是14点30分``。多轮对话里每轮都会插一条，所以只需要最新的。
TIME_SYSTEM_RE = re.compile(r"^现在是.+点.+分$")

#: 单条消息的 token 估算上限，防止一个超长粘贴把估算撑爆。
MAX_TOKEN_ESTIMATE = 1 << 22

#: 每条消息的固定开销（role 字段、模板分隔符等）。各家 chat template 不同，
#: 取 4 是个常见的经验值；只影响估算的保守程度。
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数（中英混合约 2 字符 ≈ 1 token）。

    这是**故意**保守的近似：宁可早一点开始丢历史，也不要因为低估而把
    请求撑爆上下文窗口。要精确请用 :func:`trim` 的 ``counter`` 参数。
    """
    if not text:
        return 0
    return max(1, min(len(text) // 2, MAX_TOKEN_ESTIMATE))


def _text_of(content: Any) -> str:
    """把各种 content 形状摊平成一段文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多模态：把文本块拼起来，非文本块按字符串估
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


def estimate_messages_tokens(
    messages: Sequence[Mapping[str, Any]],
    *,
    counter: Callable[[str], int] | None = None,
) -> int:
    """估算一组消息的总 token 数（含每条消息的固定开销）。"""
    count = counter or estimate_tokens
    total = 0
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, Mapping) else ""
        if isinstance(content, list):
            total += sum(
                count(_text_of(b) if isinstance(b, Mapping) else str(b)) for b in content
            )
        else:
            total += count(_text_of(content))
        total += _PER_MESSAGE_OVERHEAD
    return total


def coerce_messages(messages: Iterable[Any] | None) -> list[dict[str, Any]]:
    """规范化消息列表：丢掉非映射、``content=None`` 变空串、role 缺失补 ``user``。

    为什么需要：本地 ``apply_chat_template`` 之类要求 ``content`` 是 str，
    ``None`` 会直接 ``TypeError``。在进入模型之前把形状统一，比在每个调用点
    各自防御要省事。
    """
    out: list[dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, Mapping):
            continue
        role = str(m.get("role") or "user")
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def is_time_system(msg: Mapping[str, Any]) -> bool:
    """这条 system 消息是不是"当前时间"播报。"""
    if msg.get("role") != "system":
        return False
    return bool(TIME_SYSTEM_RE.match(str(msg.get("content") or "").strip()))


def split_messages(
    history: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """拆成 ``(常驻 system, 最新一条临时时间 system, 对话)`` 三块。

    裁剪策略就是在这三块上分别做的，所以单独抽出来便于测试与复用。
    时间 system 只留最新一条——旧的既浪费预算，又让模型看到矛盾的"现在几点"。
    """
    msgs = coerce_messages([m for m in (history or []) if isinstance(m, Mapping)])
    times = [m for m in msgs if is_time_system(m)]
    systems = [m for m in msgs if m["role"] == "system" and not is_time_system(m)]
    dialog = [m for m in msgs if m["role"] != "system"]
    return systems, times[-1:], dialog


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class BudgetSettings:
    """预算设置。所有字段都有可直接使用的默认值。"""

    #: 最多保留多少**轮**（一轮 = 一问一答 = 两条消息）
    max_history_turns: int = 50
    #: 上限，超出就继续从最早的一对开始丢
    max_context_tokens: int = 48000
    #: 辅助链路（情感判定 / 立绘选择之类）用的更小轮数
    max_aux_history_turns: int = 24
    #: 生成上限，**不参与裁剪**，只是集中放在这里方便一并下发
    max_new_tokens: int = 2048
    reply_max_tokens: int = 1024
    #: 天气等特定链路的窗口
    weather_history_messages: int = 16

    def __post_init__(self) -> None:
        if self.max_history_turns < 0:
            raise ValueError(f"max_history_turns 不能为负：{self.max_history_turns}")
        if self.max_context_tokens < 1:
            raise ValueError(f"max_context_tokens 至少为 1：{self.max_context_tokens}")


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    """读一个整数并夹到合理区间。配置写错时**夹紧**而不是报错。"""
    try:
        val = int(value)
    except (TypeError, ValueError):
        val = default
    return max(lo, min(val, hi))


#: 各字段的安全区间：低于下界会退化（比如 0 轮等于没有上下文），
#: 高于上界通常是配置写错了数量级。
_BOUNDS: dict[str, tuple[int, int, int]] = {
    # field: (default, lo, hi)
    "max_history_turns": (50, 4, 200),
    "max_context_tokens": (48000, 2048, 128000),
    "max_aux_history_turns": (24, 4, 100),
    "max_new_tokens": (2048, 256, 16384),
    "reply_max_tokens": (1024, 128, 8192),
    "weather_history_messages": (16, 4, 64),
}


def budget_settings(cfg: Mapping[str, Any] | None = None) -> BudgetSettings:
    """从 ``{"chat": {...}}`` 读预算设置，缺项用默认、越界夹紧。

    只认 ``chat`` 这一个位置是有意的：预算必须来自一个明确的地方，
    而不是"在配置树里到处找找看"。
    """
    block: Mapping[str, Any] = {}
    if isinstance(cfg, Mapping):
        candidate = cfg.get("chat")
        if isinstance(candidate, Mapping):
            block = candidate

    values = {
        name: _clamp_int(block.get(name, default), default, lo, hi)
        for name, (default, lo, hi) in _BOUNDS.items()
    }
    return BudgetSettings(**values)


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------


def _pair_count(messages: Sequence[Any]) -> int:
    return len(messages) // 2


def trim(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    settings: BudgetSettings | None = None,
    max_turns: int | None = None,
    max_tokens: int | None = None,
    counter: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    """裁剪对话历史，返回 ``[常驻 system..., 最新时间 system?, ...对话]``。

    两步，顺序重要：

    1. **按轮数截断**——只保留最近 ``max_turns`` 轮。这一步便宜，先做。
    2. **按 token 预算继续丢**——从最早的一对开始，直到装得下为止。

    永不丢掉常驻 system，也永不丢掉最后两条 dialog：只剩一轮也要能回答
    "我刚才问了什么"。所以 ``max_turns=0`` 的结果是**最后一轮**，不是空列表。

    :param max_turns: 覆盖 ``settings.max_history_turns``
    :param max_tokens: 覆盖 ``settings.max_context_tokens``
    :param counter: 自定义 token 计数（默认 :func:`estimate_tokens`）
    """
    st = settings or BudgetSettings()
    turn_limit = st.max_history_turns if max_turns is None else max(0, int(max_turns))
    token_limit = (
        st.max_context_tokens if max_tokens is None else max(1, int(max_tokens))
    )

    systems, times, dialog = split_messages(history)

    # 保留上限按「条」算（一轮 = 两条），且**永不低于 2**：留最后一轮是底线。
    # 注意不能写成 `if turn_limit > 0:` —— 那会让 max_turns=0 变成"不裁剪"，
    # 与"最多 0 轮"的语义正好相反。
    keep = max(2, turn_limit * 2)
    if len(dialog) > keep:
        dialog = dialog[-keep:]

    # 就地对半丢。保留最后两条的底线靠 while 条件保证。
    while _pair_count(dialog) > 1:
        trial = systems + times + dialog
        if estimate_messages_tokens(trial, counter=counter) <= token_limit:
            break
        dialog = dialog[2:]

    return systems + times + dialog


def trim_auxiliary(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    settings: BudgetSettings | None = None,
    max_turns: int | None = None,
) -> list[dict[str, Any]]:
    """辅助链路的裁剪：只留**第一条** system 与最近若干轮。

    情感判定 / 立绘选择这类链路不需要完整人设，也不需要时间播报——
    它们只要"最近发生了什么"。所以比主链路更狠：system 只留第一条，
    时间 system 全部丢掉。
    """
    st = settings or BudgetSettings()
    limit = st.max_aux_history_turns if max_turns is None else max(0, int(max_turns))
    msgs = coerce_messages([m for m in (history or []) if isinstance(m, Mapping)])
    # 时间播报先剔掉再看第一条 system：否则「现在是14点30分」会被当成
    # 人设留下来 —— 那既不是人设，也和 docstring 承诺的行为相反。
    system = [m for m in msgs if m["role"] == "system" and not is_time_system(m)][:1]
    dialog = [m for m in msgs if m["role"] != "system"]
    if limit > 0 and len(dialog) > limit * 2:
        dialog = dialog[-(limit * 2) :]
    return system + dialog


# ---------------------------------------------------------------------------
# 兼容别名
# ---------------------------------------------------------------------------

#: 原项目里的名字。语义与 :func:`trim` 完全一致，保留是为了让调用方
#: 换 import 时不用改函数名。
trim_conversation_history = trim
