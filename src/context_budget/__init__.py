# -*- coding: utf-8 -*-
"""context-budget —— 把对话历史裁到 token 预算内，保留人设与最近几轮。

Copyright (C) 2026 canyueY <https://github.com/canyueY>
SPDX-License-Identifier: AGPL-3.0-or-later

零依赖、不绑定 tokenizer。职责只有一个：**决定丢哪些消息**。

快速上手::

    from context_budget import BudgetSettings, trim

    history = trim(raw_history, settings=BudgetSettings(max_context_tokens=32000))
    reply = model.chat(history)
"""
from __future__ import annotations

from .budget import (
    MAX_TOKEN_ESTIMATE,
    TIME_SYSTEM_RE,
    BudgetSettings,
    budget_settings,
    coerce_messages,
    estimate_messages_tokens,
    estimate_tokens,
    is_time_system,
    split_messages,
    trim,
    trim_auxiliary,
    trim_conversation_history,
)

__version__ = "0.1.0"

__all__ = [
    "MAX_TOKEN_ESTIMATE",
    "TIME_SYSTEM_RE",
    "BudgetSettings",
    "__version__",
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
