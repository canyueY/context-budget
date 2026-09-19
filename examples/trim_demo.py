# -*- coding: utf-8 -*-
"""最小可运行示例：看历史是怎么被裁掉的。

Copyright (C) 2026 canyueY <https://github.com/canyueY>
SPDX-License-Identifier: AGPL-3.0-or-later

直接跑：

    python examples/trim_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_budget import BudgetSettings, estimate_messages_tokens, trim  # noqa: E402


def build_history(turns: int, *, size: int = 60) -> list[dict]:
    """造一段带人设、堆了好几轮时间播报的对话历史。"""
    history: list[dict] = [{"role": "system", "content": "你是丛雨，说话简短，别用敬语。"}]
    for i in range(1, turns + 1):
        history.append({"role": "system", "content": f"现在是{i}点0分"})
        history.append({"role": "user", "content": f"第{i}个问题。" + "问" * size})
        history.append({"role": "assistant", "content": f"第{i}个回答。" + "答" * size})
    return history


def show(label: str, messages: list[dict]) -> None:
    roles = {"system": "SYS", "user": "USR", "assistant": "AST"}
    times = sum(1 for m in messages if m["role"] == "system" and "现在是" in m["content"])
    print(f"\n{label}")
    print(f"  条数={len(messages):<4} 估算 token={estimate_messages_tokens(messages):<6} "
          f"时间播报={times}")
    kept = [f"{roles.get(m['role'], m['role'])}:{m['content'][:12]}" for m in messages]
    if len(kept) > 8:
        print(f"  {' | '.join(kept[:3])}  ...  {' | '.join(kept[-3:])}")
    else:
        print(f"  {' | '.join(kept)}")


def main() -> int:
    history = build_history(12)
    show("原始历史（12 轮，每轮都插了时间播报）", history)

    st = BudgetSettings(max_history_turns=4, max_context_tokens=10**9)
    show(f"按轮数裁（max_history_turns={st.max_history_turns}）",
         trim(history, settings=st))

    tight = BudgetSettings(max_history_turns=50, max_context_tokens=1500)
    out = trim(history, settings=tight)
    show(f"按预算裁（max_context_tokens={tight.max_context_tokens}）", out)

    tiny = trim(history, max_turns=0, max_tokens=1)
    show("极小预算（max_tokens=1）—— 最后一轮仍在，这是底线", tiny)

    print("\n" + "=" * 70)
    print("要点：")
    print("  · 人设那条 system 在任何裁剪下都没丢")
    print("  · 时间播报只剩最新一条（12 条 -> 1 条）")
    print("  · 对话按「一问一答」成对丢弃，不会出现无源之答")
    print("  · 预算再小也留最后一轮，否则模型不知道你刚问了什么")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
