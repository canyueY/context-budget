# -*- coding: utf-8 -*-
"""对话预算裁剪测试。

Copyright (C) 2026 canyueY <https://github.com/canyueY>
SPDX-License-Identifier: AGPL-3.0-or-later

重点覆盖那些"裁错了不会报错、但会让模型答非所问"的地方：
system 有没有被误丢、问答有没有被拆散、时间播报有没有堆积。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from context_budget import (  # noqa: E402
    MAX_TOKEN_ESTIMATE,
    BudgetSettings,
    budget_settings,
    coerce_messages,
    estimate_messages_tokens,
    estimate_tokens,
    is_time_system,
    split_messages,
    trim,
    trim_auxiliary,
)


def dialog(n: int, *, size: int = 4) -> list[dict]:
    """造 n 轮对话（每轮一问一答）。"""
    out: list[dict] = []
    for i in range(n):
        out.append({"role": "user", "content": "u" * size})
        out.append({"role": "assistant", "content": "a" * size})
    return out


# ---------------------------------------------------------------------------
# token 估算
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0  # type: ignore[arg-type]

    def test_roughly_two_chars_per_token(self):
        assert estimate_tokens("a" * 100) == 50

    def test_single_char_still_counts_one(self):
        # 不能返回 0，否则调用方会把"有内容"当成"没内容"
        assert estimate_tokens("a") == 1

    def test_capped(self):
        assert estimate_tokens("a" * (MAX_TOKEN_ESTIMATE * 4)) == MAX_TOKEN_ESTIMATE

    def test_messages_include_per_message_overhead(self):
        msgs = [{"role": "user", "content": "aaaa"}]
        # 4 字符 -> 2 token，加固定开销 4
        assert estimate_messages_tokens(msgs) == 6

    def test_messages_handles_multimodal_blocks(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a" * 20},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        assert estimate_messages_tokens(msgs) > 10

    def test_custom_counter_used(self):
        msgs = [{"role": "user", "content": "x"}]
        assert estimate_messages_tokens(msgs, counter=lambda s: 100) == 104

    def test_non_string_content_does_not_crash(self):
        assert estimate_messages_tokens([{"role": "user", "content": 12345}]) > 0

    def test_missing_content_ok(self):
        assert estimate_messages_tokens([{"role": "user"}]) == 4


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------


class TestCoerceMessages:
    def test_none_becomes_empty_string(self):
        out = coerce_messages([{"role": "user", "content": None}])
        assert out == [{"role": "user", "content": ""}]

    def test_missing_content_becomes_empty_string(self):
        assert coerce_messages([{"role": "system"}])[0]["content"] == ""

    def test_non_string_content_stringified(self):
        out = coerce_messages([{"role": "user", "content": 42}])
        assert out[0]["content"] == "42"

    def test_missing_role_defaults_to_user(self):
        assert coerce_messages([{"content": "hi"}])[0]["role"] == "user"

    def test_non_mapping_dropped(self):
        out = coerce_messages(["nope", None, 42, {"role": "user", "content": "ok"}])
        assert len(out) == 1

    def test_none_input(self):
        assert coerce_messages(None) == []

    def test_extra_keys_dropped(self):
        out = coerce_messages([{"role": "user", "content": "x", "junk": 1}])
        assert out == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# 时间 system 识别
# ---------------------------------------------------------------------------


class TestTimeSystem:
    @pytest.mark.parametrize(
        "text", ["现在是14点30分", "现在是 9 点 5 分", "现在是23点59分"]
    )
    def test_matches(self, text):
        assert is_time_system({"role": "system", "content": text}) is True

    @pytest.mark.parametrize(
        "text",
        ["现在几点了", "现在是下午", "你是丛雨", "现在是14点", "现在是14点30分。"],
    )
    def test_non_matches(self, text):
        assert is_time_system({"role": "system", "content": text}) is False

    def test_only_system_role(self):
        assert is_time_system({"role": "user", "content": "现在是14点30分"}) is False

    def test_whitespace_tolerated(self):
        assert is_time_system({"role": "system", "content": "  现在是14点30分  "}) is True


class TestSplitMessages:
    def test_three_way_split(self):
        h = [
            {"role": "system", "content": "你是丛雨"},
            {"role": "system", "content": "现在是14点30分"},
            *dialog(2),
        ]
        systems, times, d = split_messages(h)
        assert len(systems) == 1 and systems[0]["content"] == "你是丛雨"
        assert len(times) == 1
        assert len(d) == 4

    def test_only_latest_time_kept(self):
        h = [
            {"role": "system", "content": "现在是1点0分"},
            {"role": "system", "content": "现在是2点0分"},
            {"role": "system", "content": "现在是3点0分"},
        ]
        _, times, _ = split_messages(h)
        assert len(times) == 1
        assert times[0]["content"] == "现在是3点0分"

    def test_empty_input(self):
        assert split_messages(None) == ([], [], [])


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class TestBudgetSettings:
    def test_defaults(self):
        st = BudgetSettings()
        assert st.max_history_turns == 50
        assert st.max_context_tokens == 48000
        assert st.max_aux_history_turns == 24

    def test_reads_chat_block(self):
        st = budget_settings(
            {"chat": {"max_history_turns": 12, "max_context_tokens": 8000}}
        )
        assert st.max_history_turns == 12
        assert st.max_context_tokens == 8000

    def test_missing_block_uses_defaults(self):
        for cfg in (None, {}, {"chat": None}, {"chat": "nope"}):
            assert budget_settings(cfg).max_history_turns == 50

    def test_out_of_range_clamped_not_raised(self):
        # 配置写错数量级时夹紧，而不是让整个程序起不来
        st = budget_settings({"chat": {"max_history_turns": 99999}})
        assert st.max_history_turns == 200
        st2 = budget_settings({"chat": {"max_history_turns": 1}})
        assert st2.max_history_turns == 4

    def test_garbage_falls_back(self):
        st = budget_settings({"chat": {"max_history_turns": "abc"}})
        assert st.max_history_turns == 50

    def test_negative_raises_directly(self):
        with pytest.raises(ValueError):
            BudgetSettings(max_history_turns=-1)
        with pytest.raises(ValueError):
            BudgetSettings(max_context_tokens=0)


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------


class TestTrimTurns:
    def test_keeps_last_n_turns(self):
        h = dialog(10)
        out = trim(h, max_turns=3, max_tokens=10**9)
        assert len(out) == 6
        assert out[-2:] == h[-2:]

    def test_under_limit_untouched(self):
        h = dialog(3)
        assert trim(h, max_turns=10, max_tokens=10**9) == h

    def test_empty_input(self):
        assert trim(None) == []
        assert trim([]) == []

    def test_zero_turns_keeps_last_turn_not_nothing(self):
        # max_turns=0 落到「永不丢最后两条」的底线：只剩一轮也好过空列表
        h = dialog(5)
        out = trim(h, max_turns=0, max_tokens=10**9)
        assert out == h[-2:]


class TestTrimTokens:
    def test_token_budget_drops_oldest_pairs(self):
        # 每轮 2 条 * 4 字符 = 每轮约 12 token（含开销）
        h = dialog(20, size=40)
        out = trim(h, max_turns=100, max_tokens=200)
        assert estimate_messages_tokens(out) <= 200
        assert len(out) < len(h)

    def test_never_drops_last_turn(self):
        # 预算小到装不下也要留下最后一轮，否则模型不知道"我刚问了什么"
        h = dialog(20, size=4000)
        out = trim(h, max_turns=100, max_tokens=10)
        assert len(out) == 2
        assert out == h[-2:]

    def test_drops_in_pairs_not_singles(self):
        h = dialog(10, size=40)
        out = trim(h, max_turns=100, max_tokens=200)
        assert len(out) % 2 == 0

    def test_first_kept_message_is_a_user_turn(self):
        # 从中间截断会让模型看到"助手回答了但用户没问过"
        h = dialog(20, size=40)
        out = trim(h, max_turns=100, max_tokens=200)
        assert out[0]["role"] == "user"

    def test_custom_counter_respected(self):
        h = dialog(10, size=4)
        out = trim(h, max_turns=100, max_tokens=100, counter=lambda s: 10**6)
        assert len(out) == 2


class TestTrimPreservesSystem:
    def test_system_always_kept(self):
        h = [{"role": "system", "content": "你是丛雨，说话要简短。"}] + dialog(30, size=40)
        out = trim(h, max_turns=2, max_tokens=100)
        assert out[0]["content"] == "你是丛雨，说话要简短。"

    def test_system_survives_tiny_budget(self):
        h = [{"role": "system", "content": "s" * 5000}] + dialog(5)
        out = trim(h, max_turns=1, max_tokens=1)
        assert any(m["role"] == "system" for m in out)

    def test_multiple_systems_kept_in_order(self):
        h = [
            {"role": "system", "content": "人设A"},
            {"role": "system", "content": "规则B"},
            *dialog(5),
        ]
        out = trim(h, max_turns=1, max_tokens=10**9)
        assert [m["content"] for m in out[:2]] == ["人设A", "规则B"]

    def test_only_latest_time_kept(self):
        h = [
            {"role": "system", "content": "现在是1点0分"},
            {"role": "system", "content": "人设"},
            {"role": "system", "content": "现在是9点0分"},
            *dialog(3),
        ]
        out = trim(h, max_turns=10, max_tokens=10**9)
        times = [m["content"] for m in out if is_time_system(m)]
        assert times == ["现在是9点0分"]

    def test_output_order_is_system_then_time_then_dialog(self):
        h = [
            {"role": "system", "content": "人设"},
            {"role": "system", "content": "现在是9点0分"},
            *dialog(3),
        ]
        out = trim(h, max_turns=10, max_tokens=10**9)
        roles = [m["role"] for m in out]
        assert roles == ["system", "system", "user", "assistant", "user", "assistant",
                         "user", "assistant"]

    def test_time_message_dropped_when_no_time_present(self):
        h = [{"role": "system", "content": "人设"}, *dialog(2)]
        out = trim(h, max_turns=10, max_tokens=10**9)
        assert len(out) == 5

    def test_system_messages_are_not_counted_as_dialog_turns(self):
        h = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}] + dialog(2)
        out = trim(h, max_turns=2, max_tokens=10**9)
        assert len(out) == 6


class TestTrimAuxiliary:
    def test_keeps_only_first_system(self):
        h = [
            {"role": "system", "content": "人设"},
            {"role": "system", "content": "规则"},
            *dialog(3),
        ]
        out = trim_auxiliary(h, max_turns=10)
        systems = [m for m in out if m["role"] == "system"]
        assert len(systems) == 1
        assert systems[0]["content"] == "人设"

    def test_time_system_dropped_entirely(self):
        h = [{"role": "system", "content": "现在是9点0分"}, *dialog(2)]
        out = trim_auxiliary(h, max_turns=10)
        assert not any(is_time_system(m) for m in out)

    def test_turn_limit_applied(self):
        out = trim_auxiliary(dialog(20), max_turns=3)
        assert len(out) == 6

    def test_uses_settings_default(self):
        st = BudgetSettings(max_aux_history_turns=2)
        out = trim_auxiliary(dialog(20), settings=st)
        assert len(out) == 4

    def test_empty(self):
        assert trim_auxiliary(None) == []

    def test_under_limit_untouched(self):
        h = dialog(2)
        assert trim_auxiliary(h, max_turns=10) == h


class TestCompatibilityAlias:
    def test_alias_same_behaviour(self):
        from context_budget import trim_conversation_history

        h = dialog(10, size=40)
        assert trim_conversation_history(h, max_turns=2, max_tokens=10**9) == trim(
            h, max_turns=2, max_tokens=10**9
        )
