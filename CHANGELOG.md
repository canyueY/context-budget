# Changelog

本文件记录 context-budget 的对外变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-09-19

首次发布。从 [MurasamePet](https://github.com/canyueY/MurasamePet) 的
`Murasame/context_memory.py` 抽出并补上 API 与测试。

### Added

- `trim()`：两步裁剪（先按轮数，再按 token 预算从最早的一对开始丢）。
- `BudgetSettings` / `budget_settings(cfg)`：预算即数据；越界**夹紧**而非报错。
- `trim_auxiliary()`：辅助链路的更狠裁剪（只留第一条 system）。
- `estimate_tokens` / `estimate_messages_tokens`：字符数近似，含每条消息的固定开销。
- `coerce_messages()`：规范化消息形状（`content=None` 变空串、role 缺失补 `user`）。
- `split_messages()` / `is_time_system()`：把历史拆成「常驻 system / 最新时间 system /
  对话」三块，单独可测。
- `TIME_SYSTEM_RE`：时间播报的识别正则，公开可改。
- `counter=` 参数：可换成精确 tokenizer（如 tiktoken），默认不引入任何依赖。

### Changed

相对于抽取前的实现：

- **`trim_auxiliary` 现在真的会丢掉时间播报。** 原实现取"第一条 system"时
  没有排除时间消息，于是「现在是14点30分」会被当成第一条 system 留下来 ——
  既不是人设，也与该函数"辅助链路只要最近发生了什么"的意图相反。
- **`max_turns=0` 不再等于"不裁剪"。** 原实现用 `if turn_limit > 0:` 守卫截断，
  于是 0 被当成"没有限制"，与实际语义正好相反；现在 0 落到"只留最后一轮"的底线。
- `trim()` 的返回值顺序固定为 `[常驻 system..., 最新时间 system?, ...对话]`
  （原实现的顺序依赖输入顺序，现在显式保证）。
- `chat_memory_settings()` 改名为 `budget_settings()`，并去掉 `cfg` 必须为
  `dict` 的限制（`Mapping` 即可）；原名字不再保留，因为 `context_memory` 是
  项目内部模块名，不适合作为公开 API。
- `trim_conversation_history()` 保留为 `trim()` 的别名，便于调用方换 import 时
  不改函数名。

### Fixed

- 配置里的整数越界不再静默产生荒谬行为，而是夹到安全区间
  （`max_history_turns` 4..200、`max_context_tokens` 2048..128000 等）。

[0.1.0]: https://github.com/canyueY/context-budget/releases/tag/v0.1.0
