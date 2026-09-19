# logs/ — AI Coding 日志目录

存放开发中与 AI 工具的真实对话日志，和作品代码一并提交。

当前已提交 `bilibilidev` 的 Codex CLI 只读审计会话。采集兼容过程、隐私边界和校验方法见 [`docs/AI_Coding日志说明.md`](../docs/AI_Coding日志说明.md)。

## 目录结构

```text
logs/
└── bilibilidev/                 # 本次日志对应的 GitHub 用户名
    ├── manifest.json            # 会话清单
    └── <date>/                  # 日期 YYYY-MM-DD
        └── <tool>__<sid>.jsonl  # 一个会话一个文件（工具名与 session id 用 __ 连接）
```

- `<tool>`：`claude-code` / `opencode` / `codex` / `kiro`
- 每个 `.jsonl` 每行一个事件，由组委会提供的日志归集工具生成并通过官方验证器检查。

导出与提交的完整步骤、字段定义见[《AI Coding 日志归集与提交手册》](https://github.com/open-vela/docs/blob/dev-ai-contest-2026/zh-cn/contest_2026/ai_coding_log_guide.md)。
