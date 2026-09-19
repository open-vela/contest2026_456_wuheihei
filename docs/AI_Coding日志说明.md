# AI Coding 日志说明

本仓库提交一段 Codex CLI 只读审计会话，日志位于 `logs/bilibilidev/`。

## 采集过程

1. 在通过官方 `repo init` / `repo sync` 建立、且根目录含 `.repo/` 的 OpenVela 工作区内运行 Codex CLI。
2. 会话只读核验 v11 模型、Python 五折指标、C 主机模拟器证据和 OpenVela 功能边界。
3. 赛事官方采集器 1.3.0 的 `snapshot_core.py` 只识别旧式 `.message` 记录，而当前 Codex CLI 0.154 使用 `response_item` 记录。官方 Stop Hook 已运行，但未能解析当前格式。
4. `scripts/export_codex_contest_log.py` 对原始 Codex rollout 做确定性字段映射，再把规范化的临时 transcript 交给官方 `snapshot_core.py`。编号、脱敏、manifest 和最终 JSONL 均由官方采集核心生成。

## 完整性与隐私

- 导出会话 ID：`01a0b7b2-2d28-7620-85df-ad37b4e0f554`
- 工具：Codex CLI
- 模型：`gpt-5.6-sol`
- 导出事件：71 条
- 采集模式：`cli`
- 官方校验结果：`ALL OK`
- 本次导出新建比赛仓的只读审计会话；更早的桌面会话不在赛事采集器支持范围内，未纳入提交。

## 校验方法

使用赛事官方 AI Coding 日志归集包提供的校验脚本，在完整 OpenVela `repo` 工作区的队伍仓库目录执行：

```bash
python3 validate-log.py logs/
```

日志记录了 AI 参与仓库审计、证据核对与问题识别的过程，不代表项目全部历史会话。
