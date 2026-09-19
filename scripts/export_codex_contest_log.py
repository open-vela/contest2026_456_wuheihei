#!/usr/bin/env python3
"""Bridge current Codex CLI rollouts to the official contest collector.

The openvela contest collector 1.3.0 expects the older Claude-compatible
``message`` transcript shape. Codex CLI 0.154 stores the same real session as
``response_item`` records. This adapter performs a deterministic schema
normalization and then delegates redaction, event numbering, manifest writing,
and repository export to the official ``snapshot_core.py``.

It never edits a generated contest JSONL file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def joined_text(blocks: object) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
            else:
                parts.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
    return "\n".join(part for part in parts if part)


def normalize_rollout(source: Path) -> tuple[list[dict[str, object]], str, str, str]:
    normalized: list[dict[str, object]] = []
    session_id = source.stem
    cwd = ""
    model = ""

    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        timestamp = str(record.get("timestamp") or "")
        record_type = record.get("type")
        payload = record.get("payload") or {}

        if record_type == "session_meta":
            session_id = str(payload.get("session_id") or payload.get("id") or session_id)
            cwd = str(payload.get("cwd") or cwd)
            model = str(payload.get("model") or model)
            continue

        if record_type == "turn_context":
            model = str(payload.get("model") or model)
            continue

        if record_type != "response_item" or not isinstance(payload, dict):
            continue

        item_type = payload.get("type")
        if item_type == "message":
            role = str(payload.get("role") or "system")
            if role == "developer":
                role = "system"
            if role not in {"user", "assistant", "system"}:
                role = "system"
            text = joined_text(payload.get("content"))
            if text:
                message: dict[str, object] = {
                    "role": role,
                    "content": text,
                }
                if role == "assistant" and model:
                    message["model"] = model
                normalized.append({"type": role, "timestamp": timestamp, "message": message})
            continue

        if item_type == "reasoning":
            summaries = payload.get("summary")
            thinking = joined_text(summaries)
            if thinking:
                normalized.append(
                    {
                        "type": "assistant",
                        "timestamp": timestamp,
                        "message": {
                            "role": "assistant",
                            "model": model,
                            "content": [{"type": "thinking", "thinking": thinking}],
                        },
                    }
                )
            continue

        if item_type == "custom_tool_call":
            call_id = str(payload.get("call_id") or payload.get("id") or "call_unknown")
            normalized.append(
                {
                    "type": "assistant",
                    "timestamp": timestamp,
                    "message": {
                        "role": "assistant",
                        "model": model,
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call_id,
                                "name": str(payload.get("name") or "unknown"),
                                "input": payload.get("input"),
                            }
                        ],
                    },
                }
            )
            continue

        if item_type == "custom_tool_call_output":
            call_id = str(payload.get("call_id") or payload.get("id") or "call_unknown")
            normalized.append(
                {
                    "type": "user",
                    "timestamp": timestamp,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": joined_text(payload.get("output")),
                            }
                        ],
                    },
                }
            )

    if not cwd:
        raise ValueError("session metadata does not contain cwd")
    if not normalized:
        raise ValueError("no supported Codex response items found")
    return normalized, session_id, cwd, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path, help="Codex rollout JSONL")
    parser.add_argument("--normalized-output", type=Path, required=True)
    parser.add_argument("--collector-core", type=Path, required=True)
    args = parser.parse_args()

    source = args.rollout.expanduser().resolve()
    output = args.normalized_output.expanduser().resolve()
    core = args.collector_core.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not core.is_file():
        raise FileNotFoundError(core)

    events, session_id, cwd, model = normalize_rollout(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"source_sha256={digest}")
    print(f"session_id={session_id}")
    print(f"model={model or 'unknown'}")
    print(f"normalized_events={len(events)}")

    payload = {
        "session_id": session_id,
        "cwd": cwd,
        "transcript_path": str(output),
        "hook_event_name": "Stop",
    }
    result = subprocess.run(
        [sys.executable, str(core), "--tool", "codex"],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        cwd=cwd,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
