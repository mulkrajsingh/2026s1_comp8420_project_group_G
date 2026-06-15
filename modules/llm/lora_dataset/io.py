"""Shared helpers for adapter instruction JSONL (chat format for SFT)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from app.prompt_library import SYSTEM_GUARDRAILS
from app.runtime import prompt_text_for_record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def validate_messages_record(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Record must include messages with at least system/user/assistant turns.")
    roles = {message.get("role") for message in messages}
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("messages must include user and assistant roles.")


def chat_record(
    *,
    user_content: str,
    assistant_content: str,
    source: str,
    task: str,
    prompt_id: str,
    license_note: str,
    system_content: str = SYSTEM_GUARDRAILS,
) -> dict[str, Any]:
    record = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": source,
        "task": task,
        "prompt_id": prompt_id,
        "license_note": license_note,
    }
    validate_messages_record(record)
    return record


def prompt_record_to_chat(
    prompt_record: dict[str, Any],
    assistant_content: str,
    *,
    source: str,
    prompt_id: str | None = None,
    license_note: str,
    strategy: str = "few_shot",
) -> dict[str, Any]:
    user_content = prompt_text_for_record(prompt_record, strategy)
    return chat_record(
        user_content=user_content,
        assistant_content=assistant_content,
        source=source,
        task=prompt_record["task"],
        prompt_id=prompt_id or str(prompt_record["prompt_id"]),
        license_note=license_note,
        system_content=prompt_record.get("system_guardrails", SYSTEM_GUARDRAILS),
    )


def load_messages_jsonl_dir(directory: Path, source_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.jsonl")):
        for index, row in enumerate(read_jsonl(path)):
            if "messages" not in row:
                raise ValueError(f"{path} row {index} missing messages.")
            row.setdefault("source", source_label)
            row.setdefault("prompt_id", f"{source_label}_{path.stem}_{index}")
            validate_messages_record(row)
            records.append(row)
    return records


def truncate(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."
