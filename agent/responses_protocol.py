from __future__ import annotations

from typing import Any


def build_json_schema_response_input(
    *,
    system_prompt: str,
    user_text: str,
    image_data_urls: list[str],
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Build an Ark Responses API request with server-enforced JSON Schema output."""
    user_content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    user_content.extend({"type": "input_image", "image_url": url} for url in image_data_urls)
    return {
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        "thinking": {"type": "disabled"},
        "stream": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
            }
        },
    }


def response_output_text(data: dict[str, Any]) -> str:
    """Extract text from both compact and canonical Responses API envelopes."""
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"output_text", "text"}:
                continue
            value = part.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts).strip()
