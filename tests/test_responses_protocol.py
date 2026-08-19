from __future__ import annotations

from agent.responses_protocol import build_json_schema_response_input, response_output_text
from state_machine.protocol import state_bootstrap_json_schema


def test_responses_request_uses_json_schema_and_multimodal_input() -> None:
    schema = state_bootstrap_json_schema()
    payload = build_json_schema_response_input(
        system_prompt="system",
        user_text="context",
        image_data_urls=["data:image/png;base64,abc"],
        schema_name="fsm_state_bootstrap",
        schema=schema,
    )
    assert "messages" not in payload
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is False
    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "fsm_state_bootstrap",
        "schema": schema,
    }
    assert payload["input"][0]["content"][0] == {"type": "input_text", "text": "system"}
    assert payload["input"][1]["content"] == [
        {"type": "input_text", "text": "context"},
        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
    ]


def test_response_output_text_supports_both_responses_envelopes() -> None:
    assert response_output_text({"output_text": '{"ok":true}'}) == '{"ok":true}'
    nested = {
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]},
        ]
    }
    assert response_output_text(nested) == '{"ok":true}'


def test_state_bootstrap_schema_is_closed_and_bounded() -> None:
    schema = state_bootstrap_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["bootstrap_operations"]["maxItems"] == 2

    element_variants = schema["properties"]["elements"]["items"]["anyOf"]
    for variant in element_variants:
        assert variant["additionalProperties"] is False
        bbox = variant["properties"]["bbox"]
        assert bbox["minItems"] == bbox["maxItems"] == 4

    operation = schema["properties"]["bootstrap_operations"]["items"]
    assert operation["additionalProperties"] is False
    steps = operation["properties"]["steps"]
    assert steps["minItems"] == 1
    assert steps["maxItems"] == 2
