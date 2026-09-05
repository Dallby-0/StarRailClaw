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
        "strict": True,
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
    assert schema["properties"]["scene_mode"]["enum"] == ["ui_2d", "scene_3d", "unknown"]

    execution_variants = schema["properties"]["execution"]["anyOf"]
    reactive = next(item for item in execution_variants if item["properties"]["kind"]["enum"] == ["reactive_2d"])
    invoke_tool = next(item for item in execution_variants if item["properties"]["kind"]["enum"] == ["invoke_tool"])
    assert reactive["properties"]["bootstrap_operations"]["maxItems"] == 2
    assert "find_and_interact_with_next_object" in invoke_tool["properties"]["tool_name"]["enum"]

    element_variants = schema["properties"]["elements"]["items"]["anyOf"]
    for variant in element_variants:
        assert variant["additionalProperties"] is False
        bbox = variant["properties"]["bbox"]
        assert bbox["minItems"] == bbox["maxItems"] == 4
        assert bbox["items"]["minimum"] == 0
        assert bbox["items"]["maximum"] == 1000

    operation = reactive["properties"]["bootstrap_operations"]["items"]
    assert operation["additionalProperties"] is False
    assert "safety" not in operation["properties"]
    providers = operation["properties"]["providers"]
    assert providers["minItems"] == 1
    assert providers["maxItems"] == 4
    provider = providers["items"]
    assert provider["additionalProperties"] is False
    point = provider["properties"]["locators"]["items"]["anyOf"][0]
    assert point["properties"]["coordinate_space"]["enum"] == ["logical"]
    assert "effect_hints" in provider["properties"]
    assert "deferred_hints" in provider["properties"]
    assert all(
        variant["properties"]["type"]["enum"] != ["run_preset"]
        for variant in provider["properties"]["locators"]["items"]["anyOf"]
    )

    intent = schema["properties"]["intent_proposal"]["anyOf"][1]
    assert set(intent["properties"]) == {"kind", "phase", "transitions", "completion"}
    transition = intent["properties"]["transitions"]["items"]
    assert set(transition["properties"]) == {"from_phase", "event", "next_phase", "status"}
