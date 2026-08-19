from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import requests

from agent.responses_protocol import build_json_schema_response_input


@dataclass
class SessionMessageCache:
    """Per-session conversation cache with 2-hour TTL."""

    ttl_seconds: int = 2 * 60 * 60
    _sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, timestamp: float) -> bool:
        return self._now() - timestamp > self.ttl_seconds

    def _ensure(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None or self._is_expired(session["updated_at"]):
            session = {"messages": [], "updated_at": self._now()}
            self._sessions[session_id] = session
        return session

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        session = self._ensure(session_id)
        return session["messages"]

    def set_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._sessions[session_id] = {
            "messages": messages,
            "updated_at": self._now(),
        }


class DoubaoClient:
    """Minimal OpenAI-compatible client for ByteDance Volcano Engine ARK."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "ep-xxxxxxxxxxxxxxxx",
        timeout_s: int = 60,
        cache_ttl_seconds: int = 2 * 60 * 60,
    ) -> None:
        cfg = self._load_user_config()
        self.api_key = (api_key or os.getenv("ARK_API_KEY", "") or str(cfg.get("api_key", ""))).strip()
        if not self.api_key:
            raise ValueError("Missing API key. Set ARK_API_KEY or ~/starrailclaw/config.json api_key.")

        self.base_url = (
            base_url
            or os.getenv("ARK_BASE_URL", "")
            or str(cfg.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"))
        ).rstrip("/")
        self.model = str(os.getenv("ARK_MODEL", "") or cfg.get("model", model)).strip()
        self.image_model = str(os.getenv("ARK_IMAGE_MODEL", "") or cfg.get("image_model", "doubao-image-v1")).strip()
        self.reasoning = cfg.get("reasoning")
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.timeout_s = self._config_int("ARK_TIMEOUT_S", cfg.get("timeout_s"), timeout_s)
        self.max_retries = self._config_int("ARK_MAX_RETRIES", cfg.get("max_retries"), 3)
        self.retry_backoff_s = self._config_float("ARK_RETRY_BACKOFF_S", cfg.get("retry_backoff_s"), 2.0)
        self.pre_call_sleep_s = self._config_float("ARK_PRE_CALL_SLEEP_S", cfg.get("pre_call_sleep_s"), 4.0)
        self.max_tokens = self._config_int("ARK_MAX_TOKENS", cfg.get("max_tokens"), 4096)
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.session = requests.Session()
        self.cache = SessionMessageCache(ttl_seconds=cache_ttl_seconds)

    @staticmethod
    def _user_config_path() -> Path:
        return Path.home() / "starrailclaw" / "config.json"

    @classmethod
    def _load_user_config(cls) -> Dict[str, Any]:
        path = cls._user_config_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _config_int(env_name: str, config_value: Any, default: int) -> int:
        raw = os.getenv(env_name, "")
        value = raw if raw.strip() else config_value
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _config_float(env_name: str, config_value: Any, default: float) -> float:
        raw = os.getenv(env_name, "")
        value = raw if raw.strip() else config_value
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def encode_image_to_data_url(image_rgb) -> str:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode screenshot PNG")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _stable_digest(obj: Any) -> str:
        raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _retry_after_s(resp: requests.Response) -> float | None:
        raw = resp.headers.get("Retry-After", "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def _sleep_before_retry(self, attempt: int, resp: requests.Response | None = None, exc: Exception | None = None) -> None:
        retry_after = self._retry_after_s(resp) if resp is not None else None
        delay = retry_after if retry_after is not None else self.retry_backoff_s * (2 ** (attempt - 1))
        reason = f"status={resp.status_code}" if resp is not None else f"error={type(exc).__name__}"
        print(f"[llm][retry] {reason} attempt={attempt}/{self.max_retries} sleep_s={delay:.1f}")
        time.sleep(delay)

    def _sleep_before_request(self, path: str) -> None:
        if self.pre_call_sleep_s <= 0:
            return
        print(f"[llm][pre-call-wait] path={path} sleep_s={self.pre_call_sleep_s:.1f}")
        time.sleep(self.pre_call_sleep_s)

    def _post_json(self, path: str, payload: Dict[str, Any], *, error_label: str) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
        }
        last_exc: requests.RequestException | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                self._sleep_before_request(path)
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt, exc=exc)
                    continue
                raise
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt <= self.max_retries:
                    self._sleep_before_retry(attempt, resp=resp)
                    continue
            if resp.status_code >= 400:
                raise RuntimeError(f"{error_label} {resp.status_code}: {resp.text}")
            return resp.json()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{error_label}: request failed after retries")

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._post_json("chat/completions", payload, error_label="ARK API error")
        self._log_chat_response_status(data)
        return data

    def _post_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._post_json("responses", payload, error_label="ARK Responses API error")
        self._log_responses_status(data)
        return data

    @staticmethod
    def _usage_int(usage: dict[str, Any], key: str) -> int:
        try:
            return int(usage.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _token_k(tokens: int) -> str:
        return f"{tokens / 1000:.1f}k"

    def _log_chat_response_status(self, data: Dict[str, Any]) -> None:
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        finish_reason = str(choice.get("finish_reason", "") or "")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = self._usage_int(usage, "prompt_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            self.usage_totals["prompt_tokens"] += prompt_tokens
            self.usage_totals["completion_tokens"] += completion_tokens
            self.usage_totals["total_tokens"] += total_tokens
        print(
            "[llm][response] "
            f"finish_reason={finish_reason or '-'} "
            f"current={self._token_k(total_tokens)} "
            f"prompt={self._token_k(prompt_tokens)} "
            f"completion={self._token_k(completion_tokens)} "
            f"run_total={self._token_k(self.usage_totals['total_tokens'])}"
        )
        if finish_reason == "length":
            print("[llm][response][warning] completion truncated by max output tokens")

    def _log_responses_status(self, data: Dict[str, Any]) -> None:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = self._usage_int(usage, "input_tokens")
        output_tokens = self._usage_int(usage, "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        if total_tokens > 0:
            self.usage_totals["prompt_tokens"] += input_tokens
            self.usage_totals["completion_tokens"] += output_tokens
            self.usage_totals["total_tokens"] += total_tokens
        status = str(data.get("status", "") or "-")
        incomplete = data.get("incomplete_details")
        print(
            "[llm][responses] "
            f"status={status} "
            f"current={self._token_k(total_tokens)} "
            f"input={self._token_k(input_tokens)} "
            f"output={self._token_k(output_tokens)} "
            f"run_total={self._token_k(self.usage_totals['total_tokens'])}"
        )
        if incomplete:
            print(f"[llm][responses][warning] incomplete_details={incomplete}")

    def responses_json_schema(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_data_urls: List[str],
        schema_name: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a self-contained Responses request constrained by JSON Schema."""
        payload = build_json_schema_response_input(
            system_prompt=system_prompt,
            user_text=user_text,
            image_data_urls=image_data_urls,
            schema_name=schema_name,
            schema=schema,
        )
        payload["model"] = self.model
        return self._post_response(payload)

    def get_image_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an annotated image based on prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Image generation prompt"}
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def _generate_image_data_url(self, prompt: str) -> str:
        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1,
            "response_format": "b64_json",
        }
        data = self._post_json("images/generations", payload, error_label="ARK image API error")
        b64_image = data["data"][0]["b64_json"]
        return f"data:image/png;base64,{b64_image}"

    def try_run_client_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[bool, str]:
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        if name != "generate_image":
            return False, ""
        args_text = fn.get("arguments", "{}")
        args = json.loads(args_text) if args_text else {}
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return True, json.dumps({"ok": False, "error": "missing prompt"}, ensure_ascii=False)
        image_url = self._generate_image_data_url(prompt)
        result = {"ok": True, "tool": "generate_image", "prompt": prompt, "image_url": image_url}
        return True, json.dumps(result, ensure_ascii=False)

    def chat_with_session(
        self,
        *,
        session_id: str,
        system_prompt: str,
        user_message: Dict[str, Any],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] = "auto",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        history = self.cache.get_messages(session_id)

        if not history:
            history.append({"role": "system", "content": system_prompt})

        history.append(user_message)

        result = self._chat_from_history(
            session_id=session_id,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        assistant_msg = result["choices"][0]["message"]
        history.append(assistant_msg)
        self.cache.set_messages(session_id, history)
        return result

    def _chat_from_history(
        self,
        *,
        session_id: str,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] = "auto",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        request_messages = list(self.cache.get_messages(session_id))
        # Local cache key carries session_id, valid for 2h in SessionMessageCache.
        cache_key = self._stable_digest({"session_id": session_id, "messages": request_messages})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
            "tools": tools or [],
            "tool_choice": tool_choice,
            "parallel_tool_calls": False,
            "metadata": {"session_id": session_id, "cache_key": cache_key},
        }
        if isinstance(self.reasoning, dict):
            payload["reasoning"] = self.reasoning
        elif isinstance(self.reasoning_effort, str) and self.reasoning_effort.strip():
            payload["reasoning"] = {"effort": self.reasoning_effort.strip()}
        return self._post_chat(payload)

    def continue_session(
        self,
        *,
        session_id: str,
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] = "auto",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        history = self.cache.get_messages(session_id)
        result = self._chat_from_history(
            session_id=session_id,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        assistant_msg = result["choices"][0]["message"]
        history.append(assistant_msg)
        self.cache.set_messages(session_id, history)
        return result

    def append_tool_message(self, session_id: str, tool_call_id: str, content: str) -> None:
        history = self.cache.get_messages(session_id)
        history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self.cache.set_messages(session_id, history)
