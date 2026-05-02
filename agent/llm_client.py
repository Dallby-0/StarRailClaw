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
        self.timeout_s = timeout_s
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

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"ARK API error {resp.status_code}: {resp.text}")
        return resp.json()

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
        resp = self.session.post(
            f"{self.base_url}/images/generations",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"ARK image API error {resp.status_code}: {resp.text}")
        data = resp.json()
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
