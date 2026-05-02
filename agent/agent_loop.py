from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import requests

from agent.llm_client import DoubaoClient
from agent.tooling.tools import ToolExecutor, build_tool_schemas
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient

DEFAULT_SYSTEM_PROMPT = (
    "# 目标"
    "你是一个安卓agent，拥有在安卓模拟器上模拟点击的能力，正在尝试通关模拟宇宙-差分宇宙"
    "# 核心规则\n"
    "每轮你会收到一张游戏截图，你需要分析当前游戏状态，决定是否需要点击屏幕来推进游戏。\n\n"
    "# 坐标与点击规则\n"
    "1. 坐标原点固定为截图左上角，x轴向右为正，y轴向下为正，最大坐标以归一化后的(999, 999)为准\n"
    "2. 点击坐标必须是目标元素（如按钮、图标）的中心像素点，禁止以图标左上角、文字位置或阴影位置作为点击点。\n"
    "3. 当需要点击时，必须调用 tap 工具，给出精确的 tap(x=?, y=?) 指令\n"
    "4. 不论是否需要点击，给出最简短的理由。\n\n"
)

ANSWER_IMAGE_DIR = Path("debug") / "answers"


def _save_answer_image_data_url(data_url: str, out_dir: Path) -> str | None:
    prefix = "data:image/"
    if not data_url.startswith(prefix):
        return None
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None
    ext = "png"
    mime = header[len("data:") :].split(";")[0]
    if "/" in mime:
        ext = mime.split("/", 1)[1] or "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_annotated.{ext}"
    raw = base64.b64decode(b64)
    out.write_bytes(raw)
    return str(out)


def _save_answer_image_http_url(image_url: str, out_dir: Path) -> str | None:
    if not (image_url.startswith("http://") or image_url.startswith("https://")):
        return None
    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None
    ctype = (resp.headers.get("Content-Type") or "").lower()
    ext = "png"
    if "jpeg" in ctype or "jpg" in ctype:
        ext = "jpg"
    elif "webp" in ctype:
        ext = "webp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_annotated.{ext}"
    out.write_bytes(resp.content)
    return str(out)


def _save_answer_image_base64(raw_b64: str, out_dir: Path, ext: str = "png") -> str | None:
    if not raw_b64:
        return None
    try:
        raw = base64.b64decode(raw_b64)
    except Exception:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_annotated.{ext}"
    out.write_bytes(raw)
    return str(out)


def _extract_and_save_llm_image(assistant_msg: dict, out_dir: Path) -> str | None:
    content = assistant_msg.get("content")
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "image_url":
            image_obj = part.get("image_url")
            if isinstance(image_obj, dict):
                url = image_obj.get("url", "")
            elif isinstance(image_obj, str):
                url = image_obj
            else:
                url = ""
            if isinstance(url, str) and url:
                saved = _save_answer_image_data_url(url, out_dir) or _save_answer_image_http_url(url, out_dir)
                if saved:
                    return saved
        if ptype in {"output_image", "image"}:
            b64 = part.get("b64_json") or part.get("image_base64") or ""
            if isinstance(b64, str) and b64:
                saved = _save_answer_image_base64(b64, out_dir)
                if saved:
                    return saved
    return None


def _save_local_annotated_frame(frame_rgb, out_dir: Path, taps: list[tuple[int, int]]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "local_annotated.png"
    image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for i, (x, y) in enumerate(taps, start=1):
        cv2.circle(image_bgr, (int(x), int(y)), 10, (0, 255, 0), 2)
        cv2.drawMarker(
            image_bgr,
            (int(x), int(y)),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            image_bgr,
            f"tap{i}({int(x)},{int(y)})",
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out), image_bgr)
    return str(out)


def _build_user_message_from_frame(image_rgb) -> dict:
    data_url = DoubaoClient.encode_image_to_data_url(image_rgb)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


def _truncate_data_urls(obj, max_len: int = 120):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _truncate_data_urls(v, max_len=max_len)
        return out
    if isinstance(obj, list):
        return [_truncate_data_urls(v, max_len=max_len) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image/") and len(obj) > max_len:
        return obj[:max_len] + "...<truncated>"
    return obj


def _print_json_block(title: str, payload) -> None:
    try:
        safe = _truncate_data_urls(payload)
        print(f"{title}\n{json.dumps(safe, ensure_ascii=False, indent=2)}")
    except Exception as exc:
        print(f"{title}\n<failed to print json: {exc}>")


def run_agent_loop(
    *,
    system_prompt: str,
    session_id: str,
    serial: str | None = None,
    adb_path: str | None = None,
    interval_s: int = 5,
) -> None:
    target_serial = resolve_target_serial(serial=serial, adb_path=adb_path, auto_connect=True)
    emulator = EmulatorClient(serial=target_serial, adb_path=adb_path)
    llm = DoubaoClient()
    tools = build_tool_schemas()
    executor = ToolExecutor(emulator)

    if llm.reasoning is None and (llm.reasoning_effort is None or str(llm.reasoning_effort).strip() == ""):
        print("[agent] reasoning=OFF (no reasoning payload)")
    else:
        print(f"[agent] reasoning=ON reasoning={llm.reasoning} reasoning_effort={llm.reasoning_effort}")
    print(f"[agent] started, serial={target_serial}, session_id={session_id}")
    ANSWER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[agent] answer image dir: {ANSWER_IMAGE_DIR.resolve()}")

    round_idx = 1
    while True:
        round_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        round_dir = ANSWER_IMAGE_DIR / f"round{round_idx}_{round_ts}"
        round_dir.mkdir(parents=True, exist_ok=True)
        frame = emulator.screenshot(prefer_png=True)
        fh, fw = frame.shape[:2]
        print(f"[round {round_idx}] screenshot captured ({fw}x{fh})")
        print(f"[round {round_idx}] original_image_size={fw}x{fh}")
        if (fw, fh) != (1280, 720):
            print(f"[round {round_idx}] resize screenshot to 1280x720")
        frame_for_llm = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        executor.set_viewport_size(1280, 720)
        user_message = _build_user_message_from_frame(frame_for_llm)

        resp = llm.chat_with_session(
            session_id=session_id,
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_choice="auto",
        )
        _print_json_block(f"[round {round_idx}][llm_input][user_message]", user_message)
        _print_json_block(f"[round {round_idx}][llm_output][raw_response]", resp)
        round_taps: list[tuple[int, int]] = []

        while True:
            assistant_msg = resp["choices"][0]["message"]
            _print_json_block(f"[round {round_idx}][assistant_msg]", assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []
            text = assistant_msg.get("content")
            if isinstance(text, str) and text.strip():
                print(f"[round {round_idx}][assistant] {text.strip()}")
            elif isinstance(text, list):
                parts = [p.get("text", "") for p in text if isinstance(p, dict) and p.get("type") in {"text", "output_text"}]
                joined = "".join(parts).strip()
                if joined:
                    print(f"[round {round_idx}][assistant] {joined}")
                saved = _extract_and_save_llm_image(assistant_msg, round_dir)
                if saved:
                    print(f"[round {round_idx}][assistant_image] saved: {saved}")
                else:
                    part_types = [p.get("type") for p in text if isinstance(p, dict)]
                    print(f"[round {round_idx}][assistant_image] not found in content parts: {part_types}")

            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                result = executor.run_tool_call(tool_call)
                args_text = tool_call.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_text) if args_text else {}
                except json.JSONDecodeError:
                    args = {"raw": args_text}
                if isinstance(args, dict):
                    args_compact = ", ".join(f"{k}={v}" for k, v in args.items())
                else:
                    args_compact = str(args)
                ok = "ok" if "\"ok\": true" in result.lower() else "err"
                print(f"[round {round_idx}][tool] {tool_name}({args_compact}) -> {ok}")
                try:
                    result_obj = json.loads(result)
                except json.JSONDecodeError:
                    result_obj = {}
                if tool_name == "tap" and isinstance(result_obj, dict):
                    req = result_obj.get("requested")
                    remap = result_obj.get("remapped_viewport")
                    actual = result_obj.get("actual")
                    print(f"[round {round_idx}][tool][tap_coords] requested={req} remapped={remap} actual={actual}")
                if (
                    tool_name == "tap"
                    and isinstance(result_obj, dict)
                    and isinstance(result_obj.get("actual"), dict)
                ):
                    ax = result_obj["actual"].get("x")
                    ay = result_obj["actual"].get("y")
                    if isinstance(ax, int) and isinstance(ay, int):
                        round_taps.append((ax, ay))
                llm.append_tool_message(
                    session_id=session_id,
                    tool_call_id=tool_call["id"],
                    content=result,
                )

            # Continue same turn until model stops calling tools.
            resp = llm.continue_session(
                session_id=session_id,
                tools=tools,
                tool_choice="auto",
            )
            _print_json_block(f"[round {round_idx}][llm_output][continue_raw_response]", resp)

        local_saved = _save_local_annotated_frame(frame_for_llm, round_dir, round_taps)
        if round_taps:
            print(f"[round {round_idx}][local_image] saved with taps: {local_saved}")
        else:
            print(f"[round {round_idx}][local_image] saved(no tap): {local_saved}")

        time.sleep(interval_s)
        round_idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal ARK-driven emulator agent loop")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for task policy")
    parser.add_argument("--session-id", default=f"session-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--adb-path", default=None)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    run_agent_loop(
        system_prompt=args.system,
        session_id=args.session_id,
        serial=args.serial,
        adb_path=args.adb_path,
        interval_s=args.interval,
    )


if __name__ == "__main__":
    main()
