from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .openai_client import get_client
from .context_loader import retrieve_context
import os


def build_tool_schema(available_tools: List[str]) -> List[dict]:
    """Return the OpenAI Chat tools schema for the CAD copilot."""
    return [
        {
            "type": "function",
            "function": {
                "name": "select_tool",
                "description": "Activate a CAD tool by id (e.g., line, circle, move).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_id": {"type": "string", "enum": available_tools},
                    },
                    "required": ["tool_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_circle",
                "description": "Create a circle by radius and optional center coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "radius": {"type": "number", "minimum": 0.0},
                        "cx": {"type": "number"},
                        "cy": {"type": "number"},
                    },
                    "required": ["radius"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_line",
                "description": "Create a line by endpoints (x1,y1) to (x2,y2).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extrude",
                "description": "Extrude the current/last sketch or selected profile by a distance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "distance": {"type": "number"},
                    },
                    "required": ["distance"],
                },
            },
        },
    ]


class CADActionBus:
    """Dispatch CAD actions exposed to the language model."""

    def __init__(self) -> None:
        self.handlers: Dict[str, Callable[..., Any]] = {}

    def on(self, name: str, fn: Callable[..., Any]) -> None:
        self.handlers[name] = fn

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self.handlers:
            raise RuntimeError(f"No handler registered for tool '{name}'")
        return self.handlers[name](**kwargs)


def chat_with_tools(
    user_text: str,
    *,
    available_tools: List[str],
    bus: CADActionBus,
    model: str = "gpt-4o-mini",
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Send a chat turn and execute any requested tool calls."""
    client = get_client()

    # Load static guide (optional)
    guide_path = os.path.join(os.path.dirname(__file__), "..", "..", "COPILOT_README.md")
    static_guide = ""
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            static_guide = f.read()
    except Exception:
        static_guide = ""

    # Lightweight repo-grounding for this query
    try:
        dynamic_ctx = retrieve_context(user_text, k=4, max_chars=3000)
    except Exception:
        dynamic_ctx = ""

    system = {
        "role": "system",
        "content": (
            "You are the AdaptiveCAD Copilot. Follow the guide and prefer AdaptiveCAD-native constructs.\n\n"
            "### Guide\n" + static_guide + "\n\n"
            "### Repo-context (top matches)\n" + dynamic_ctx
        ),
    }

    base_history: List[Dict[str, str]] = []
    # Only allow plain user/assistant messages from prior to avoid malformed tool state
    if prior_messages:
        for m in prior_messages:
            role = m.get("role")
            if role in ("user", "assistant"):
                base_history.append({"role": role, "content": m.get("content", "")})
    messages = [system] + base_history + [{"role": "user", "content": user_text}]
    tools = build_tool_schema(available_tools)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )

    message = response.choices[0].message

    if getattr(message, "tool_calls", None):
        tool_feedback = []
        for call in message.tool_calls:
            if call.type != "function" or not call.function:
                continue
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            result = bus.call(name, **args)
            tool_feedback.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": json.dumps({"ok": True, "result": result}),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": message.tool_calls,
            }
        )
        messages.extend(tool_feedback)

        follow_up = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return follow_up.choices[0].message.content or "(no response)"

    return message.content or "(no response)"