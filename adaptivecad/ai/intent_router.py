from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from adaptivecad.plugins.macro_engine import MacroDef, MacroParam, MacroStep
from adaptivecad.plugins.tool_registry import ToolRegistry

from .context_loader import retrieve_context
from .openai_client import get_client


def build_tool_schema(available_tools: List[str]) -> List[dict]:
    """Return the OpenAI Chat tools schema for the CAD copilot."""
    base_tools = [
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
                "name": "create_pi_circle",
                "description": "Create a pi_a-native circle (superellipse with equal axes, n=2).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "radius": {"type": "number", "minimum": 0.0},
                        "cx": {"type": "number"},
                        "cy": {"type": "number"},
                        "segments": {"type": "integer", "minimum": 16, "default": 256},
                    },
                    "required": ["radius"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "upgrade_profile_to_pi_a",
                "description": (
                    "Mark the current or provided profile as pi_a " "for downstream operations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile_id": {"type": "string"},
                    },
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
                        "metric": {"type": "string", "enum": ["pi_a", "euclid"]},
                    },
                    "required": ["distance"],
                },
            },
        },
    ]
    return base_tools


def _custom_tools_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "create_custom_tool",
                "description": "Create or update a custom tool (macro) and optionally pin it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "author": {"type": "string"},
                        "ui": {
                            "type": "object",
                            "properties": {
                                "icon_text": {"type": "string"},
                                "pinned": {"type": "boolean"},
                            },
                        },
                        "params": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["number", "int", "string", "bool"],
                                    },
                                    "default": {},
                                    "label": {"type": "string"},
                                    "min": {"type": "number"},
                                    "max": {"type": "number"},
                                },
                                "required": ["name"],
                            },
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "call": {"type": "string"},
                                    "args": {"type": "object"},
                                },
                                "required": ["call"],
                            },
                        },
                    },
                    "required": ["id", "name", "steps"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pin_custom_tool",
                "description": "Pin/unpin a custom tool to the radial wheel.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "pinned": {"type": "boolean"}},
                    "required": ["id", "pinned"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_custom_tools",
                "description": "List available custom tools (ids and names).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_custom_tool",
                "description": "Execute a custom tool by id with optional parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "params": {"type": "object"}},
                    "required": ["id"],
                },
            },
        },
    ]


def _to_macro_def(payload: dict) -> MacroDef:
    params = [MacroParam(**p) for p in payload.get("params", [])]
    steps = [MacroStep(**s) for s in payload.get("steps", [])]
    return MacroDef(
        id=payload["id"],
        name=payload["name"],
        author=payload.get("author", "copilot"),
        description=payload.get("description", ""),
        params=params,
        steps=steps,
        ui=payload.get("ui", {}),
    )


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
    registry: Optional[ToolRegistry] = None,
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
            "You are the AdaptiveCAD Copilot. "
            "Follow the guide and prefer AdaptiveCAD-native constructs.\n\n"
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
    if registry is not None:
        tools += _custom_tools_schemas()

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
            # Registry-backed calls
            if (
                name
                in ("create_custom_tool", "pin_custom_tool", "list_custom_tools", "run_custom_tool")
                and registry is not None
            ):
                if name == "create_custom_tool":
                    m = _to_macro_def(args)
                    registry.add_or_update(m, persist=True)
                    result = {"created_or_updated": m.id}
                elif name == "pin_custom_tool":
                    registry.set_pinned(args["id"], bool(args["pinned"]))
                    result = {"pinned": bool(args["pinned"])}
                elif name == "list_custom_tools":
                    result = [
                        {"id": m.id, "name": m.name, "pinned": bool(m.ui.get("pinned"))}
                        for m in registry.list()
                    ]
                elif name == "run_custom_tool":
                    out = registry.run(args["id"], params_values=args.get("params"))
                    result = {"results": out}
            else:
                # Normal CAD tool call via bus
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
