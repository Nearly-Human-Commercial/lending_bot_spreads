from __future__ import annotations

"""Dynamic tool definitions for Lending Bot.

This module defines:
- `ToolSchema` – dataclass describing a function/tool in JSON‑Schema style.
- `ToolRegistry` – runtime registry that returns the complete `tools` list ready
  for Azure OpenAI Assistants API.

Custom tools (`webSearch`, `getRateSheet`, `createLoanDoc`) and built‑ins are
registered in `ToolRegistry.default()` so the rest of the codebase imports a
single source of truth.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ToolSchema:
    """Schema for a single custom function tool."""

    name: str
    description: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]

    def to_openai_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by the Assistants API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                },
            },
        }

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Tracks built‑in and custom tools and produces the payload for AOAI."""

    def __init__(self) -> None:
        self._tools: List[ToolSchema | Dict[str, str]] = []

    # ------------- registration helpers --------------------------------------

    def add_function(self, schema: ToolSchema) -> None:
        self._tools.append(schema)

    def add_builtin(self, builtin_name: str) -> None:
        if builtin_name not in {"code_interpreter", "file_search"}:
            raise ValueError(f"Unknown built‑in tool: {builtin_name}")
        self._tools.append({"type": builtin_name})

    # ------------- factory ----------------------------------------------------

    @classmethod
    def default(cls) -> "ToolRegistry":
        """Return a registry pre‑populated with our standard toolset."""
        reg = cls()
        reg.add_builtin("code_interpreter")
        reg.add_builtin("file_search")

        reg.add_function(
            ToolSchema(
                name="webSearch",
                description="Real‑time Bing search for rates, regulations, or news.",
                properties={
                    "query": {"type": "string", "description": "Search query."},
                    "freshness_days": {
                        "type": "integer",
                        "description": "Restrict results to the last N days (optional).",
                    },
                },
                required=["query"],
            )
        )

        reg.add_function(
            ToolSchema(
                name="getRateSheet",
                description="Fetch the latest pricing for a loan scenario.",
                properties={
                    "loanType": {"type": "string"},
                    "fico": {"type": "integer"},
                    "ltv": {"type": "number"},
                },
                required=["loanType", "fico", "ltv"],
            )
        )

        reg.add_function(
            ToolSchema(
                name="createLoanDoc",
                description="Generate or fill a lending document and return the file_id.",
                properties={
                    "borrowerId": {"type": "string"},
                    "templateId": {"type": "string"},
                },
                required=["borrowerId", "templateId"],
            )
        )
        return reg

    # ------------- public API -------------------------------------------------

    def openai_tools(self) -> List[Dict[str, Any]]:
        """Return the tools list encoded for Assistants API creation."""
        payload: List[Dict[str, Any]] = []
        for t in self._tools:
            payload.append(t if isinstance(t, dict) else t.to_openai_dict())
        return payload
