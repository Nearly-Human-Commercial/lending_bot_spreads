from __future__ import annotations

"""Utility wrappers for running Azure OpenAI Assistants conversations.

Contains:
- `AssistantRunner`: high‑level helper to send a user message, handle tool calls,
  and return the assistant's final reply.
- `ToolDispatcher`: maps tool calls emitted by the model to actual Python
  functions / HTTP requests. Replace mock implementations with real services.
"""

import time
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

# -----------------------------------------------------------------------------
# Tool dispatcher (project‑specific glue)
# -----------------------------------------------------------------------------


class ToolDispatcher:
    """Executes function calls issued by the assistant (AOAI run step)."""

    def __init__(self, client: AzureOpenAI):
        self.client = client

    # ---- main entry ---------------------------------------------------------

    def handle_step(self, run_obj: Any) -> List[Dict[str, Any]]:
        """Return a list of tool_outputs ready for submit_tool_outputs()."""
        outputs: List[Dict[str, Any]] = []
        for call in run_obj.required_action.submit_tool_outputs.tool_calls:
            name = call.function.name
            args = call.function.arguments
            if name == "webSearch":
                outputs.append({"tool_call_id": call.id, "output": self.web_search(**args)})
            elif name == "getRateSheet":
                outputs.append({"tool_call_id": call.id, "output": self.get_rate_sheet(**args)})
            elif name == "createLoanDoc":
                outputs.append({"tool_call_id": call.id, "output": self.create_loan_doc(**args)})
            else:
                outputs.append({"tool_call_id": call.id, "output": f"Unknown tool {name}"})
        return outputs

    # ---- mock tool implementations -----------------------------------------

    def web_search(self, query: str, freshness_days: int | None = None) -> str:
        return f"<mock WebSearch results for '{query}', freshness={freshness_days}>"

    def get_rate_sheet(self, loanType: str, fico: int, ltv: float) -> str:
        return f"RateSheet mock → {loanType}|FICO {fico}|LTV {ltv} ⇒ 6.25%/0.2 pts"

    def create_loan_doc(self, borrowerId: str, templateId: str) -> str:
        return f"file_mock_{borrowerId}_{templateId}"

# -----------------------------------------------------------------------------
# Assistant runner
# -----------------------------------------------------------------------------


class AssistantRunner:
    """Blocking convenience wrapper around the Assistants run lifecycle."""

    def __init__(
        self,
        *,
        assistant_id: str,
        client: AzureOpenAI,
        tool_dispatcher: ToolDispatcher,
    ) -> None:
        self.assistant_id = assistant_id
        self.client = client
        self.tool_dispatcher = tool_dispatcher

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(self, user_message: str, *, thread_id: Optional[str] = None) -> str:
        # 1) ensure we have a thread
        thread = (
            self.client.beta.threads.create()
            if thread_id is None
            else self.client.beta.threads.retrieve(thread_id)
        )

        # 2) post user message
        self.client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_message
        )

        # 3) start run
        run = self.client.beta.threads.runs.create(
            assistant_id=self.assistant_id,
            thread_id=thread.id,
            stream=False,
        )

        # 4) loop until completion
        while run.status in {"queued", "in_progress", "requires_action"}:
            if run.status == "requires_action":
                outputs = self.tool_dispatcher.handle_step(run)
                run = self.client.beta.threads.runs.submit_tool_outputs(
                    run_id=run.id,
                    thread_id=thread.id,
                    tool_outputs=outputs,
                )
            else:
                time.sleep(0.75)
                run = self.client.beta.threads.runs.retrieve(
                    run_id=run.id, thread_id=thread.id
                )

        # 5) fetch assistant's final message
        msgs = self.client.beta.threads.messages.list(thread_id=thread.id, order="asc")
        return msgs.data[-1].content[0].text.value
