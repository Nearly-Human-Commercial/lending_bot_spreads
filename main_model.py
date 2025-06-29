from __future__ import annotations

"""End-to-end Lending Copilot pipeline integrating tools, vector store, and AOAI.

Key update (2025‑06‑11):
- Added **temporary file indexing**. Pass a list of local file paths (PDFs, DOCX, etc.)
  via `temp_files` and the constructor will upload them to Azure OpenAI, build a
  transient vector store, and attach it to the assistant so `file_search` can
  retrieve their contents immediately.
"""

from pathlib import Path
from typing import List, Optional

from openai import AzureOpenAI
from haystack import Pipeline
from haystack.nodes import PreProcessor, ReconstructorRetriever  # type: ignore

from tools import ToolRegistry
from conversation_utils import AssistantRunner, ToolDispatcher

# -----------------------------------------------------------------------------
# Main pipeline class
# -----------------------------------------------------------------------------


class LendingBotPipeline:
    """Encapsulates the full build/run lifecycle for the Lending Bot."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        api_version: str = "2025-03-01-preview",
        model: str = "GPT4O_DEPLOYMENT",
        vector_store_id: Optional[str] = None,
        temp_files: Optional[List[str]] = None,  # NEW
    ) -> None:
        # Step 1 → Initialize tools ------------------------------------------------
        self.registry = ToolRegistry.default()

        # Step 2 → Connect to/prepare vector store ---------------------------------
        self.client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )

        # If caller supplied raw files, upload & build a *temporary* vector store.
        if temp_files:
            vector_store_id = self._index_temp_files(temp_files)
        # else: use supplied vector_store_id or None

        # Step 3 → Create AOAI assistant ------------------------------------------
        assistant = self.client.beta.assistants.create(
            name="Lending Copilot",
            model=model,
            instructions=(
                "You answer lending questions, cite sources, and call tools as needed."
            ),
            tools=self.registry.openai_tools(),
            tool_resources={
                "file_search": {"vector_store_ids": [vector_store_id]}
                if vector_store_id
                else {}
            },
        )
        self.assistant_id = assistant.id

        # Haystack components ------------------------------------------------------
        self.retriever = ReconstructorRetriever(document_store=None)  # TODO
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=PreProcessor(), name="pre", inputs=["Query"])
        self.pipeline.add_node(component=self.retriever, name="ret", inputs=["pre"])

        # Glue helpers -------------------------------------------------------------
        self.tool_dispatcher = ToolDispatcher(self.client)
        self.runner = AssistantRunner(
            assistant_id=self.assistant_id,
            client=self.client,
            tool_dispatcher=self.tool_dispatcher,
        )

    # -------------------------------------------------------------------------
    # Utility: upload & embed temporary files
    # -------------------------------------------------------------------------

    def _index_temp_files(self, files: List[str]) -> str:
        """Upload local files → create vector store → return its ID."""
        uploaded_ids = []
        for p in files:
            file_path = Path(p)
            if not file_path.exists():
                raise FileNotFoundError(file_path)
            upload = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="assistants",
            )
            uploaded_ids.append(upload.id)

        vs = self.client.beta.vector_stores.create(
            name="temp_vs", file_ids=uploaded_ids
        )
        # Poll until ready (simplified – production code should back‑off)
        while vs.status != "completed":
            vs = self.client.beta.vector_stores.retrieve(vs.id)
        return vs.id

    # -------------------------------------------------------------------------
    # Public inference API – mirrors an "infer" call
    # -------------------------------------------------------------------------

    def run(self, prompt: str) -> str:
        _ = self.pipeline.run(query=prompt)  # (optional preprocessing)
        return self.runner.run(prompt)
