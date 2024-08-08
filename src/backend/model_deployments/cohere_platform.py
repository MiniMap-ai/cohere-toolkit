import json
import logging
import os
import time
from typing import Any, Dict, Generator, List

import cohere
import requests
from cohere.types import StreamedChatResponse

from backend.chat.collate import to_dict
from backend.chat.enums import StreamEvent
from backend.model_deployments.base import BaseDeployment
from backend.model_deployments.utils import get_model_config_var
from backend.schemas.cohere_chat import CohereChatRequest
from backend.tools.minimap import LangChainMinimapRetriever

from cohere.types.tool import Tool

MINIMAP_TOOL = Tool(
    name="Minimap",
    description="Fetches the most relevant news and content from Minimap.ai.",
    parameter_definitions={
        "query": {
            "description": "Search API that takes a query or phrase. Results should be presented as an executive summary, grouped and summarized for the user with section headings and bullet points.",
            "type": "str",
            "required": True,
        }
    },
)

COHERE_API_KEY_ENV_VAR = "COHERE_API_KEY"
COHERE_ENV_VARS = [COHERE_API_KEY_ENV_VAR]


MODELS =  [
        {
            'name': 'command-r',
            'endpoints': ['generate', 'chat', 'summarize'],
            'finetuned': False,
            'context_length': 128000,
            'tokenizer_url': 'https://storage.googleapis.com/cohere-public/tokenizers/command-r.json',
            'default_endpoints': []
        },
        {
            'name': 'command-r-plus',
            'endpoints': ['generate', 'chat', 'summarize'],
            'finetuned': False,
            'context_length': 128000,
            'tokenizer_url': 'https://storage.googleapis.com/cohere-public/tokenizers/command-r-plus.json',
            'default_endpoints': ['chat']
        },
]

preamble = f"""
You are a news summarization assistant. You're equipped with Minimap.ai's news search tool.

You will be provided with large number of news articles. You're task is to provide users with salient summaries of major trends in the news. Summaries should be presented as a mini news brief, with topic headings and contain bullet points with key information.

You can elaborate on or use a more precise query than what the user provided to get more specific results.

Always ask the user if there's a specific point or topic they want to drill down on.

Today's date is {time.strftime("%Y-%m-%d")}.
"""

class CohereDeployment(BaseDeployment):
    """Cohere Platform Deployment."""

    client_name = "cohere-toolkit"
    api_key = get_model_config_var(COHERE_API_KEY_ENV_VAR)

    def __init__(self, **kwargs: Any):
        # Override the environment variable from the request
        self.client = cohere.Client(api_key=self.api_key, client_name=self.client_name)

    @property
    def rerank_enabled(self) -> bool:
        return True

    @classmethod
    def list_models(cls) -> List[str]:
        if not CohereDeployment.is_available():
            return []

        # url = "https://api.cohere.ai/v1/models"
        # headers = {
        #     "accept": "application/json",
        #     "authorization": f"Bearer {cls.api_key}",
        # }

        # response = requests.get(url, headers=headers)
        # logging.info(response.json())
        # if not response.ok:
        #     logging.warning("Couldn't get models from Cohere API.")
        #     return []

        models = MODELS

        return [
            model["name"]
            for model in models
            if model.get("endpoints") and "chat" in model["endpoints"]
        ]

    @classmethod
    def is_available(cls) -> bool:
        return all([os.environ.get(var) is not None for var in COHERE_ENV_VARS])

    def invoke_chat(self, chat_request: CohereChatRequest, **kwargs: Any) -> Any:
        logging.info(f"Invoking chat with chat_request")
        response = self.client.chat(
            **chat_request.model_dump(exclude={"stream"}),
            **kwargs,
        )
        yield to_dict(response)

    def invoke_chat_stream(
        self, chat_request: CohereChatRequest, **kwargs: Any
    ) -> Generator[StreamedChatResponse, None, None]:

        chat_params = {
            **chat_request.model_dump(exclude={"stream", "file_ids"}),
            **kwargs
        }

        # Append "Minimap" to the tools list
        chat_params["tools"] = chat_params.get("tools", []) + [MINIMAP_TOOL]

        # Set the preamble
        chat_params["preamble"] = preamble

        stream = self.client.chat_stream(
            **chat_params,
        )

        for event in stream:
            yield to_dict(event)

    def invoke_search_queries(
        self,
        message: str,
        chat_history: List[Dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        res = self.client.chat(
            message=message,
            chat_history=chat_history,
            search_queries_only=True,
            **kwargs,
        )

        if not res.search_queries:
            return []

        return [s.text for s in res.search_queries]

    def invoke_rerank(
        self, query: str, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        return self.client.rerank(
            query=query, documents=documents, model="rerank-english-v2.0", **kwargs
        )

    def invoke_tools(
        self,
        chat_request: CohereChatRequest,
        **kwargs: Any,
    ) -> Generator[StreamedChatResponse, None, None]:

        yield from self.invoke_chat_stream(chat_request, **kwargs)
