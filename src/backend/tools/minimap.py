import datetime
import os
import pandas as pd
from typing import Any, Dict, List

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_cohere import CohereEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.retrievers import WikipediaRetriever
# from langchain_community.vectorstores import Chroma
# from langchain import LangChain, Tool

from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

import json
import logging
import time
import requests
from typing import Any, Dict, Iterator, List

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator

from urllib.parse import urljoin
from backend.tools.base import BaseTool as CohereBaseTool

"""
Plug in your lang chain retrieval implementation here.
We have an example flows with wikipedia and vector DBs.

More details: https://python.langchain.com/docs/integrations/retrievers
"""

logging = logging.getLogger(__name__)


def hash_string(s: str) -> str:
    """
    Hash a string using the djb2 algorithm.
    """
    hash = 5381
    for x in s:
        hash = ((hash << 5) + hash) + ord(x)
    return str(hash)

class MinimapAPIWrapper(BaseModel):
    """
    Wrapper around Minimap.ai API.

    This wrapper will use the Minimap API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.

    Parameters:
        top_k_results: number of the top-scored document
        MAX_QUERY_LENGTH: maximum length of the query.
          Default is 300 characters.
        doc_content_chars_max: maximum length of the document content.
          Content will be truncated if it exceeds this length.
          Default is 2000 characters.
        max_retry: maximum number of retries for a request. Default is 5.
        sleep_time: time to wait between retries.
          Default is 0.2 seconds.
        email: email address to be used for the PubMed API.
    """

    parse: Any  #: :meta private:

    base_url: str = os.environ.get("MINIMAP_API_URL", "http://host.docker.internal:8081")
    search_endpoint: str = urljoin(base_url, "/api/v1/platform/search")
    search_content_endpoint: str = urljoin(base_url, "/api/v1/platform/search/results/content")

    max_retry: int = 5

    # Default values for the parameters
    top_k_results: int = 15
    MAX_QUERY_LENGTH: int = 300
    doc_content_chars_max: int = 2000


    def run(self, query: str) -> str:
        """
        Run search queries against the Minimap search PI
        Returns a list of documents, each with a title, summary, and id.
        """

        try:
            # Create teh url params
            query_params = {
                'query': query,
                'map_id': 5541696631538816
            }

            response = requests.get(self.search_endpoint, params=query_params)

            if response.status_code != 200:
                return f"Minimap API returned status code {response.status_code}"

            search_status = response.json()
            search_key = search_status['key']

            # Wait 0.25 seconds before checking the status
            time.sleep(0.25)
            tries = self.max_retry
            search_results = None
            while True:
                response = requests.get(self.search_content_endpoint, params={'search_key': search_key})
                if response.status_code == 404:
                    if tries > 0:
                        time.sleep(0.5)
                        continue
                    else:
                        raise Exception(f"Error fetching results. The API returned status code {response.status_code}")

                if response.status_code != 200:
                    raise Exception(f"Error fetching results. The API returned status code {response.status_code}")


                search_results = response.json()
                break

            df = pd.DataFrame(search_results['results'])

            df['title_hash'] = df['title'].apply(hash_string)

            # Rename description to text
            df.rename(columns={
                'description': 'text',
                'id': 'document_id'
                }, inplace=True)

            # drop duplicates
            df = df.drop_duplicates(subset=['title_hash'])
            df['search_key'] = search_key

            results = df.to_dict(orient='records')

            # limit the number of results to top_k_results
            results = results[:self.top_k_results]

            logging.info(results)

            return results

        except Exception as ex:
            return f"Minimap exception: {ex}"


class MinimapQueryRun(BaseTool):
    """Tool that searches the Minimap.ai API."""

    name: str = "pub_med"
    description: str = (
        "A wrapper around Minimap.ai. "
        "Useful for searching news articles across a wide range of topics."
        "from sports, politics, sciences, life style, and all sorts of news and news-like content."
        "Input should be a search query."
    )

    api_wrapper: MinimapAPIWrapper = Field(default_factory=MinimapAPIWrapper)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Minimap tool."""
        return self.api_wrapper.run(query)

class LangChainMinimapRetriever(CohereBaseTool):
    """
    This class retrieves documents from Wikipedia using the langchain package.
    This requires wikipedia package to be installed.
    """

    def __init__(self):
        self.client = MinimapAPIWrapper()

    @classmethod
    def is_available(cls) -> bool:
        return True

    def call(self, parameters: dict, **kwargs: Any) -> List[Dict[str, Any]]:
        query = parameters.get("query", "")
        results = self.client.run(query)

        logging.info(results)
        logging.info(f"Minimap results")


        return results # [dict(result) for result in results]