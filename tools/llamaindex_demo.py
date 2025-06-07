#!/usr/bin/env python3
"""Example LlamaIndex pipeline for loading and querying DOCX files."""

import argparse
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient

COLLECTION_NAME = "docx_demo"


def build_index(doc_folder: str) -> VectorStoreIndex:
    """Load DOCX files and build a vector index."""
    azure_key = os.environ["AZURE_OPENAI_KEY"]
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")

    docs = SimpleDirectoryReader(doc_folder, required_exts=[".docx"]).load_data()

    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    post = SemanticSplitterNodePostprocessor(buffer_size=3, threshold=0.7)
    nodes = post.postprocess_nodes(nodes)

    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=azure_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_version,
    )

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)

    return VectorStoreIndex(nodes, embed_model=embed_model, vector_store=vector_store)


def query_loop(index: VectorStoreIndex, top_k: int, rerank: bool) -> None:
    """Interactive query loop using GPT-4o-mini."""
    azure_key = os.environ["AZURE_OPENAI_KEY"]
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")

    llm = AzureOpenAI(
        model="gpt-4o-mini",
        api_key=azure_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_version,
    )

    retriever = index.as_retriever(similarity_top_k=top_k)
    postprocessors = [SentenceTransformerRerank()] if rerank else None
    engine = RetrieverQueryEngine.from_args(retriever, llm=llm, node_postprocessors=postprocessors)

    while True:
        query = input("Query> ").strip()
        if not query:
            break
        response = engine.query(query)
        print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DOCX files with LlamaIndex")
    parser.add_argument("--docs", default="./docs", help="Folder containing DOCX files")
    parser.add_argument("--top_k", type=int, default=3, help="Number of nodes to retrieve")
    parser.add_argument("--rerank", action="store_true", help="Apply a reranker to retrieved nodes")
    args = parser.parse_args()

    index = build_index(args.docs)
    query_loop(index, args.top_k, args.rerank)


if __name__ == "__main__":
    main()
