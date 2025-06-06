"""Minimal LlamaIndex pipeline for corporate documents."""
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodePostprocessor
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

COLLECTION_NAME = "docs_luat"


def build_index(doc_folder: str) -> VectorStoreIndex:
    azure_key = os.environ["AZURE_OPENAI_KEY"]
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")

    reader = SimpleDirectoryReader(doc_folder)
    docs = reader.load_data()

    splitter = SentenceSplitter()
    nodes = splitter.get_nodes_from_documents(docs)

    post = SemanticSplitterNodePostprocessor(buffer_size=3, threshold=90)
    nodes = post.postprocess_nodes(nodes)

    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=azure_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_version,
    )

    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                                 api_key=os.getenv("QDRANT_API_KEY"))
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)

    return VectorStoreIndex(nodes, embed_model=embed_model, vector_store=vector_store)


def query_loop(index: VectorStoreIndex) -> None:
    azure_key = os.environ["AZURE_OPENAI_KEY"]
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")

    llm = AzureOpenAI(
        model="gpt-4o",
        api_key=azure_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_version,
    )

    query_engine = index.as_query_engine(llm=llm)
    while True:
        query = input("Query> ")
        if not query:
            break
        resp = query_engine.query(query)
        print(resp)


def main() -> None:
    folder = os.getenv("DOC_FOLDER", "./docs")
    index = build_index(folder)
    query_loop(index)


if __name__ == "__main__":
    main()
