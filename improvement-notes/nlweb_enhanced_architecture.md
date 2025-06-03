# Cách Hoạt Động Của NLWeb - Enhanced với Blob Storage & LlamaIndex

## Tổng Quan

NLWeb là một framework mã nguồn mở được Microsoft phát triển để tạo ra các giao diện trò chuyện tự nhiên (conversational interfaces) cho websites. Hệ thống này hoạt động như một "search engine thông minh" với khả năng hiểu và trả lời các câu hỏi bằng ngôn ngữ tự nhiên.

**🆕 [ENHANCED]** Phiên bản mở rộng này bổ sung khả năng lưu trữ blob storage và chunking thông minh với LlamaIndex để tăng độ chính xác và khả năng truy xuất nguồn gốc.

## Kiến Trúc Tổng Thể

### Các Thành Phần Chính

```
┌─────────────────────────────────────────────────────────────┐
│                    NLWeb Enhanced Architecture               │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (static/)                                    │
│  ├── HTML/CSS/JS files                                      │
│  └── Chat Interface Components                              │
├─────────────────────────────────────────────────────────────┤
│  Web Server (webserver/)                                    │
│  ├── WebServer.py - HTTP server chính                       │
│  ├── StreamingWrapper.py - Hỗ trợ streaming response        │
│  └── Static File Handler                                    │
├─────────────────────────────────────────────────────────────┤
│  Core Processing Engine (core/)                             │
│  ├── baseHandler.py - Xử lý request cơ bản                  │
│  ├── generate_answer.py - Tạo câu trả lời (RAG mode)        │
│  ├── ranking.py - Xếp hạng kết quả                          │
│  ├── fastTrack.py - Xử lý nhanh cho query đơn giản          │
│  ├── post_ranking.py - Xử lý sau khi xếp hạng               │
│  ├── state.py - Quản lý trạng thái hội thoại                │
│  ├── whoHandler.py - Xử lý thông tin người dùng             │
│  └── mcp_handler.py - Xử lý MCP protocol                    │
├─────────────────────────────────────────────────────────────┤
│  🆕 Enhanced Chunking Layer (chunking/)                     │
│  ├── semantic_chunker.py - LlamaIndex integration           │
│  ├── metadata_extractor.py - Extract structure metadata     │
│  ├── chunk_processor.py - Main orchestrator                 │
│  └── document_analyzer.py - Content type detection          │
├─────────────────────────────────────────────────────────────┤
│  🆕 Blob Storage Management (storage/)                      │
│  ├── blob_manager.py - Azure Blob Storage operations        │
│  ├── document_store.py - Document lifecycle management      │
│  ├── retrieval_service.py - On-demand document retrieval    │
│  └── metadata_indexer.py - Enhanced metadata management     │
├─────────────────────────────────────────────────────────────┤
│  Pre-Processing (pre_retrieval/)                            │
│  ├── analyze_query.py - Phân tích câu hỏi                   │
│  ├── decontextualize.py - Khử ngữ cảnh                      │
│  ├── relevance_detection.py - Kiểm tra độ liên quan         │
│  ├── memory.py - Quản lý bộ nhớ hội thoại                   │
│  └── required_info.py - Kiểm tra thông tin cần thiết        │
├─────────────────────────────────────────────────────────────┤
│  LLM Integration (llm/)                                     │
│  ├── llm_provider.py - Quản lý các nhà cung cấp LLM         │
│  ├── azure_oai.py - Azure OpenAI                           │
│  ├── openai.py - OpenAI                                    │
│  ├── anthropic.py - Anthropic Claude                       │
│  ├── gemini.py - Google Gemini                             │
│  ├── azure_deepseek.py - DeepSeek trên Azure               │
│  ├── azure_llama.py - Llama trên Azure                     │
│  ├── snowflake.py - Snowflake Cortex                       │
│  └── inception.py - Microsoft Inception                    │
├─────────────────────────────────────────────────────────────┤
│  Vector Database Integration (retrieval/)                   │
│  ├── retriever.py - Logic tìm kiếm chính                    │
│  ├── qdrant_retrieve.py - Qdrant vector database            │
│  ├── azure_search_client.py - Azure AI Search              │
│  ├── milvus_client.py - Milvus database                     │
│  └── snowflake_client.py - Snowflake vector search          │
├─────────────────────────────────────────────────────────────┤
│  Embedding Services (embedding/)                            │
│  ├── embedding.py - Logic embedding chính                   │
│  ├── azure_oai_embedding.py - Azure OpenAI embeddings      │
│  ├── openai_embedding.py - OpenAI embeddings               │
│  ├── gemini_embedding.py - Google embeddings               │
│  └── snowflake_embedding.py - Snowflake embeddings         │
├─────────────────────────────────────────────────────────────┤
│  Prompt Management (prompts/)                               │
│  ├── prompts.py - Quản lý và xử lý prompts                  │
│  ├── prompt_runner.py - Thực thi prompts                    │
│  └── site_type.xml - Định nghĩa prompts theo loại site      │
├─────────────────────────────────────────────────────────────┤
│  Configuration (config/)                                    │
│  ├── config.py - Xử lý cấu hình                            │
│  ├── config_llm.yaml - Cấu hình LLM providers              │
│  ├── config_embedding.yaml - Cấu hình embedding services    │
│  ├── config_retrieval.yaml - Cấu hình vector databases      │
│  ├── config_webserver.yaml - Cấu hình web server           │
│  ├── 🆕 config_storage.yaml - Cấu hình blob storage         │
│  └── config_nlweb.yaml - Cấu hình chung NLWeb              │
└─────────────────────────────────────────────────────────────┘
```

## 🆕 Enhanced Data Pipeline với Blob Storage & LlamaIndex

### New Document Processing Flow

```
Raw Document → Document Analysis → Semantic Chunking → Blob Storage → Vector DB
     ↓              ↓                    ↓               ↓            ↓
Original File → Content Type → LlamaIndex Chunks → Azure Blob → Embeddings
```

### 🆕 Enhanced Chunking Layer (chunking/)

#### semantic_chunker.py - LlamaIndex Integration
```python
from llama_index import Document
from llama_index.text_splitter import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter
)
from llama_index.embeddings import OpenAIEmbedding

class SemanticChunker:
    """LlamaIndex-powered semantic chunking with metadata preservation"""
    
    def __init__(self):
        # Initialize LlamaIndex components
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=OpenAIEmbedding()
        )
        
        self.sentence_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=256,
            chunk_overlap=25
        )
    
    async def chunk_document(self, document_content, document_type, metadata):
        """
        Chunk document based on content type with semantic awareness
        """
        # Create LlamaIndex document
        doc = Document(text=document_content, metadata=metadata)
        
        # Choose chunking strategy based on document type
        if document_type == "recipe":
            return await self._chunk_recipe(doc)
        elif document_type == "article":
            return await self._chunk_article(doc)
        elif document_type == "product":
            return await self._chunk_product(doc)
        else:
            return await self._chunk_generic(doc)
    
    async def _chunk_recipe(self, doc):
        """Recipe-specific chunking strategy"""
        chunks = []
        schema_obj = json.loads(doc.text) if isinstance(doc.text, str) else doc.text
        
        # Chunk 1: Recipe Overview
        if "name" in schema_obj and "description" in schema_obj:
            overview_text = f"{schema_obj['name']}\n{schema_obj.get('description', '')}"
            chunks.append({
                "text": overview_text,
                "chunk_type": "recipe_overview",
                "metadata": {
                    "section": "overview",
                    "recipe_name": schema_obj["name"],
                    "cuisine": schema_obj.get("recipeCuisine", ""),
                    "category": schema_obj.get("recipeCategory", "")
                }
            })
        
        # Chunk 2: Ingredients (semantic grouping)
        if "recipeIngredient" in schema_obj:
            ingredients_text = "\n".join(schema_obj["recipeIngredient"])
            # Use semantic splitter to group related ingredients
            ingredient_nodes = self.semantic_splitter.get_nodes_from_documents([
                Document(text=ingredients_text)
            ])
            
            for i, node in enumerate(ingredient_nodes):
                chunks.append({
                    "text": node.text,
                    "chunk_type": "ingredients",
                    "metadata": {
                        "section": "ingredients",
                        "ingredient_group": i + 1,
                        "total_groups": len(ingredient_nodes)
                    }
                })
        
        # Chunk 3: Instructions (step-by-step)
        if "recipeInstructions" in schema_obj:
            for i, instruction in enumerate(schema_obj["recipeInstructions"]):
                instruction_text = instruction.get("text", str(instruction))
                chunks.append({
                    "text": instruction_text,
                    "chunk_type": "instruction_step",
                    "metadata": {
                        "section": "instructions",
                        "step_number": i + 1,
                        "total_steps": len(schema_obj["recipeInstructions"])
                    }
                })
        
        # Chunk 4: Nutrition & Additional Info
        nutrition_fields = ["nutrition", "recipeYield", "prepTime", "cookTime"]
        nutrition_info = {}
        for field in nutrition_fields:
            if field in schema_obj:
                nutrition_info[field] = schema_obj[field]
        
        if nutrition_info:
            chunks.append({
                "text": json.dumps(nutrition_info, indent=2),
                "chunk_type": "nutrition_info",
                "metadata": {
                    "section": "nutrition",
                    "data_type": "structured"
                }
            })
        
        return chunks
    
    async def _chunk_article(self, doc):
        """Article-specific chunking with heading awareness"""
        # Use semantic splitter for natural paragraph breaks
        nodes = self.semantic_splitter.get_nodes_from_documents([doc])
        
        chunks = []
        for i, node in enumerate(nodes):
            # Detect if chunk contains heading
            is_heading = self._detect_heading(node.text)
            
            chunks.append({
                "text": node.text,
                "chunk_type": "article_section" if is_heading else "article_content",
                "metadata": {
                    "section": f"section_{i+1}",
                    "is_heading": is_heading,
                    "word_count": len(node.text.split()),
                    "position": i + 1,
                    "total_chunks": len(nodes)
                }
            })
        
        return chunks
    
    def _detect_heading(self, text):
        """Simple heading detection logic"""
        lines = text.strip().split('\n')
        first_line = lines[0] if lines else ""
        
        # Common heading patterns
        heading_patterns = [
            lambda x: x.isupper() and len(x.split()) <= 8,  # ALL CAPS
            lambda x: x.startswith('#'),  # Markdown headers
            lambda x: len(x) < 100 and x.endswith(':'),  # Short line ending with colon
            lambda x: not x.endswith('.') and len(x.split()) <= 10  # Short without period
        ]
        
        return any(pattern(first_line) for pattern in heading_patterns)
```

#### chunk_processor.py - Main Orchestrator
```python
class ChunkProcessor:
    """Main orchestrator for document chunking pipeline"""
    
    def __init__(self, blob_manager, embedding_service):
        self.semantic_chunker = SemanticChunker()
        self.blob_manager = blob_manager
        self.embedding_service = embedding_service
        self.document_analyzer = DocumentAnalyzer()
    
    async def process_document(self, document_data, original_metadata):
        """
        Complete document processing pipeline:
        1. Analyze document type
        2. Semantic chunking with LlamaIndex
        3. Store original in blob storage
        4. Generate embeddings for chunks
        5. Store chunks in vector DB with blob references
        """
        
        # Step 1: Analyze document
        doc_analysis = await self.document_analyzer.analyze(document_data)
        
        # Step 2: Store original document in blob storage
        blob_info = await self.blob_manager.store_original_document(
            content=document_data,
            metadata={
                **original_metadata,
                "document_type": doc_analysis["type"],
                "processing_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Step 3: Semantic chunking with LlamaIndex
        chunks = await self.semantic_chunker.chunk_document(
            document_content=document_data,
            document_type=doc_analysis["type"],
            metadata=original_metadata
        )
        
        # Step 4: Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await self.embedding_service.get_embedding(chunk["text"])
            
            # Enhanced metadata with blob reference
            enhanced_metadata = {
                **original_metadata,
                "chunk_id": f"{original_metadata['url']}#chunk_{i}",
                "chunk_type": chunk["chunk_type"],
                "chunk_metadata": chunk["metadata"],
                "chunk_position": i,
                "total_chunks": len(chunks),
                "original_blob_url": blob_info["blob_url"],
                "parent_document_id": blob_info["document_id"],
                "semantic_chunking_version": "llamaindex_v1.0"
            }
            
            processed_chunk = {
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": enhanced_metadata
            }
            
            processed_chunks.append(processed_chunk)
        
        # Step 5: Store chunk references in blob
        await self.blob_manager.store_chunk_index(
            document_id=blob_info["document_id"],
            chunks_metadata=[chunk["metadata"] for chunk in processed_chunks]
        )
        
        return processed_chunks
```

### 🆕 Blob Storage Management (storage/)

#### blob_manager.py - Azure Blob Storage Operations
```python
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
import json
import uuid
from datetime import datetime

class BlobStorageManager:
    """Azure Blob Storage integration for document lifecycle management"""
    
    def __init__(self, connection_string, container_documents="documents", 
                 container_chunks="chunks", container_metadata="metadata"):
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_documents = container_documents
        self.container_chunks = container_chunks
        self.container_metadata = container_metadata
        
        # Ensure containers exist
        self._ensure_containers()
    
    def _ensure_containers(self):
        """Create containers if they don't exist"""
        containers = [self.container_documents, self.container_chunks, self.container_metadata]
        for container in containers:
            try:
                self.blob_service.create_container(container)
            except Exception:
                pass  # Container might already exist
    
    async def store_original_document(self, content, metadata):
        """Store original document with metadata"""
        document_id = str(uuid.uuid4())
        blob_name = f"{metadata['site']}/{document_id}.json"
        
        # Prepare document package
        document_package = {
            "document_id": document_id,
            "original_metadata": metadata,
            "content": content,
            "stored_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        # Upload to blob storage
        blob_client = self.blob_service.get_blob_client(
            container=self.container_documents,
            blob=blob_name
        )
        
        await blob_client.upload_blob(
            json.dumps(document_package, ensure_ascii=False),
            overwrite=True,
            metadata={
                "document_id": document_id,
                "site": metadata["site"],
                "document_type": metadata.get("document_type", "unknown"),
                "content_type": "application/json"
            }
        )
        
        blob_url = blob_client.url
        
        return {
            "document_id": document_id,
            "blob_url": blob_url,
            "blob_name": blob_name,
            "container": self.container_documents
        }
    
    async def store_chunk_index(self, document_id, chunks_metadata):
        """Store chunk index for fast chunk-to-document mapping"""
        chunk_index = {
            "document_id": document_id,
            "chunks": chunks_metadata,
            "total_chunks": len(chunks_metadata),
            "created_at": datetime.utcnow().isoformat()
        }
        
        blob_name = f"indexes/{document_id}_chunks.json"
        blob_client = self.blob_service.get_blob_client(
            container=self.container_metadata,
            blob=blob_name
        )
        
        await blob_client.upload_blob(
            json.dumps(chunk_index, ensure_ascii=False),
            overwrite=True
        )
    
    async def retrieve_original_document(self, blob_url_or_document_id):
        """Retrieve original document by blob URL or document ID"""
        try:
            if blob_url_or_document_id.startswith("http"):
                # It's a blob URL
                blob_client = BlobClient.from_blob_url(blob_url_or_document_id)
            else:
                # It's a document ID, construct blob path
                # This requires metadata lookup or consistent naming
                blob_name = await self._find_blob_by_document_id(blob_url_or_document_id)
                blob_client = self.blob_service.get_blob_client(
                    container=self.container_documents,
                    blob=blob_name
                )
            
            content = await blob_client.download_blob()
            document_package = json.loads(content.readall())
            
            return document_package
            
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Document not found: {blob_url_or_document_id}")
    
    async def retrieve_chunk_context(self, document_id, chunk_id):
        """Retrieve surrounding chunks for better context"""
        try:
            # Get chunk index
            blob_name = f"indexes/{document_id}_chunks.json"
            blob_client = self.blob_service.get_blob_client(
                container=self.container_metadata,
                blob=blob_name
            )
            
            content = await blob_client.download_blob()
            chunk_index = json.loads(content.readall())
            
            # Find target chunk and get surrounding context
            chunks = chunk_index["chunks"]
            target_position = None
            
            for i, chunk in enumerate(chunks):
                if chunk.get("chunk_id") == chunk_id:
                    target_position = i
                    break
            
            if target_position is not None:
                # Return current chunk + 1 before + 1 after for context
                start_idx = max(0, target_position - 1)
                end_idx = min(len(chunks), target_position + 2)
                context_chunks = chunks[start_idx:end_idx]
                
                return {
                    "target_chunk": chunks[target_position],
                    "context_chunks": context_chunks,
                    "position": target_position,
                    "total_chunks": len(chunks)
                }
            
            return None
            
        except ResourceNotFoundError:
            return None
```

#### document_store.py - Document Lifecycle Management
```python
class DocumentStore:
    """High-level document storage and retrieval interface"""
    
    def __init__(self, blob_manager, vector_db, chunk_processor):
        self.blob_manager = blob_manager
        self.vector_db = vector_db
        self.chunk_processor = chunk_processor
    
    async def ingest_document(self, document_data, metadata):
        """Complete document ingestion pipeline"""
        try:
            # Process document with enhanced chunking
            processed_chunks = await self.chunk_processor.process_document(
                document_data, metadata
            )
            
            # Store chunks in vector database
            storage_results = []
            for chunk in processed_chunks:
                result = await self.vector_db.store(
                    vector=chunk["embedding"],
                    metadata=chunk["metadata"],
                    content=chunk["text"]
                )
                storage_results.append(result)
            
            return {
                "status": "success",
                "document_id": processed_chunks[0]["metadata"]["parent_document_id"],
                "chunks_stored": len(processed_chunks),
                "storage_results": storage_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "document_metadata": metadata
            }
    
    async def retrieve_document_with_context(self, document_id, include_chunks=True):
        """Retrieve document with optional chunk context"""
        # Get original document
        original_doc = await self.blob_manager.retrieve_original_document(document_id)
        
        result = {
            "document": original_doc,
            "document_id": document_id
        }
        
        if include_chunks:
            # Get chunk information
            chunk_index_blob = f"indexes/{document_id}_chunks.json"
            try:
                chunk_info = await self.blob_manager.retrieve_chunk_index(document_id)
                result["chunks_info"] = chunk_info
            except FileNotFoundError:
                result["chunks_info"] = None
        
        return result
```

## 🆕 Enhanced Retrieval với Blob Storage Integration

### Enhanced retriever.py
```python
class EnhancedRetriever:
    """Enhanced retrieval with blob storage and chunk context"""
    
    def __init__(self, vector_db, blob_manager, embedding_service):
        self.vector_db = vector_db
        self.blob_manager = blob_manager
        self.embedding_service = embedding_service
    
    async def search_with_context(self, query, site=None, context_mode="chunk"):
        """
        Enhanced search with multiple context modes:
        - chunk: Return individual chunks (default)
        - document: Return full documents
        - hybrid: Return chunks with document context
        """
        
        # Generate query embedding
        query_embedding = await self.embedding_service.get_embedding(query)
        
        # Vector search for chunks
        chunk_results = await self.vector_db.search(
            vector=query_embedding,
            filter={"site": site} if site else {},
            limit=50
        )
        
        if context_mode == "chunk":
            return await self._process_chunk_results(chunk_results)
        elif context_mode == "document":
            return await self._process_document_results(chunk_results)
        elif context_mode == "hybrid":
            return await self._process_hybrid_results(chunk_results)
        else:
            raise ValueError(f"Unknown context_mode: {context_mode}")
    
    async def _process_chunk_results(self, chunk_results):
        """Process results as individual chunks"""
        processed_results = []
        
        for result in chunk_results:
            metadata = result["metadata"]
            
            # Get chunk context if available
            context = None
            if "parent_document_id" in metadata and "chunk_id" in metadata:
                context = await self.blob_manager.retrieve_chunk_context(
                    metadata["parent_document_id"],
                    metadata["chunk_id"]
                )
            
            processed_result = {
                "type": "chunk",
                "url": metadata.get("url"),
                "name": metadata.get("name"),
                "text": result["content"],
                "chunk_type": metadata.get("chunk_type"),
                "chunk_metadata": metadata.get("chunk_metadata", {}),
                "score": result.get("score", 0),
                "source_blob_url": metadata.get("original_blob_url"),
                "context": context
            }
            
            processed_results.append(processed_result)
        
        return processed_results
    
    async def _process_hybrid_results(self, chunk_results):
        """Process results with both chunk and document context"""
        # Group chunks by document
        documents = {}
        for result in chunk_results:
            doc_id = result["metadata"].get("parent_document_id")
            if doc_id:
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "chunks": [],
                        "metadata": result["metadata"]
                    }
                documents[doc_id]["chunks"].append(result)
        
        # Enrich with document context
        enriched_results = []
        for doc_id, doc_data in documents.items():
            # Get original document for context
            try:
                original_doc = await self.blob_manager.retrieve_original_document(doc_id)
                doc_data["original_document"] = original_doc
            except FileNotFoundError:
                doc_data["original_document"] = None
            
            enriched_results.append(doc_data)
        
        return enriched_results
```

## 🆕 Enhanced Answer Generation với Source Attribution

### Enhanced generate_answer.py
```python
class EnhancedGenerateAnswer(NLWebHandler):
    """Enhanced answer generation with rich source attribution"""
    
    async def synthesizeAnswer(self):
        """Generate answer with detailed source citations"""
        
        # Get relevant chunks with context
        relevant_chunks = await self.enhanced_retriever.search_with_context(
            query=self.decontextualized_query,
            site=self.site,
            context_mode="hybrid"
        )
        
        # Build enhanced prompt with source attribution
        prompt = await self._build_enhanced_prompt(relevant_chunks)
        
        # Generate answer with citations
        response = await PromptRunner(self).run_prompt_with_sources(
            "EnhancedSynthesizePrompt", 
            sources=relevant_chunks
        )
        
        # Format response with rich citations
        formatted_response = await self._format_response_with_citations(
            response, relevant_chunks
        )
        
        # Send enhanced message
        message = {
            "message_type": "nlws_enhanced",
            "answer": formatted_response["answer"],
            "citations": formatted_response["citations"],
            "source_documents": formatted_response["source_documents"],
            "metadata": {
                "chunking_method": "llamaindex_semantic",
                "context_mode": "hybrid",
                "sources_count": len(relevant_chunks)
            }
        }
        
        await self.send_message(message)
    
    async def _build_enhanced_prompt(self, relevant_chunks):
        """Build prompt with rich source context"""
        sources_context = []
        
        for i, chunk_group in enumerate(relevant_chunks):
            if chunk_group["type"] == "document_group":
                doc_info = f"""
Source {i+1}: {chunk_group['metadata']['name']}
URL: {chunk_group['metadata']['url']}
Document Type: {chunk_group['metadata'].get('document_type', 'unknown')}

Relevant Sections:
"""
                for chunk in chunk_group["chunks"]:
                    doc_info += f"- {chunk['chunk_type']}: {chunk['text'][:200]}...\n"
                
                sources_context.append(doc_info)
        
        return {
            "query": self.decontextualized_query,
            "sources": "\n".join(sources_context),
            "instruction": """
            Based on the provided sources, answer the user's question.
            Include citations using [1], [2], etc. format.
            Ensure each claim is properly attributed to its source.
            """
        }
    
    async def _format_response_with_citations(self, response, relevant_chunks):
        """Format response with detailed citations"""
        citations = []
        source_documents = []
        
        for i, chunk_group in enumerate(relevant_chunks):
            citation = {
                "id": i + 1,
                "url": chunk_group["metadata"]["url"],
                "title": chunk_group["metadata"]["name"],
                "site": chunk_group["metadata"]["site"],
                "document_type": chunk_group["metadata"].get("document_type"),
                "blob_url": chunk_group["metadata"].get("original_blob_url"),
                "relevant_chunks": []
            }
            
            # Add chunk details
            for chunk in chunk_group.get("chunks", []):
                chunk_detail = {
                    "chunk_type": chunk["chunk_type"],
                    "text_preview": chunk["text"][:150] + "...",
                    "chunk_metadata": chunk.get("chunk_metadata", {})
                }
                citation["relevant_chunks"].append(chunk_detail)
            
            citations.append(citation)
            
            # Store source document reference
            source_documents.append({
                "document_id": chunk_group.get("document_id"),
                "blob_url": chunk_group["metadata"].get("original_blob_url"),
                "access_url": chunk_group["metadata"]["url"]
            })
        
        return {
            "answer": response["answer"],
            "citations": citations,
            "source_documents": source_documents
        }
```

## 🆕 Enhanced Configuration

### config/config_storage.yaml
```yaml
blob_storage:
  provider: azure_blob
  connection_string_env: AZURE_STORAGE_CONNECTION_STRING
  
  containers:
    documents: "nlweb-documents"      # Original documents
    chunks: "nlweb-chunks"            # Processed chunks
    metadata: "nlweb-metadata"        # Indexes and metadata
  
  # Retention and lifecycle policies
  lifecycle:
    hot_tier_days: 30
    cool_tier_days: 90
    archive_tier_days: 365
    delete_after_days: 2555  # 7 years
  
  # Performance settings
  performance:
    max_concurrent_uploads: 10
    chunk_size_mb: 4
    enable_compression: true
    
llamaindex:
  chunking:
    semantic_splitter:
      buffer_size: 1
      breakpoint_percentile_threshold: 95
      min_chunk_size: 100
      max_chunk_size: 1000
    
    sentence_splitter:
      chunk_size: 512
      chunk_overlap: 50
    
    token_splitter:
      chunk_size: 256
      chunk_overlap: 25
  
  # Document type specific settings
  document_types:
    recipe:
      chunking_strategy: "structured"
      preserve_schema: true
      chunk_ingredients_separately: true
      chunk_instructions_by_step: true
    
    article:
      chunking_strategy: "semantic"
      respect_headings: true
      min_paragraph_length: 50
    
    product:
      chunking_strategy: "hybrid"
      separate_specifications: true
      group_reviews: true
```

### Enhanced tools/db_load.py
```python
# 🆕 Enhanced db_load.py with blob storage and LlamaIndex integration

from chunking.chunk_processor import ChunkProcessor
from storage.blob_manager import BlobStorageManager
from storage.document_store import DocumentStore
import asyncio
import json
from datetime import datetime

class EnhancedDataLoader:
    """Enhanced data loader with blob storage and semantic chunking"""
    
    def __init__(self, config):
        # Initialize services
        self.blob_manager = BlobStorageManager(
            connection_string=config.blob_storage.connection_string,
            container_documents=config.blob_storage.containers.documents,
            container_chunks=config.blob_storage.containers.chunks,
            container_metadata=config.blob_storage.containers.metadata
        )
        
        self.chunk_processor = ChunkProcessor(
            blob_manager=self.blob_manager,
            embedding_service=self.embedding_service
        )
        
        self.document_store = DocumentStore(
            blob_manager=self.blob_manager,
            vector_db=self.vector_db,
            chunk_processor=self.chunk_processor
        )
    
    async def load_with_enhanced_processing(self, json_file, batch_size=10):
        """
        Enhanced loading with:
        1. Blob storage for originals
        2. LlamaIndex semantic chunking
        3. Batch processing for performance
        4. Error handling and recovery
        """
        
        print(f"🚀 Starting enhanced data loading from {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_items = len(data)
        processed_count = 0
        error_count = 0
        
        # Process in batches for better performance
        for i in range(0, total_items, batch_size):
            batch = data[i:i + batch_size]
            batch_tasks = []
            
            for item in batch:
                task = self._process_single_item(item)
                batch_tasks.append(task)
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_count += 1
                    print(f"❌ Error processing item {i+j}: {str(result)}")
                else:
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"✅ Processed {processed_count}/{total_items} items")
        
        print(f"🎉 Loading complete: {processed_count} successful, {error_count} errors")
        
        return {
            "total_items": total_items,
            "processed": processed_count,
            "errors": error_count,
            "success_rate": processed_count / total_items if total_items > 0 else 0
        }
    
    async def _process_single_item(self, item):
        """Process a single item with enhanced pipeline"""
        try:
            # Prepare metadata
            metadata = {
                "url": item.get("url", ""),
                "name": item.get("name", ""),
                "site": item.get("site", "unknown"),
                "document_type": self._detect_document_type(item),
                "original_schema": item.get("@type", "Thing"),
                "processing_timestamp": datetime.utcnow().isoformat(),
                "loader_version": "enhanced_v1.0"
            }
            
            # Use document store for complete processing
            result = await self.document_store.ingest_document(
                document_data=item,
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            # Log error details for debugging
            error_details = {
                "item_url": item.get("url", "unknown"),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Optionally store failed items for retry
            await self._store_failed_item(item, error_details)
            
            raise e
    
    def _detect_document_type(self, item):
        """Detect document type from schema.org type"""
        schema_type = item.get("@type", "").lower()
        
        type_mapping = {
            "recipe": "recipe",
            "article": "article", 
            "newsarticle": "article",
            "blogposting": "article",
            "product": "product",
            "event": "event",
            "organization": "organization",
            "person": "person",
            "place": "place"
        }
        
        return type_mapping.get(schema_type, "generic")
    
    async def _store_failed_item(self, item, error_details):
        """Store failed items for later retry"""
        try:
            failed_item_data = {
                "original_item": item,
                "error_details": error_details,
                "retry_count": 0,
                "max_retries": 3
            }
            
            blob_name = f"failed_items/{datetime.utcnow().strftime('%Y%m%d')}/{error_details['timestamp']}.json"
            
            blob_client = self.blob_manager.blob_service.get_blob_client(
                container="nlweb-failed-items",
                blob=blob_name
            )
            
            await blob_client.upload_blob(
                json.dumps(failed_item_data, ensure_ascii=False),
                overwrite=True
            )
            
        except Exception:
            # If we can't even store the failed item, just log it
            print(f"Failed to store failed item: {item.get('url', 'unknown')}")

    async def retry_failed_items(self, date_str=None):
        """Retry processing failed items from a specific date"""
        if not date_str:
            date_str = datetime.utcnow().strftime('%Y%m%d')
        
        try:
            # List failed items for the date
            container_client = self.blob_manager.blob_service.get_container_client("nlweb-failed-items")
            blob_list = container_client.list_blobs(name_starts_with=f"failed_items/{date_str}/")
            
            retry_count = 0
            success_count = 0
            
            for blob in blob_list:
                blob_client = container_client.get_blob_client(blob.name)
                content = await blob_client.download_blob()
                failed_item_data = json.loads(content.readall())
                
                if failed_item_data["retry_count"] < failed_item_data["max_retries"]:
                    try:
                        # Attempt to process again
                        result = await self._process_single_item(failed_item_data["original_item"])
                        
                        if result["status"] == "success":
                            success_count += 1
                            # Delete the failed item record
                            await blob_client.delete_blob()
                        
                    except Exception as e:
                        # Update retry count
                        failed_item_data["retry_count"] += 1
                        failed_item_data["last_retry_error"] = str(e)
                        failed_item_data["last_retry_timestamp"] = datetime.utcnow().isoformat()
                        
                        # Update the failed item record
                        await blob_client.upload_blob(
                            json.dumps(failed_item_data, ensure_ascii=False),
                            overwrite=True
                        )
                
                retry_count += 1
            
            print(f"🔄 Retry complete: {success_count}/{retry_count} items recovered")
            
            return {
                "total_retried": retry_count,
                "successful": success_count,
                "recovery_rate": success_count / retry_count if retry_count > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error during retry operation: {str(e)}")
            return {"error": str(e)}
```

## Quy Trình Xử Lý Một Câu Hỏi (Enhanced Life of a Chat Query)

### Bước 1: Tiếp Nhận Query (Unchanged)
```
User Input → Web Interface → WebServer → NLWebHandler
```

### Bước 2: Khởi Tạo Enhanced Handler

```python
class EnhancedNLWebHandler(NLWebHandler):
    def __init__(self, query_params, http_handler):
        # Original initialization
        super().__init__(query_params, http_handler)
        
        # 🆕 Enhanced services
        self.blob_manager = BlobStorageManager(config.blob_storage)
        self.enhanced_retriever = EnhancedRetriever(
            vector_db=self.vector_db,
            blob_manager=self.blob_manager,
            embedding_service=self.embedding_service
        )
        
        # 🆕 Context mode selection
        self.context_mode = get_param(query_params, "context_mode", str, "hybrid")
        self.include_source_documents = get_param(query_params, "include_sources", bool, True)
```

### 🆕 Bước 3: Enhanced Preparation Phase

Same parallel processing as original, but with enhanced chunking awareness:

```python
# Enhanced preparation tasks
tasks = [
    asyncio.create_task(analyze_query.DetectItemType(self).do()),
    asyncio.create_task(self.decontextualizeQuery().do()),
    asyncio.create_task(relevance_detection.RelevanceDetection(self).do()),
    asyncio.create_task(memory.Memory(self).do()),
    asyncio.create_task(required_info.RequiredInfo(self).do()),
    # 🆕 New: Chunk strategy selection
    asyncio.create_task(self._select_optimal_chunk_strategy())
]

await asyncio.gather(*tasks, return_exceptions=True)
```

### 🆕 Bước 4: Enhanced Retrieval Phase

```python
# Enhanced retrieval with context awareness
async def enhanced_retrieval(self):
    # Vector search với chunk context
    chunk_results = await self.enhanced_retriever.search_with_context(
        query=self.decontextualized_query,
        site=self.site,
        context_mode=self.context_mode
    )
    
    # 🆕 Intelligent result augmentation
    if self.should_include_document_context():
        augmented_results = await self._augment_with_document_context(chunk_results)
        return augmented_results
    
    return chunk_results
```

### 🆕 Bước 5: Enhanced Ranking với Chunk Awareness

```python
class EnhancedRanking(Ranking):
    async def rankChunkWithContext(self, chunk_result):
        # Original chunk ranking
        base_ranking = await super().rankItem(
            chunk_result["url"],
            chunk_result["text"], 
            chunk_result["name"],
            chunk_result["site"]
        )
        
        # 🆕 Context-aware scoring boost
        context_boost = 0
        if chunk_result.get("context"):
            context_boost = await self._calculate_context_relevance(
                chunk_result["context"],
                self.handler.query
            )
        
        # 🆕 Chunk type relevance
        chunk_type_score = await self._score_chunk_type_relevance(
            chunk_result["chunk_type"],
            self.handler.query
        )
        
        # Combined scoring
        enhanced_score = (
            base_ranking["score"] * 0.7 +
            context_boost * 0.2 +
            chunk_type_score * 0.1
        )
        
        return {
            **base_ranking,
            "enhanced_score": enhanced_score,
            "context_boost": context_boost,
            "chunk_type_score": chunk_type_score,
            "chunk_metadata": chunk_result.get("chunk_metadata", {})
        }
```

### 🆕 Bước 6: Enhanced Response Generation

```python
# Enhanced streaming response with rich metadata
async def send_enhanced_message(self, message):
    # Add processing metadata
    enhanced_message = {
        **message,
        "processing_metadata": {
            "chunking_method": "llamaindex_semantic",
            "context_mode": self.context_mode,
            "blob_storage_enabled": True,
            "processing_time_ms": self.get_processing_time(),
            "chunks_processed": len(self.processed_chunks),
            "documents_accessed": len(self.accessed_documents)
        }
    }
    
    await super().send_message(enhanced_message)
```

## 🆕 Enhanced Data Storage Format

### Vector Database Schema (Enhanced)
```json
{
  "url": "https://example.com/recipe/pho",
  "name": "Authentic Vietnamese Pho",
  "site": "example.com",
  "text": "Chunk content here...",
  "embedding": [0.1, -0.2, 0.3, ...],
  
  // 🆕 Enhanced metadata
  "chunk_id": "doc123#chunk_2",
  "chunk_type": "ingredients",
  "chunk_position": 2,
  "total_chunks": 5,
  "parent_document_id": "doc123",
  "original_blob_url": "https://storage.blob.core.windows.net/documents/example.com/doc123.json",
  
  // 🆕 LlamaIndex metadata
  "chunk_metadata": {
    "section": "ingredients",
    "ingredient_group": 1,
    "semantic_similarity_threshold": 0.85
  },
  
  // 🆕 Processing metadata
  "semantic_chunking_version": "llamaindex_v1.0",
  "processing_timestamp": "2025-06-03T10:30:00Z",
  "document_type": "recipe"
}
```

### 🆕 Blob Storage Document Package
```json
{
  "document_id": "doc123",
  "original_metadata": {
    "url": "https://example.com/recipe/pho",
    "name": "Authentic Vietnamese Pho",
    "site": "example.com"
  },
  "content": {
    "@context": "https://schema.org",
    "@type": "Recipe",
    "name": "Authentic Vietnamese Pho",
    "recipeIngredient": [...],
    "recipeInstructions": [...]
  },
  "processing_info": {
    "chunks_generated": 5,
    "chunking_strategy": "semantic_recipe",
    "llamaindex_version": "0.10.0",
    "processing_timestamp": "2025-06-03T10:30:00Z"
  },
  "version": "1.0",
  "stored_at": "2025-06-03T10:30:00Z"
}
```

### 🆕 Chunk Index (Metadata Container)
```json
{
  "document_id": "doc123",
  "chunks": [
    {
      "chunk_id": "doc123#chunk_0",
      "chunk_type": "recipe_overview",
      "position": 0,
      "text_preview": "Authentic Vietnamese Pho - Traditional noodle soup...",
      "metadata": {
        "section": "overview",
        "recipe_name": "Authentic Vietnamese Pho"
      }
    },
    {
      "chunk_id": "doc123#chunk_1", 
      "chunk_type": "ingredients",
      "position": 1,
      "text_preview": "1 lb beef bones, 1 onion, 2 star anise...",
      "metadata": {
        "section": "ingredients",
        "ingredient_group": 1
      }
    }
  ],
  "total_chunks": 5,
  "created_at": "2025-06-03T10:30:00Z"
}
```

## Performance Optimizations (Enhanced)

### 🆕 Smart Caching Strategy
```python
class EnhancedCaching:
    def __init__(self):
        # Multi-tier caching
        self.embedding_cache = {}        # In-memory embedding cache
        self.document_cache = {}         # Document metadata cache
        self.chunk_context_cache = {}    # Chunk context cache
        self.blob_url_cache = {}         # Blob URL resolution cache
    
    async def get_cached_document_context(self, document_id):
        """Get document context with intelligent caching"""
        if document_id in self.document_cache:
            cached_entry = self.document_cache[document_id]
            if not self._is_cache_expired(cached_entry):
                return cached_entry["data"]
        
        # Cache miss - fetch and cache
        context = await self.blob_manager.retrieve_document_context(document_id)
        self.document_cache[document_id] = {
            "data": context,
            "cached_at": time.time(),
            "ttl": 3600  # 1 hour
        }
        
        return context
```

### 🆕 Parallel Processing with Blob Operations
```python
async def parallel_document_enrichment(self, chunk_results):
    """Enrich chunks with document context in parallel"""
    
    # Group by document to minimize blob storage calls
    documents_needed = set()
    for chunk in chunk_results:
        if chunk.get("parent_document_id"):
            documents_needed.add(chunk["parent_document_id"])
    
    # Parallel fetch document contexts
    fetch_tasks = [
        self.blob_manager.retrieve_document_context(doc_id)
        for doc_id in documents_needed
    ]
    
    document_contexts = await asyncio.gather(*fetch_tasks, return_exceptions=True)
    
    # Build document context map
    context_map = {}
    for doc_id, context in zip(documents_needed, document_contexts):
        if not isinstance(context, Exception):
            context_map[doc_id] = context
    
    # Enrich chunks with context
    enriched_chunks = []
    for chunk in chunk_results:
        doc_id = chunk.get("parent_document_id")
        if doc_id in context_map:
            chunk["document_context"] = context_map[doc_id]
        enriched_chunks.append(chunk)
    
    return enriched_chunks
```

## 🆕 Monitoring & Analytics (Enhanced)

### Enhanced Performance Metrics
```python
# Enhanced metrics tracking
enhanced_metrics = {
    # Original metrics
    "query_processing_time": time.time() - start_time,
    "llm_calls_count": self.llm_call_counter,
    "items_retrieved": len(self.final_retrieved_items),
    
    # 🆕 Enhanced metrics
    "chunks_processed": len(self.processed_chunks),
    "documents_accessed": len(self.accessed_documents),
    "blob_storage_calls": self.blob_call_counter,
    "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses),
    "chunking_strategy": self.selected_chunking_strategy,
    "context_mode": self.context_mode,
    "average_chunk_relevance": self.calculate_average_relevance(),
    
    # Processing breakdown
    "time_breakdown": {
        "chunking_time": self.chunking_time,
        "blob_retrieval_time": self.blob_retrieval_time,
        "vector_search_time": self.vector_search_time,
        "context_enrichment_time": self.context_enrichment_time
    }
}
```

## 🆕 Cost Optimization Strategies

### Intelligent Storage Tiering
```python
class StorageTierManager:
    """Manage blob storage tiers for cost optimization"""
    
    async def optimize_document_storage(self):
        """Move documents to appropriate storage tiers"""
        
        # Recent documents (< 30 days) - Hot tier
        # Moderate access (30-90 days) - Cool tier  
        # Archive (> 90 days) - Cold tier
        
        containers = [self.container_documents, self.container_metadata]
        
        for container in containers:
            blobs = await self._list_blobs_with_metadata(container)
            
            for blob in blobs:
                age_days = self._calculate_blob_age(blob)
                access_frequency = await self._get_access_frequency(blob)
                
                optimal_tier = self._determine_optimal_tier(age_days, access_frequency)
                
                if blob.tier != optimal_tier:
                    await self._move_to_tier(blob, optimal_tier)
```

## Kết Luận (Enhanced)

NLWeb Enhanced với Blob Storage và LlamaIndex mang lại những cải tiến đáng kể:

### 🆕 Lợi Ích Mới:

1. **Chunking Thông Minh**: LlamaIndex semantic chunking tăng độ chính xác tìm kiếm
2. **Source Traceability**: Luôn có khả năng truy xuất document gốc
3. **Scalable Storage**: Blob storage cho phép lưu trữ unlimited documents
4. **Rich Citations**: Citations với context và metadata chi tiết
5. **Cost Optimization**: Intelligent storage tiering và caching
6. **Enhanced Context**: Chunk-level và document-level context

### 🆕 Use Cases Mới:

- **Legal Documents**: Chunk theo sections với traceability
- **Academic Papers**: Semantic chunking theo structure
- **E-commerce**: Product specs với rich metadata
- **Knowledge Bases**: Hierarchical content organization

### 🆕 Architecture Benefits:

- **Hybrid Approach**: Kết hợp vector search với document storage
- **Flexible Retrieval**: Multiple context modes (chunk/document/hybrid)
- **Audit Trail**: Complete lineage từ query đến source
- **Future-Proof**: Ready cho AI compliance và regulations

Hệ thống này đặc biệt phù hợp cho các organizations cần **high accuracy, source attribution, và scalability** trong conversational AI applications.