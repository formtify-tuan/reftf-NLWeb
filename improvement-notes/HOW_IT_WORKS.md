# Cách Hoạt Động Của NLWeb

## Tổng Quan

NLWeb là một framework mã nguồn mở được Microsoft phát triển để tạo ra các giao diện trò chuyện tự nhiên (conversational interfaces) cho websites. Hệ thống này hoạt động như một "search engine thông minh" với khả năng hiểu và trả lời các câu hỏi bằng ngôn ngữ tự nhiên.

## Kiến Trúc Tổng Thể

### Các Thành Phần Chính

```
┌─────────────────────────────────────────────────────────────┐
│                    NLWeb Architecture                        │
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
│  └── config_nlweb.yaml - Cấu hình chung NLWeb              │
└─────────────────────────────────────────────────────────────┘
```

## Quy Trình Xử Lý Một Câu Hỏi (Life of a Chat Query)

### Bước 1: Tiếp Nhận Query
```
User Input → Web Interface → WebServer → NLWebHandler
```

Khi người dùng gửi một câu hỏi:
- Web interface nhận input từ người dùng
- WebServer.py xử lý HTTP request
- Tạo một instance của NLWebHandler để xử lý query

### Bước 2: Khởi Tạo Handler (baseHandler.py)

```python
class NLWebHandler:
    def __init__(self, query_params, http_handler):
        # Các thông số cơ bản
        self.site = get_param(query_params, "site", str, "all")
        self.query = get_param(query_params, "query", str, "")
        self.prev_queries = get_param(query_params, "prev", list, [])
        self.model = get_param(query_params, "model", str, "gpt-4o-mini")
        self.decontextualized_query = get_param(query_params, "decontextualized_query", str, "")
        self.context_url = get_param(query_params, "context_url", str, "")
        self.generate_mode = get_param(query_params, "generate_mode", str, "none")
        
        # Khởi tạo state management
        self.state = NLWebHandlerState(self)
        
        # Khởi tạo các event đồng bộ hóa
        self.pre_checks_done_event = asyncio.Event()
        self.retrieval_done_event = asyncio.Event()
        self.connection_alive_event = asyncio.Event()
```

### Bước 3: Giai Đoạn Chuẩn Bị (Preparation Phase)

Hệ thống chạy song song nhiều tác vụ tiền xử lý:

#### 3.1 Fast Track Processing
```python
# fastTrack.py - Xử lý nhanh cho các query đơn giản
class FastTrack:
    async def do(self):
        # Kiểm tra xem query có thể xử lý nhanh không
        # Nếu có, thực hiện tìm kiếm và ranking ngay lập tức
        # Điều này giúp giảm độ trễ cho các query phổ biến
```

#### 3.2 Query Analysis (analyze_query.py)
```python
# Phân tích loại câu hỏi và item type
class DetectItemType:
    async def do(self):
        # Xác định loại item được tìm kiếm (Recipe, Product, Event, etc.)
        # Sử dụng LLM để phân loại dựa trên nội dung query

class DetectQueryType:
    async def do(self):
        # Xác định loại query (search, comparison, recommendation, etc.)
        # Giúp hệ thống chọn strategy xử lý phù hợp
```

#### 3.3 Decontextualization (decontextualize.py)
```python
# Khử ngữ cảnh để tạo query độc lập
class PrevQueryDecontextualizer:
    async def do(self):
        # Sử dụng LLM để kết hợp query hiện tại với lịch sử hội thoại
        # Tạo ra một query hoàn chỉnh, độc lập với ngữ cảnh
        prompt = f"""
        Dựa trên lịch sử hội thoại: {self.handler.prev_queries}
        Query hiện tại: {self.handler.query}
        Hãy tạo một query hoàn chỉnh, độc lập.
        """
```

#### 3.4 Relevance Detection (relevance_detection.py)
```python
# Kiểm tra độ liên quan của query với site
class RelevanceDetection:
    async def do(self):
        # Sử dụng LLM để đánh giá xem query có phù hợp với site không
        # Ví dụ: "Cách nấu phở" phù hợp với food site
        #        "Giá Bitcoin" không phù hợp với recipe site
```

#### 3.5 Memory Management (memory.py)
```python
# Quản lý bộ nhớ hội thoại
class Memory:
    async def do(self):
        # Lưu trữ và truy xuất thông tin từ các hội thoại trước
        # Giúp duy trì ngữ cảnh qua nhiều lượt hội thoại
```

#### 3.6 Required Info Check (required_info.py)
```python
# Kiểm tra thông tin cần thiết
class RequiredInfo:
    async def do(self):
        # Xác định xem có đủ thông tin để trả lời query không
        # Nếu thiếu thông tin, yêu cầu người dùng cung cấp thêm
```

### Bước 4: Retrieval Phase (Tìm Kiếm Dữ Liệu)

#### 4.1 Vector Database Search
```python
# retriever.py
async def search(self, query, site):
    # Tạo embedding cho query
    embedding = await self.embedding_service.get_embedding(query)
    
    # Tìm kiếm trong vector database
    # Kết hợp semantic search với structured filters
    results = await self.vector_db.search(
        vector=embedding,
        filter={"site": site},
        limit=100
    )
    
    return results  # [(url, json_str, name, site), ...]
```

#### 4.2 Supported Vector Databases
- **Qdrant**: Vector database chuyên dụng
- **Azure AI Search**: Tích hợp với Azure ecosystem
- **Milvus**: Open-source vector database
- **Snowflake**: Data warehouse với vector search

### Bước 5: Ranking Phase (Xếp Hạng Kết Quả)

#### 5.1 Regular Ranking (ranking.py)
```python
class Ranking:
    async def rankItem(self, url, json_str, name, site):
        # Sử dụng LLM để đánh giá độ liên quan của từng item
        prompt_str, ans_struc = find_prompt(site, self.item_type, "RankingPrompt")
        
        # Điền thông tin vào prompt template
        prompt = fill_ranking_prompt(prompt_str, self.handler, json_str)
        
        # Gọi LLM để scoring
        ranking = await ask_llm(prompt, ans_struc, level="low")
        
        return {
            'url': url,
            'name': name,
            'ranking': ranking,
            'schema_object': json.loads(json_str)
        }
```

#### 5.2 Generate Answer Mode (generate_answer.py)
```python
class GenerateAnswer(NLWebHandler):
    async def synthesizeAnswer(self):
        # Sử dụng các item đã được rank để tạo câu trả lời tổng hợp
        response = await PromptRunner(self).run_prompt("SynthesizePrompt")
        
        # Tạo descriptions cho các items được reference
        for url in response["urls"]:
            description = await self.getDescription(url, ...)
            
        # Gửi kết quả cuối cùng
        message = {
            "message_type": "nlws",
            "answer": response["answer"],
            "items": json_results
        }
        await self.send_message(message)
```

### Bước 6: Response Generation & Streaming

#### 6.1 Streaming Response
```python
# StreamingWrapper.py
async def write_stream(self, message):
    # Gửi từng phần của response theo thời gian thực
    # Cho phép user thấy kết quả ngay khi có
    json_data = json.dumps(message)
    await self.response.write(f"data: {json_data}\n\n")
```

#### 6.2 Message Types
- **result_batch**: Danh sách các items tìm được
- **nlws**: Câu trả lời tổng hợp (generate mode)
- **api_version**: Thông tin version API
- **error**: Thông báo lỗi

## Prompt Management System

### Cấu Trúc Prompts (prompts.py)

```python
# Hệ thống prompt được tổ chức theo hierarchy
def find_prompt(site, item_type, prompt_name):
    # Tìm prompt phù hợp dựa trên:
    # 1. Site cụ thể (nếu có)
    # 2. Item type (Recipe, Product, Event, etc.)
    # 3. Prompt name (RankingPrompt, SynthesizePrompt, etc.)
    
    # Ví dụ: site="seriouseats.com", item_type="Recipe", prompt_name="RankingPrompt"
    # Sẽ tìm prompt chuyên biệt cho ranking recipes trên Serious Eats
```

### Prompt Variables
```python
# Các biến có thể sử dụng trong prompts:
{request.site}              # Site đang được query
{request.query}             # Query của user
{request.previousQueries}   # Lịch sử hội thoại
{request.contextUrl}        # URL ngữ cảnh
{request.itemType}          # Loại item đang tìm
{site.itemType}             # Loại item của site
{item.description}          # Mô tả item (cho ranking)
{request.answers}           # Kết quả đã tìm được
```

### Site Type Definitions (site_type.xml)
```xml
<!-- Định nghĩa prompts cho từng loại site và item type -->
<Site ref="seriouseats.com">
    <Recipe>
        <Prompt ref="RankingPrompt">
            <promptString>
                Đánh giá độ liên quan của công thức nấu ăn này với query: {request.query}
                Thông tin công thức: {item.description}
                Trả về điểm từ 0-100.
            </promptString>
            <returnStruc>{"score": 0, "reason": ""}</returnStruc>
        </Prompt>
    </Recipe>
</Site>
```

## LLM Integration

### Supported LLM Providers (config_llm.yaml)

```yaml
preferred_endpoint: azure_openai

endpoints:
  azure_openai:
    api_key_env: AZURE_OPENAI_API_KEY
    api_endpoint_env: AZURE_OPENAI_ENDPOINT
    llm_type: azure_openai
    models:
      high: gpt-4.1      # Cho các tác vụ phức tạp
      low: gpt-4.1-mini  # Cho các tác vụ đơn giản
      
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    llm_type: anthropic
    models:
      high: claude-3-5-sonnet-latest
      low: claude-3-5-haiku-latest
      
  # ... các providers khác
```

### LLM Usage Strategy
- **High-level models**: Dùng cho synthesis, complex reasoning
- **Low-level models**: Dùng cho ranking, simple classification
- **Parallel calls**: Nhiều LLM calls đồng thời để tăng tốc độ

## Vector Database & Embedding

### Embedding Services
```python
# embedding.py - Tạo vector representations cho text
class EmbeddingService:
    async def get_embedding(self, text):
        # Chuyển đổi text thành vector số
        # Sử dụng các models như:
        # - OpenAI text-embedding-ada-002
        # - Azure OpenAI embeddings
        # - Google Universal Sentence Encoder
        return vector_array
```

### Data Storage Format
```json
{
  "url": "https://example.com/recipe/pho",
  "name": "Authentic Vietnamese Pho",
  "site": "example.com",
  "schema_object": {
    "@type": "Recipe",
    "name": "Authentic Vietnamese Pho",
    "description": "Traditional Vietnamese noodle soup...",
    "ingredients": [...],
    "instructions": [...],
    "image": "https://example.com/pho.jpg"
  },
  "embedding": [0.1, -0.2, 0.3, ...],  // Vector representation
  "metadata": {
    "category": "soup",
    "cuisine": "vietnamese",
    "difficulty": "medium"
  }
}
```

## Configuration System

### Hierarchical Configuration
```python
# config.py - Quản lý cấu hình từ nhiều nguồn
class ConfigManager:
    def __init__(self):
        # Thứ tự ưu tiên:
        # 1. Environment variables
        # 2. YAML config files
        # 3. Default values
        
        self.llm_config = self.load_yaml("config_llm.yaml")
        self.embedding_config = self.load_yaml("config_embedding.yaml")
        self.retrieval_config = self.load_yaml("config_retrieval.yaml")
        # ...
```

### Environment Variables
```bash
# LLM Configuration
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
ANTHROPIC_API_KEY=your_anthropic_key

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key

# Embedding Services
OPENAI_API_KEY=your_openai_key
```

## MCP (Model Context Protocol) Integration

### MCP Server Functionality
```python
# mcp_handler.py - Xử lý MCP protocol
class MCPHandler:
    async def handle_ask_request(self, question, site=None):
        # NLWeb hoạt động như một MCP server
        # Cung cấp method "ask" cho AI agents
        
        # Tạo NLWebHandler để xử lý question
        handler = NLWebHandler({
            "query": question,
            "site": site or "all",
            "generate_mode": "generate"
        })
        
        # Trả về kết quả theo format schema.org
        result = await handler.runQuery()
        return result
```

### Schema.org Integration
```python
# Tất cả kết quả được format theo schema.org vocabulary
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "Vietnamese Pho",
  "description": "Traditional noodle soup...",
  "author": {
    "@type": "Person",
    "name": "Chef Nguyen"
  },
  "recipeIngredient": [
    "1 lb beef bones",
    "1 onion",
    "2 star anise"
  ],
  "recipeInstructions": [...]
}
```

## Performance Optimizations

### Fast Track Processing
```python
# fastTrack.py - Tối ưu hóa cho queries phổ biến
class FastTrack:
    async def do(self):
        # Kiểm tra nhanh xem query có cần xử lý phức tạp không
        if self.is_simple_query():
            # Bỏ qua decontextualization, relevance check
            # Thực hiện search và ranking ngay lập tức
            await self.quick_search_and_rank()
            self.handler.fastTrackWorked = True
```

### Parallel Processing
```python
# Nhiều tác vụ chạy song song
tasks = [
    asyncio.create_task(analyze_query.DetectItemType(self).do()),
    asyncio.create_task(self.decontextualizeQuery().do()),
    asyncio.create_task(relevance_detection.RelevanceDetection(self).do()),
    asyncio.create_task(memory.Memory(self).do()),
    asyncio.create_task(required_info.RequiredInfo(self).do())
]

await asyncio.gather(*tasks, return_exceptions=True)
```

### Caching Strategy
```python
# Cached prompts, embeddings, và LLM responses
cached_prompts = {}  # Cache prompts theo (site, item_type, prompt_name)
prompt_var_cache = {}  # Cache prompt variables
embedding_cache = {}  # Cache embeddings cho text đã xử lý
```

## Error Handling & Logging

### Comprehensive Logging
```python
# utils/logging_config_helper.py
logger = get_configured_logger("nlweb_handler")

# Log levels và categories
logger.info("Query execution started")      # Thông tin chung
logger.debug("Retrieved 50 items")          # Chi tiết debug
logger.warning("No results found")          # Cảnh báo
logger.error("LLM API call failed")         # Lỗi
logger.exception("Unexpected error")        # Lỗi với stack trace
```

### Graceful Degradation
```python
# Xử lý lỗi một cách graceful
try:
    result = await llm_call(prompt)
except LLMTimeoutError:
    # Fallback to simpler processing
    result = await simple_keyword_search(query)
except LLMAPIError:
    # Return cached results if available
    result = get_cached_results(query)
```

## Deployment & Scaling

### Azure Web App Deployment
```python
# app-file.py - Entry point cho Azure
def main():
    load_dotenv()  # Load environment variables
    port = int(os.environ.get('PORT', 8000))
    
    # Start server với Azure-compatible settings
    asyncio.run(start_server(
        host='0.0.0.0',
        port=port,
        fulfill_request=fulfill_request
    ))
```

### Horizontal Scaling Considerations
- **Stateless Design**: Mỗi request độc lập, có thể xử lý trên bất kỳ instance nào
- **External Dependencies**: Vector DB, LLM APIs có thể scale riêng biệt
- **Caching**: Shared cache (Redis) cho multi-instance deployment
- **Load Balancing**: HTTP load balancer phân phối requests

## Data Pipeline

### Data Ingestion (tools/)
```python
# db_load.py - Load dữ liệu vào vector database
class DataLoader:
    async def load_schema_org_data(self, json_file):
        # Đọc dữ liệu từ JSON/JSONL files
        # Tạo embeddings cho từng item
        # Lưu vào vector database với metadata
        
    async def load_rss_data(self, rss_url):
        # Parse RSS feeds
        # Convert sang schema.org format
        # Tạo embeddings và lưu trữ
```

### Data Formats Supported
- **Schema.org JSON-LD**: Format chính được khuyến khích
- **RSS/Atom feeds**: Tự động convert sang schema.org
- **Custom JSON**: Với mapping rules
- **Structured markup**: Từ HTML pages

## Security & Privacy

### API Security
```python
# Validation và sanitization
def validate_query_params(params):
    # Kiểm tra input parameters
    # Prevent injection attacks
    # Rate limiting
    pass

# Environment-based configuration
api_key = os.environ.get('AZURE_OPENAI_API_KEY')
if not api_key:
    raise ConfigurationError("Missing API key")
```

### Data Privacy
- **No Personal Data Storage**: Chỉ lưu trữ public content
- **Query Logging**: Có thể disable cho privacy
- **Encryption**: HTTPS cho tất cả communications
- **Access Control**: API keys và authentication

## Monitoring & Analytics

### Performance Metrics
```python
# Tracking key metrics
metrics = {
    "query_processing_time": time.time() - start_time,
    "llm_calls_count": self.llm_call_counter,
    "items_retrieved": len(self.final_retrieved_items),
    "items_ranked": len(self.final_ranked_answers),
    "fast_track_success": self.fastTrackWorked
}
```

### Health Checks
```python
# Kiểm tra sức khỏe của các components
async def health_check():
    checks = {
        "vector_db": await check_vector_db_connection(),
        "llm_api": await check_llm_api_availability(),
        "embedding_service": await check_embedding_service()
    }
    return checks
```

## Kết Luận

NLWeb là một hệ thống phức tạp nhưng được thiết kế modular, cho phép:

1. **Dễ dàng tích hợp**: Với các LLM và vector databases khác nhau
2. **Linh hoạt**: Có thể tùy chỉnh prompts, control flow theo nhu cầu
3. **Scalable**: Từ laptop đến production clusters
4. **Chuẩn hóa**: Sử dụng các protocols và formats được công nhận (MCP, Schema.org)
5. **Không hallucination**: Kết quả luôn dựa trên dữ liệu thực tế

Hệ thống này đặc biệt phù hợp cho các websites có dữ liệu structured (e-commerce, recipes, events, reviews) muốn cung cấp giao diện trò chuyện tự nhiên cho người dùng mà không cần thay đổi cấu trúc dữ liệu hiện tại.
