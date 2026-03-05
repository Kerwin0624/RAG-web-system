# 架构详解

## 整体分层

```
┌──────────────────────────────────────────────────────────────┐
│                 FastAPI  (app/main.py)                        │
│  GET /         → 重定向 /ui/                                  │
│  GET /health   → 健康检查                                     │
│  POST /ingest  → 文档入库                                     │
│  POST /query   → RAG 问答                                     │
│  POST /evaluate→ RAGAS 评估                                   │
│  GET /ui/      → Gradio 界面 (mount_gradio_app)              │
│                                                              │
│  中间件: timing_middleware → X-Process-Time-MS 响应头          │
│  全局异常: global_exception_handler → 500                     │
└──────────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────────┐
│               RAG Pipeline  (app/rag/pipeline.py)            │
│                                                              │
│  ask() → 缓存检查 → 检索 → 生成 → 组装 citations → 写缓存    │
│                                                              │
│  缓存: OrderedDict LRU, 256 条, MD5 键                       │
└──────────────────────────────────────────────────────────────┘
         │                            │
┌────────┴──────────┐    ┌────────────┴────────────┐
│  RetrieverService  │    │     LLMGenerator        │
│  (retriever.py)    │    │     (generator.py)      │
│                    │    │                         │
│  4 种检索策略:      │    │  ChatPromptTemplate     │
│  - similarity      │    │  + ChatOpenAI           │
│  - bm25            │    │  + StrOutputParser      │
│  - hybrid          │    │                         │
│  - mmr             │    │  多模型: lru_cache(16)   │
│                    │    │  bind() 运行时参数       │
│  可选 Reranker     │    └─────────────────────────┘
└──────┬────────────┘
       │
┌──────┴───────────────────────────────────────────┐
│                   存储层                          │
│                                                  │
│  VectorStoreManager       BM25Store              │
│  (vector_store.py)        (bm25_store.py)        │
│                                                  │
│  Chroma + OllamaEmbed    jieba + BM25Okapi       │
│  持久化: vector_store/    持久化: bm25_corpus.pkl  │
└──────────────────────────────────────────────────┘
```

## 数据流

### 入库流程

```
用户上传文件/指定路径
    ↓
IngestService.ingest()
    ├── _load_one()        → PyPDFLoader / TextLoader / MinerU
    ├── _build_splitter()  → RecursiveCharacterTextSplitter
    ├── _split_docs()      → 添加 chunk_id 元数据
    ├── VectorStoreManager.add_documents()  → Chroma
    └── BM25Store.add_documents()           → jieba 分词 → pickle
```

### 查询流程

```
用户提问 + 参数
    ↓
RAGPipeline.ask()
    ├── 缓存命中? → 直接返回 (from_cache=true)
    ├── RetrieverService.retrieve()
    │   ├── similarity → Chroma.similarity_search_with_relevance_scores()
    │   ├── bm25      → BM25Store.search() (jieba 分词)
    │   ├── hybrid    → 向量+BM25 → min-max 归一化 → 加权融合
    │   ├── mmr       → Chroma.max_marginal_relevance_search()
    │   ├── score_threshold 过滤 (mmr 除外)
    │   └── reranker? → CrossEncoder 重排序
    ├── LLMGenerator.generate()
    │   ├── 选择 LLM (默认 / 指定模型, lru_cache)
    │   ├── bind(temperature, max_tokens, top_p)
    │   └── prompt | llm | StrOutputParser → invoke()
    └── 组装 citations + 写缓存
```

### 评估流程

```
测试集 JSON → TestsetRunner.run()
    ├── 逐题调用 RAGPipeline.ask()
    ├── 收集 model_answer + contexts
    ├── (可选) RagasEvaluator.evaluate_samples()
    │   ├── EvalDatasetBuilder.build() → HuggingFace Dataset
    │   └── ragas.evaluate() → 4 指标 + retrieval_f1
    └── 返回逐题结果 + RAGAS 指标
```

## 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **配置中心** | `app/core/config.py` | Pydantic Settings，`.env` 加载，`lru_cache` 单例 |
| **异常层级** | `app/core/errors.py` | `RAGError` → `IngestError` / `RetrievalError` / `GenerationError` / `EvaluationError` |
| **日志** | `app/core/logging.py` | `basicConfig`，格式: `时间 | 级别 | 模块 | 消息` |
| **入库** | `app/rag/ingest.py` | 文件加载 → 分块 → 双存储写入 |
| **向量存储** | `app/rag/vector_store.py` | Chroma 封装，OllamaEmbeddings |
| **BM25 存储** | `app/rag/bm25_store.py` | jieba 分词，BM25Okapi 索引，pickle 持久化 |
| **检索** | `app/rag/retriever.py` | 4 策略 + 阈值过滤 + 可选重排序 |
| **重排序** | `app/rag/reranker.py` | CrossEncoder 懒加载，`sentence-transformers` 可选依赖 |
| **生成** | `app/rag/generator.py` | LangChain LCEL 链，多模型 `lru_cache` |
| **管线** | `app/rag/pipeline.py` | 检索→生成 编排，LRU 结果缓存 |
| **RAGAS** | `app/eval/ragas_eval.py` | 评估 4 指标 + retrieval_f1 |
| **测试运行** | `app/eval/testset_runner.py` | 批量测试 + 逐题收集 |
| **数据集** | `app/eval/dataset_builder.py` | EvalSample → HuggingFace Dataset |
| **报告** | `app/eval/reporting.py` | JSON + CSV 导出 |
| **Schema** | `app/schemas/rag.py` | 请求/响应 Pydantic 模型 |
| **UI** | `app/web/gradio_ui.py` | 4 Tab Gradio 界面 |
| **入口** | `app/main.py` | 服务组装 + API 路由 + Gradio 挂载 |

## 依赖注入模式

`app/main.py` 作为组合根，手动创建所有服务实例并注入依赖：

```python
# 创建顺序
settings → vector_store_manager → bm25_store → reranker_service
        → ingest_service(settings, vsm, bm25)
        → retriever_service(settings, vsm, bm25, reranker)
        → generator_service(settings)
        → pipeline(retriever, generator)
        → eval_llm / eval_embeddings → evaluator → testset_runner
```

这种手动注入方式在小型项目中保持了简洁和可测试性，每个服务均可通过构造函数注入 mock 对象进行单元测试。

## Prompt 设计

系统 Prompt 位于 `app/rag/generator.py`：

```
你是一个严谨的RAG问答助手。
请根据提供的上下文回答问题：
1. 仅使用上下文中的事实，不要臆造。
2. 如果上下文不足，请明确说明"根据现有资料无法确定"。
3. 回答尽量简洁，并优先使用中文。
```

Human 消息模板: `问题：{question}\n\n上下文：\n{context}\n\n请输出最终答案。`

设计理由：
- 明确约束"仅使用上下文"减少幻觉
- "无法确定"兜底防止强行回答
- 中文优先匹配目标用户群
