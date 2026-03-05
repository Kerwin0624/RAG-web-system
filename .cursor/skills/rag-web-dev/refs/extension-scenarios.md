# 常见扩展场景 Step-by-Step

## 场景一：添加新检索策略

以添加"语义+BM25+MMR 三路融合"策略为例。

### Step 1: 在 `app/rag/retriever.py` 添加检索方法

```python
def _triple_search(
    self,
    query: str,
    k: int,
    vector_weight: float,
    bm25_weight: float,
) -> list[tuple[Document, float | None]]:
    """向量 + BM25 + MMR 三路融合检索。"""
    vector_results = self._vector_search(query, k)
    bm25_results = self._bm25_search(query, k)
    mmr_results = self._mmr_search(query, k)

    vector_scores = _normalize_scores(vector_results)
    bm25_scores = _normalize_scores(bm25_results)

    combined: dict[str, tuple[Document, float]] = {}

    # 向量 + BM25 加权
    for doc, norm_score in vector_scores:
        key = _doc_key(doc)
        combined[key] = (doc, vector_weight * norm_score)
    for doc, norm_score in bm25_scores:
        key = _doc_key(doc)
        if key in combined:
            existing_doc, existing_score = combined[key]
            combined[key] = (existing_doc, existing_score + bm25_weight * norm_score)
        else:
            combined[key] = (doc, bm25_weight * norm_score)

    # MMR 结果加成（在已有结果基础上增加多样性加分）
    mmr_bonus = 0.1
    for doc, _ in mmr_results:
        key = _doc_key(doc)
        if key in combined:
            existing_doc, existing_score = combined[key]
            combined[key] = (existing_doc, existing_score + mmr_bonus)
        else:
            combined[key] = (doc, mmr_bonus)

    ranked = sorted(combined.values(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
```

### Step 2: 在 `retrieve()` 方法中注册

在 `app/rag/retriever.py` 的 `retrieve()` 条件分支中添加：

```python
elif search_type == "triple":
    results = self._triple_search(query, fetch_k, vector_weight, bm25_weight)
```

### Step 3: 更新 Schema

在 `app/schemas/rag.py` 中更新 `QueryRequest`：

```python
search_type: Literal["similarity", "bm25", "hybrid", "mmr", "triple"] = "similarity"
```

### Step 4: 更新 Gradio UI

在 `app/web/gradio_ui.py` 的 `SEARCH_TYPE_CHOICES` 中添加：

```python
SEARCH_TYPE_CHOICES = [
    ("向量相似度", "similarity"),
    ("关键词检索（BM25）", "bm25"),
    ("混合检索", "hybrid"),
    ("MMR 多样性检索", "mmr"),
    ("三路融合", "triple"),       # ← 新增
]
```

### Step 5: 添加测试

在 `tests/test_retrieval.py` 中添加：

```python
def test_triple_search(tmp_path):
    # 使用 DummyManager + BM25Store 构造测试环境
    settings = Settings(LLM_BASE_URL="https://example.com/v1", LLM_API_KEY="x", LLM_MODEL="mock")
    manager = DummyManager()
    bm25 = BM25Store(corpus_path=tmp_path / "bm25.pkl")
    ingest = IngestService(settings=settings, vector_store_manager=manager, bm25_store=bm25)

    test_file = tmp_path / "doc.txt"
    test_file.write_text("Python 是一种编程语言。", encoding="utf-8")
    ingest.ingest(paths=[str(test_file)])

    retriever = RetrieverService(settings=settings, vector_store_manager=manager, bm25_store=bm25)
    hits = retriever.retrieve("Python", top_k=2, search_type="triple")
    assert len(hits) >= 1
```

---

## 场景二：接入新 LLM

以接入 DeepSeek API 为例。

### Step 1: 确认 API 兼容性

DeepSeek API 兼容 OpenAI 格式，无需修改代码，只需配置环境变量。

### Step 2: 更新 `.env`

```
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-deepseek-key
LLM_MODEL=deepseek-chat
```

### Step 3: 在 UI 中添加预设（可选）

在 `app/web/gradio_ui.py` 的 `MODEL_CHOICES` 中添加：

```python
MODEL_CHOICES = [
    ("Kimi K2.5", "kimi-k2.5"),
    ("MiniMax M2.5", "MiniMax/MiniMax-M2.5"),
    ("GLM-5", "glm-5"),
    ("Qwen 3.5 Plus", "qwen3.5-plus"),
    ("Qwen 3.5 Flash", "qwen3.5-flash"),
    ("DeepSeek Chat", "deepseek-chat"),  # ← 新增
    ("自定义模型", "__custom__"),
]
```

### 如果 API 不兼容 OpenAI 格式

需要在 `app/rag/generator.py` 中扩展：

```python
from langchain_community.chat_models import ChatXXX  # 对应 LangChain 集成

class LLMGenerator:
    def _get_llm(self, model_name: str):
        if model_name.startswith("xxx-"):
            return ChatXXX(model=model_name, ...)
        return ChatOpenAI(
            model=model_name,
            base_url=self._settings.llm_base_url,
            ...
        )
```

---

## 场景三：添加新文档格式

以支持 `.docx` 格式为例。

### Step 1: 安装依赖

```bash
pip install python-docx
```

并在 `pyproject.toml` 中添加可选依赖：

```toml
[project.optional-dependencies]
docx = ["python-docx>=1.1.0"]
```

### Step 2: 更新 `app/rag/ingest.py`

```python
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

def _load_one(self, path: Path, pdf_loader: str = "pypdf") -> list[Document]:
    # ... 已有逻辑 ...

    if suffix == ".docx":
        return self._load_docx(path)

    return TextLoader(str(path), encoding="utf-8").load()

def _load_docx(self, path: Path) -> list[Document]:
    from docx import Document as DocxDocument
    doc = DocxDocument(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [LCDocument(page_content=text, metadata={"source": str(path)})]
```

### Step 3: 更新 Gradio UI

在 `app/web/gradio_ui.py` 的文件上传组件中：

```python
upload_files = gr.File(
    file_types=[".txt", ".md", ".pdf", ".docx"],
    ...
)
```

### Step 4: 添加测试

```python
def test_docx_ingest(tmp_path):
    from docx import Document as DocxDocument
    doc = DocxDocument()
    doc.add_paragraph("测试文档内容")
    docx_path = tmp_path / "test.docx"
    doc.save(str(docx_path))

    # ... 构建 IngestService 并测试 ...
```

---

## 场景四：添加新评估指标

以添加 RAGAS 的 `answer_correctness` 指标为例。

### Step 1: 更新 `app/eval/ragas_eval.py`

```python
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,   # ← 新增
    context_precision,
    context_recall,
    faithfulness,
)

result = evaluate(
    dataset=dataset,
    metrics=[answer_relevancy, answer_correctness, context_precision, context_recall, faithfulness],
    **extra,
)
```

### Step 2: 更新可视化

在 `app/web/gradio_ui.py` 的 `_build_ragas_chart()` 中：

```python
label_map = {
    "context_precision": "上下文精度",
    "context_recall": "上下文召回率",
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "answer_correctness": "答案正确性",  # ← 新增
    "retrieval_f1": "检索F1",
}
display_keys = [
    "context_precision", "context_recall", "faithfulness",
    "answer_relevancy", "answer_correctness", "retrieval_f1",
]
```

---

## 场景五：切换 Embedding 模型

### 使用其他 Ollama 模型

```bash
ollama pull nomic-embed-text
```

更新 `.env`：

```
EMBEDDING_MODEL_NAME=nomic-embed-text
```

> 注意：切换模型后需要重新入库所有文档，旧向量库与新模型不兼容。

### 清空向量库

```bash
# Windows
rmdir /s /q vector_store

# macOS / Linux
rm -rf vector_store/
```

重启服务后会自动创建空向量库。

---

## 场景六：自定义 Prompt 模板

修改 `app/rag/generator.py` 中的 `SYSTEM_PROMPT`：

```python
SYSTEM_PROMPT = """你是一个专业的技术文档助手。
请根据上下文严格回答问题：
1. 使用准确的技术术语。
2. 如果涉及代码，请用 markdown 代码块格式化。
3. 如果信息不足，说明缺少哪些信息。
4. 回答使用中文。
"""
```

修改 Human 消息模板（同文件中 `ChatPromptTemplate` 的 `human` 部分）：

```python
("human", "## 问题\n{question}\n\n## 参考资料\n{context}\n\n请给出准确回答。"),
```
