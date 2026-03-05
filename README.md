# RAG Web System (LangChain 1.0+)

一个基于 **FastAPI + Gradio + LangChain 1.0+** 的检索增强生成（RAG）应用，包含：

- 检索模块：文档加载、分块、向量化、相似检索
- 生成模块：基于检索上下文调用 LLM API 回答
- 评估模块：基于 RAGAS 的自动化评估
- Web 模块：可视化调参、问答交互、评估报告展示

## 1. 快速开始

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

访问：
- API 文档：`http://127.0.0.1:8000/docs`
- Gradio UI：`http://127.0.0.1:8000/ui`

## 2. 目录结构

```text
app/
  core/         # 配置、日志、异常
  rag/          # 检索与生成核心逻辑
  eval/         # RAGAS 评估
  web/          # Gradio 界面
  schemas/      # 接口数据模型
tests/          # 单元与集成测试
```

## 3. 配置说明

核心环境变量位于 `.env.example`：
- Embedding 本地模型：`EMBEDDING_MODEL_NAME`、`EMBEDDING_MODEL_DEVICE`
- 向量库持久化路径：`VECTOR_STORE_DIR`
- LLM API：`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`
- 检索/生成默认参数：`SEARCH_TOP_K`、`DEFAULT_TEMPERATURE` 等

## 4. 部署

```bash
docker compose up --build
```

默认以 Gunicorn + Uvicorn workers 运行，支持多用户并发访问。

