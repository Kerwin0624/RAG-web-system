# RAG Web System

一套面向 **RAG 应用开发者**的完整开发框架，基于 **FastAPI + Gradio + LangChain 1.0+** 构建。

目标是让开发者在面对不同子项目时，能**快速进行针对性调优**，大幅提升迭代效率。

## 核心功能

- **完整 RAG 流程**：文档上传、检索、问答一站式完成
- **自动化评估**：基于测试集 + RAGAS 的系统化评估，支持逐题分析与可视化
- **全链路可调参**：所有关键模块均支持参数调节，调参结果可保存，方便持续迭代
- **多模型支持**：内置 5 种 LLM 可供切换，运行时动态选择
- **主流 RAG 优化方法覆盖**：向量检索、BM25 关键词检索、混合检索、MMR 多样性检索、Reranker 重排序等（暂未加入父子索引等高级 RAG 技巧）
- **评估结果可视化**：最终评估结果会以可视化图表输出，方便直观查看效果、明确后续优化方向。

## 快速开始

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

## 目录结构

```text
app/
  core/         # 配置、日志、异常
  rag/          # 检索与生成核心逻辑
  eval/         # RAGAS 评估
  web/          # Gradio 界面
  schemas/      # 接口数据模型
tests/          # 单元与集成测试
```

## 配置说明

核心环境变量位于 `.env.example`：
- Embedding 本地模型：`EMBEDDING_MODEL_NAME`、`EMBEDDING_MODEL_DEVICE`
- 向量库持久化路径：`VECTOR_STORE_DIR`
- LLM API：`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`
- 检索/生成默认参数：`SEARCH_TOP_K`、`DEFAULT_TEMPERATURE` 等

## 部署

```bash
docker compose up --build
```

默认以 Gunicorn + Uvicorn workers 运行，支持多用户并发访问。

