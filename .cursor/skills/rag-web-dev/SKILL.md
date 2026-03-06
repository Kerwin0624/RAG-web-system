# RAG Web System 开发指南 Skill

**版本：0.2**（与项目 v0.4 迭代经验同步）

> 本 Skill 为 **RAG Web System** 项目的完整开发指南。  
> 面向新成员快速上手、日常开发与架构扩展。所有路径使用正斜杠 `/`。

---

## 1. 项目概览

| 属性 | 值 |
|------|---|
| **定位** | 面向 RAG 应用开发者的可调参全链路开发框架 |
| **技术栈** | Python 3.10+ · FastAPI · Gradio · LangChain 1.0+ |
| **向量库** | ChromaDB（本地持久化） |
| **Embedding** | Ollama + bge-m3（本地部署，免费、无网络延迟） |
| **LLM** | OpenAI 兼容 API（支持 Kimi / MiniMax / GLM / Qwen 等） |
| **评估** | RAGAS + 自定义测试集 |
| **部署** | Uvicorn (dev) / Gunicorn + Docker (prod) |

### 1.1 架构决策摘要

| 决策 | 理由 |
|------|------|
| Ollama 本地 Embedding | 免费、低延迟、数据不出本机，中文支持好（bge-m3） |
| Chroma 而非 Milvus/FAISS | 零运维、纯 Python、内置持久化，适合中小规模 |
| OpenAI 兼容 API | 一套代码适配所有主流大模型供应商 |
| Gradio 而非 React | 纯 Python 快速构建可调参 UI，无需前端工程 |
| BM25 + 向量 双存储 | 混合检索兼顾语义理解与关键词精确匹配 |
| LRU Pipeline 缓存 | 相同参数问题直接返回，减少重复 LLM 调用开销 |
| Pydantic Settings | 类型安全的环境变量管理，支持 `.env` 文件 |
| 自定义异常层级 | 统一错误处理，API 层根据异常类型返回合适 HTTP 状态码 |

> 详细架构图和模块职责请参考 → [`refs/architecture.md`](refs/architecture.md)

---

## 2. 环境搭建（跨平台）

### 2.1 前置依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 推荐 3.11 |
| Ollama | 最新 | 运行 Embedding 模型 |
| Git | 2.x | 版本管理 |

### 2.2 安装步骤

**1) 克隆仓库**

```bash
git clone <repo-url> rag-web-system
cd rag-web-system
```

**2) 创建虚拟环境**

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows CMD
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3) 安装依赖**

```bash
# 基础安装
pip install -e .

# 启用重排序（可选）
pip install -e ".[reranker]"

# 启用 MinerU PDF 解析（可选）
pip install -e ".[mineru]"

# 开发依赖
pip install -e ".[dev]"
```

**4) 配置环境变量**

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

编辑 `.env`，至少修改以下字段：

```
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=your-api-key
LLM_MODEL=qwen3.5-plus
```

**5) 启动 Ollama 并拉取 Embedding 模型**

```bash
ollama pull bge-m3
ollama serve   # 若未作为服务运行
```

**6) 环境检测**

```bash
python scripts/check_env.py
```

**7) 启动服务**

```bash
# 开发模式（macOS / Linux）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 开发模式（Windows，若遇 WinError 10013 权限错误则改用 127.0.0.1）
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 生产模式（macOS / Linux）
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 app.main:app

# Docker
docker compose up --build
```

访问地址：
- Gradio UI: `http://127.0.0.1:8000/ui`
- API 文档: `http://127.0.0.1:8000/docs`
- 健康检查: `http://127.0.0.1:8000/health`

> 详细安装排错请参考 → [`refs/troubleshooting.md`](refs/troubleshooting.md)

---

## 3. 项目结构

```
rag-web-system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口，服务组装
│   ├── core/
│   │   ├── config.py           # Pydantic Settings 配置中心
│   │   ├── errors.py           # 自定义异常层级
│   │   └── logging.py          # 日志初始化
│   ├── rag/
│   │   ├── pipeline.py         # RAG 主管线（检索→生成→缓存）
│   │   ├── ingest.py           # 文档加载、分块、入库
│   │   ├── retriever.py        # 4 种检索策略实现
│   │   ├── generator.py        # LLM 生成（多模型支持 + 仅答知识库）
│   │   ├── vector_store.py     # Chroma 向量存储
│   │   ├── bm25_store.py       # BM25 关键词索引（jieba 分词）
│   │   └── reranker.py         # CrossEncoder 重排序（懒加载）
│   ├── eval/
│   │   ├── ragas_eval.py       # RAGAS 评估器 + 优化建议生成
│   │   ├── testset_runner.py   # 测试集批量运行（含指标目标传递）
│   │   ├── dataset_builder.py  # HuggingFace Dataset 构建
│   │   └── reporting.py        # 评估报告导出（JSON/CSV）
│   ├── schemas/
│   │   └── rag.py              # Pydantic 请求/响应模型
│   └── web/
│       └── gradio_ui.py        # Gradio 4-Tab 界面（含成本/效率提示）
├── docs/
│   ├── preview/                # 功能/界面预览截图（6 张）
│   └── release-notes-v0.4.md   # v0.4 版本发布说明
├── tests/
│   ├── test_retrieval.py       # 入库+检索集成测试
│   ├── test_generation.py      # Pipeline 缓存测试
│   └── test_eval.py            # 评估模块测试
├── scripts/
│   └── check_env.py            # 环境检测脚本
├── data/                       # 文档存放目录
├── vector_store/               # 向量库持久化目录
├── README.md                   # 项目说明（含功能/界面预览）
├── .gitignore                  # Git 忽略规则
├── pyproject.toml              # 项目元数据与依赖
├── Dockerfile                  # Docker 镜像
├── docker-compose.yml          # 编排配置
├── .env.example                # 环境变量模板
└── .env                        # 实际环境变量（不入库）
```

---

## 4. 核心配置参数

所有参数通过 `app/core/config.py` 中 `Settings` 类管理，支持 `.env` 文件和环境变量。

#### 应用与基础设施

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `APP_NAME` | RAG Web System | 应用名称（显示在 FastAPI docs） |
| `APP_ENV` | dev | 运行环境标识 |
| `HOST` | 0.0.0.0 | 监听地址（Windows 若遇权限错误可改 127.0.0.1） |
| `PORT` | 8000 | 监听端口 |
| `LOG_LEVEL` | INFO | 日志级别 |
| `DATA_DIR` | ./data | 文档存放目录 |
| `VECTOR_STORE_DIR` | ./vector_store | 向量库持久化目录 |
| `COLLECTION_NAME` | default_collection | ChromaDB 集合名 |

#### Embedding 与 LLM

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama 服务地址 |
| `EMBEDDING_MODEL_NAME` | bge-m3 | Ollama 本地 Embedding 模型 |
| `LLM_BASE_URL` | （必填） | OpenAI 兼容 API 地址 |
| `LLM_API_KEY` | （必填） | API 密钥 |
| `LLM_MODEL` | （必填） | 默认使用的 LLM 模型名 |
| `LLM_TIMEOUT` | 60 | LLM 请求超时（秒） |
| `LLM_MAX_RETRIES` | 3 | LLM 请求重试次数 |

#### RAG 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PDF_LOADER` | pypdf | PDF 解析引擎（pypdf / mineru） |
| `CHUNK_SIZE` | 800 | 分块最大字符数，中文场景 500–1000 较优 |
| `CHUNK_OVERLAP` | 120 | 相邻块重叠字符数，约 chunk_size 的 15% |
| `SEARCH_TOP_K` | 4 | 默认检索 Top-K |
| `SEARCH_SCORE_THRESHOLD` | 0.25 | 相似度过滤阈值 |
| `VECTOR_WEIGHT` | 0.7 | 混合检索中向量权重 |
| `BM25_WEIGHT` | 0.3 | 混合检索中 BM25 权重 |
| `DEFAULT_TEMPERATURE` | 0.2 | 低温度确保回答稳定 |
| `DEFAULT_MAX_TOKENS` | 512 | 生成最大 token |
| `DEFAULT_TOP_P` | 1.0 | 核采样阈值 |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | CrossEncoder 重排序模型 |
| `EVAL_BATCH_SIZE` | 8 | RAGAS 评估批大小 |

### 参数设计理由

- **CHUNK_SIZE=800**：中文语境下约 400 个汉字，既包含足够语义又避免上下文过长
- **CHUNK_OVERLAP=120**：15% 重叠保证句子不在边界断裂
- **VECTOR_WEIGHT=0.7**：语义检索为主，关键词为辅，经验值
- **TEMPERATURE=0.2**：RAG 场景需要忠实于上下文，低温度减少幻觉

---

## 5. 检索策略

| 策略 | `search_type` | 适用场景 |
|------|---------------|---------|
| 向量相似度 | `similarity` | 通用语义检索，默认选择 |
| BM25 关键词 | `bm25` | 精确术语、代码片段检索 |
| 混合检索 | `hybrid` | 兼顾语义+关键词，推荐复杂场景 |
| MMR 多样性 | `mmr` | 需要多角度覆盖的场景 |

可选 **Reranker 重排序**（`reranker_enabled=true`）：先多取候选（`fetch_k=max(k*3, 20)`），再用 CrossEncoder 精排。

---

## 6. API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/` | 自动重定向到 `/ui/` |
| `GET` | `/health` | 健康检查 |
| `POST` | `/ingest` | 文档入库 |
| `POST` | `/query` | RAG 问答 |
| `POST` | `/evaluate` | RAGAS 评估 |
| `GET` | `/ui/` | Gradio 界面 |
| `GET` | `/docs` | OpenAPI 文档 |

---

## 7. 测试

```bash
# 运行所有测试
pytest

# 带覆盖率
pytest --cov=app

# 仅运行检索测试
pytest tests/test_retrieval.py -v
```

测试使用 `DummyVectorStore` / `DummyManager` 模拟向量存储，无需启动 Ollama。

---

## 8. 常见扩展场景

以下为最常见的扩展操作，每个场景提供完整 step-by-step 示例。

> 详细代码示例请参考 → [`refs/extension-scenarios.md`](refs/extension-scenarios.md)

### 8.1 添加新检索策略

1. 在 `app/rag/retriever.py` 的 `RetrieverService` 中添加私有方法
2. 在 `retrieve()` 的条件分支中注册新策略
3. 在 `app/schemas/rag.py` 的 `QueryRequest.search_type` 的 `Literal` 中添加新值
4. 在 `app/web/gradio_ui.py` 的 `SEARCH_TYPE_CHOICES` 中添加 UI 选项
5. 添加对应单元测试

### 8.2 接入新 LLM

1. 在 `.env` 中配置新的 `LLM_BASE_URL` / `LLM_API_KEY`
2. 在 `app/web/gradio_ui.py` 的 `MODEL_CHOICES` 中添加新模型
3. 若非 OpenAI 兼容 API，需在 `app/rag/generator.py` 中适配

### 8.3 添加新文档格式

1. 在 `app/rag/ingest.py` 的 `SUPPORTED_EXTENSIONS` 中添加扩展名
2. 在 `_load_one()` 方法中添加加载逻辑
3. 在 Gradio UI 的 `file_types` 中添加格式
4. 添加对应测试

### 8.4 添加新评估指标

1. 在 `app/eval/ragas_eval.py` 的 `evaluate_samples()` 中导入新指标
2. 将指标添加到 `metrics` 列表
3. 在 `app/web/gradio_ui.py` 的 `_build_ragas_chart()` 中更新可视化
4. 若需纳入「优化建议」，在 `get_optimization_suggestions()` 中增加该指标的阈值判断与建议文案
5. 若需在前端暴露目标，在标准答案评测 UI 中增加 `gr.Number` 目标输入，并传入 `_run_testset`

### 8.5 新增前端参数时

1. 在对应 `gr.Slider` / `gr.Dropdown` / `gr.Checkbox` 上增加 `info="成本：... 效率：..."` 浮窗说明
2. 在标准答案评测 Tab 的同名控件上同步补充 `info`

---

## 9. Docker 部署

```bash
# 构建并启动
docker compose up --build -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

注意事项：
- 容器内 Ollama 不可用，需外部 Ollama 服务，将 `OLLAMA_BASE_URL` 指向宿主机
- `data/` 和 `vector_store/` 通过 volume 挂载持久化
- 默认 2 个 Gunicorn worker

---

## 10. 环境检测

运行环境检测脚本，一键排查所有依赖和服务状态：

```bash
python scripts/check_env.py
```

脚本检查项：Python 版本、核心包、Ollama 连接、Embedding 模型、LLM 配置、目录权限。

---

## 11. 报错排查

> 完整排查清单请参考 → [`refs/troubleshooting.md`](refs/troubleshooting.md)

---

## 12. 项目经验总结（v0.4 迭代）

以下为项目迭代至 v0.4 时沉淀的经验，便于后续维护与类似项目参考。

### 12.1 产品与运营侧

| 项 | 做法 | 说明 |
|----|------|------|
| **成本/效率透明** | 为 Gradio 各参数（分块、检索、生成、模型等）增加 `info` 浮窗 | 鼠标悬停即可看到该参数对成本与效率的影响，便于产品/运营决策 |
| **指标目标可配置** | 标准答案评测中增加 Faithfulness / Answer Relevancy / Context Recall / Context Precision 四个目标（默认 0.85 / 0.8 / 0.8 / 0.7） | 用户可自行修改；RAGAS 打分低于目标时再给出优化方向建议，避免噪音 |
| **优化建议与目标绑定** | `get_optimization_suggestions(summary, target_*=...)` 按目标阈值生成建议 | 建议文案中带「当前 x.xx < 目标 x.xx」，便于验收与迭代 |

### 12.2 安全与成本控制

| 项 | 做法 | 说明 |
|----|------|------|
| **仅答知识库** | 系统 prompt 明确「仅依据上下文作答」；无上下文时直接返回固定话术、不调用 LLM | 避免通用问答/闲聊消耗 API，防止滥用 |
| **无上下文不调用 LLM** | `generator.generate()` 在 `contexts` 为空时返回 `NO_CONTEXT_REPLY`，不发起请求 | 节省费用并统一拒绝话术 |

### 12.3 工程与部署

| 项 | 做法 | 说明 |
|----|------|------|
| **setuptools 包发现** | 在 `pyproject.toml` 中增加 `[tool.setuptools.packages.find]` 且 `include = ["app*"]` | 项目根目录存在 `data/`、`vector_store/` 等非包目录时，避免被当成多顶层包导致 `pip install -e .` 失败 |
| **Windows 启动 host** | 开发时使用 `--host 127.0.0.1` 而非 `0.0.0.0`（若遇 WinError 10013） | 部分 Windows 环境下绑定 0.0.0.0 会触发权限错误，127.0.0.1 可正常监听 |
| **功能/界面预览** | 在 README 中增加「功能/界面预览」区块，按模块依次贴图，并在开头提示浏览者先看预览 | 便于 GitHub 访客快速了解界面与能力；仓库简介可提示「建议先看 README 功能/界面预览」 |

### 12.4 扩展时注意

- 新增检索策略或生成参数时，建议同步在 Gradio 对应控件上增加 `info`（成本/效率一句话）。
- 新增 RAGAS 相关指标时，若需纳入「优化建议」，在 `get_optimization_suggestions` 中增加对应阈值与建议文案，并在标准答案评测 UI 中暴露目标输入。

---

## 引用文件索引

| 文件 | 内容 |
|------|------|
| [`refs/architecture.md`](refs/architecture.md) | 架构详解、数据流图、模块职责 |
| [`refs/extension-scenarios.md`](refs/extension-scenarios.md) | 扩展场景 step-by-step 代码示例 |
| [`refs/troubleshooting.md`](refs/troubleshooting.md) | 常见报错排查清单 |
| [`refs/default-params.md`](refs/default-params.md) | 默认参数完整说明及调优建议 |
| [`refs/implicit-knowledge.md`](refs/implicit-knowledge.md) | 隐性知识：团队约定与运行环境上下文 |
| [`docs/release-notes-v0.4.md`](../../docs/release-notes-v0.4.md) | v0.4 版本发布说明 |
