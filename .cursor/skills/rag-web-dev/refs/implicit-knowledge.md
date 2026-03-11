# 隐性知识：团队约定与运行环境上下文

> 以下信息无法从代码直接推断，由项目维护者确认记录。

## 1. LLM 供应商

- **主要平台**: 阿里云百炼（DashScope）
- 代码中 `config.py` 存在 `DASHSCOPE_API_KEY` 环境变量回退机制
- 推荐配置：

```
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=qwen3.5-plus
```

- Gradio UI 预设模型列表（Kimi / MiniMax / GLM / Qwen）均通过百炼统一入口访问
- 如需切换供应商，只需更改 `LLM_BASE_URL` 和 `LLM_API_KEY`

## 2. Ollama 部署

- **方式**: 不固定，按需选择
- 开发时通常为本机安装 Ollama
- Docker 部署时需将 `OLLAMA_BASE_URL` 指向宿主机或独立 Ollama 服务
- 建议在 Skill/文档中保留多种部署方式的说明

## 3. 目标文档类型

- **主要处理**: 中文业务文档、合同、报告
- 这解释了以下设计决策：
  - `CHUNK_SIZE=800`：约 400 个中文字符，适合段落级中文文本
  - Embedding 选择 `bge-m3`：优秀的中文语义理解能力
  - `jieba` 分词用于 BM25：专为中文设计
  - System Prompt 中"优先使用中文"的约束
  - 分隔符包含中文标点（`。！？；，`）
- 中文业务文档建议：
  - 分块时可考虑添加中文句号 `。` 作为优先分隔符
  - 混合检索（`hybrid`）对中文效果通常优于纯向量检索
  - 关键术语（如合同条款编号、专有名词）依赖 BM25 精确匹配

## 4. 项目定位

- **个人学习/研究项目**
- 当前无需考虑高并发和分布式部署
- Gunicorn 默认 2 worker 足够
- Pipeline LRU 缓存 256 条适用于单人使用场景
- 侧重功能迭代和 RAG 调优实验

## 5. 评测集构建规范

- **方法**: LLM 生成后人工筛选
- 评测集文件: `data/rag_eval_set.json`
- JSON 格式：
  ```json
  [
    {
      "id": "q-1",
      "question": "问题文本",
      "gold_answer": "标准答案",
      "question_type": "事实型",
      "difficulty": "简单",
      "has_answer": true
    }
  ]
  ```
- 构建流程：
  1. 针对入库文档，用 LLM 生成候选 QA 对
  2. 人工审核：删除低质量/歧义/与文档不符的条目
  3. 补充边界 case（如"无法回答"场景，`has_answer: false`）
  4. 标注 `question_type`（事实型/推理型/对比型等）和 `difficulty`

## 6. 网络环境

- **国内网络**，需要 HuggingFace 镜像
- 下载 Reranker 模型时需配置：
  ```bash
  # Windows PowerShell
  $env:HF_ENDPOINT = "https://hf-mirror.com"

  # macOS / Linux
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- pip 安装建议使用国内镜像（可选）：
  ```bash
  pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
  ```
- Docker 构建时也建议配置 pip 镜像以加速
