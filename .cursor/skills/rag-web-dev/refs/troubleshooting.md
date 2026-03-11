# 常见报错排查清单

## 1. Ollama 连接失败

**报错信息**：
```
ConnectionError: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded
```
或
```
requests.exceptions.ConnectionError: Connection refused
```

**排查步骤**：

1. 确认 Ollama 正在运行：
   ```bash
   # Windows / macOS / Linux
   curl http://localhost:11434/api/tags
   ```
2. 若未启动：
   ```bash
   ollama serve
   ```
3. 确认 Embedding 模型已拉取：
   ```bash
   ollama list
   # 应显示 bge-m3
   ollama pull bge-m3   # 若不存在
   ```
4. 若 Ollama 运行在非默认端口，更新 `.env`：
   ```
   OLLAMA_BASE_URL=http://localhost:YOUR_PORT
   ```
5. Docker 环境下 Ollama 在宿主机运行时：
   ```
   # Linux
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   # 或使用宿主机 IP
   OLLAMA_BASE_URL=http://192.168.x.x:11434
   ```

---

## 2. LLM API 调用失败

**报错信息**：
```
GenerationError: Generation failed: Error code: 401
```
或
```
openai.AuthenticationError: Incorrect API key provided
```

**排查步骤**：

1. 确认 `.env` 中 `LLM_API_KEY` 正确
2. 确认 `LLM_BASE_URL` 末尾包含 `/v1`（大多数 OpenAI 兼容 API 需要）
3. 确认 `LLM_MODEL` 名称与供应商一致
4. 测试 API 连通性：
   ```bash
   curl -H "Authorization: Bearer YOUR_KEY" YOUR_BASE_URL/models
   ```
5. 检查 API 余额/配额是否耗尽
6. 如果使用阿里云百炼（DashScope），可设置 `DASHSCOPE_API_KEY` 环境变量作为回退

---

## 3. 入库失败

### 3.1 "文件不存在" / "目录不存在"

```
IngestError: 文件不存在: xxx
IngestError: 目录不存在: xxx
```

- 检查路径是否正确（Windows 注意反斜杠 `\` vs `/`）
- 确认文件扩展名在支持列表中：`.txt`、`.md`、`.pdf`

### 3.2 "不支持的文件格式"

```
IngestError: 不支持的文件格式: .xxx
```

- 仅支持 `.txt`、`.md`、`.pdf`
- 如需扩展，参考 `refs/extension-scenarios.md` 的"添加新文档格式"

### 3.3 PDF 解析失败（MinerU）

```
IngestError: MinerU (magic-pdf) 未安装
IngestError: MinerU PDF 解析失败: xxx
```

- 安装 MinerU: `pip install "rag-web-system[mineru]"`
- MinerU 依赖较重，确保系统满足 magic-pdf 的依赖要求
- 可回退到 pypdf: 在 UI 或 API 中将 `pdf_loader` 设为 `pypdf`

### 3.4 入库后检索不到结果

- 确认入库成功（API 返回 `chunks_indexed > 0`）
- 检查向量库目录 `vector_store/` 是否有文件生成
- 尝试降低 `SEARCH_SCORE_THRESHOLD`（如设为 0.1）
- 确认 Ollama Embedding 服务正常

---

## 4. 检索相关

### 4.1 混合检索结果为空

```
RetrievalError: 检索失败: xxx
```

- 确认 BM25 索引已建立（`vector_store/bm25_corpus.pkl` 文件存在）
- 若 BM25 文件损坏，删除后重新入库：
  ```bash
  # Windows
  del vector_store\bm25_corpus.pkl
  # macOS / Linux
  rm vector_store/bm25_corpus.pkl
  ```

### 4.2 检索结果质量差

- 尝试切换检索策略（`hybrid` 通常优于单一策略）
- 调整 `VECTOR_WEIGHT` / `BM25_WEIGHT` 比例
- 启用 Reranker（需安装 `sentence-transformers`）
- 增大 `SEARCH_TOP_K`
- 调整分块参数（较小的 `CHUNK_SIZE` 提高精度，较大的保留上下文）

### 4.3 Reranker 报错

```
RuntimeError: sentence-transformers is required for reranking
```

- 安装: `pip install "rag-web-system[reranker]"`
- 首次加载模型较慢（需下载 ~1GB 模型文件）
- 如在国内网络环境，设置 HuggingFace 镜像：
  ```bash
  # Windows PowerShell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  # macOS / Linux
  export HF_ENDPOINT=https://hf-mirror.com
  ```

---

## 5. RAGAS 评估问题

### 5.1 "RAGAS not available"

```json
{"status": "skipped", "reason": "RAGAS not available or import failed: ..."}
```

- 确认 `ragas` 已安装: `pip install ragas>=0.2.10`
- RAGAS 依赖 LLM 进行评估，确保 `LLM_BASE_URL` / `LLM_API_KEY` 有效
- RAGAS 评估较慢（20 条 ~数分钟），耐心等待

### 5.2 评估结果全为 0 或 NaN

- 确认评估样本格式正确（`question`、`answer`、`ground_truth`、`contexts`）
- 确认 `contexts` 不为空列表
- 检查 LLM 是否返回有效内容

### 5.3 评估超时

- 增大 `LLM_TIMEOUT`（如设为 120）
- 减少测试集条目数
- 考虑使用更快的 LLM 模型

---

## 6. Gradio UI 问题

### 6.1 无法访问 `/ui/`

- 确认服务已启动且端口未被占用
- 检查防火墙/安全组设置
- 确认 `HOST=0.0.0.0`（非 `127.0.0.1`）以允许外部访问

### 6.2 中文显示乱码

- 确认系统安装了中文字体
- RAGAS 图表乱码：系统需安装 SimHei 或 Microsoft YaHei 字体
- Linux 安装中文字体：
  ```bash
  sudo apt install fonts-wqy-microhei
  ```

---

## 7. Docker 相关

### 7.1 构建失败

- 确保 `pyproject.toml` 和 `README.md` 存在（Dockerfile COPY 阶段需要）
- 检查网络（`pip install` 需要访问 PyPI）
- 使用国内镜像加速：
  ```dockerfile
  RUN pip install --no-cache-dir -e . -i https://mirrors.aliyun.com/pypi/simple/
  ```

### 7.2 容器内无法连接 Ollama

- Ollama 运行在宿主机时，容器内不能用 `localhost`
- 使用 `host.docker.internal`（Docker Desktop）或宿主机实际 IP
- 在 `.env` 中：
  ```
  OLLAMA_BASE_URL=http://host.docker.internal:11434
  ```

### 7.3 数据丢失

- 确认 `docker-compose.yml` 中 volume 挂载正确：
  ```yaml
  volumes:
    - ./data:/app/data
    - ./vector_store:/app/vector_store
  ```
- 不要使用 `docker compose down -v`（会删除 volume）

---

## 8. 通用排查技巧

### 查看详细日志

```bash
# 设置 DEBUG 级别
LOG_LEVEL=DEBUG uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 运行环境检测

```bash
python scripts/check_env.py
```

### 重置向量库

```bash
# Windows
rmdir /s /q vector_store
mkdir vector_store

# macOS / Linux
rm -rf vector_store/ && mkdir vector_store
```

### 验证 API 连通性

```bash
# 健康检查
curl http://localhost:8000/health

# Ollama
curl http://localhost:11434/api/tags

# LLM API
curl -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hello"}]}' \
  YOUR_BASE_URL/chat/completions
```
