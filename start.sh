#!/usr/bin/env bash
# RAG Web System 一键启动脚本
# 用法：chmod +x start.sh && ./start.sh
# 可选参数：
#   --skip-ollama   跳过 Ollama 检查（适用于已单独管理 Ollama 的场景）
#   --port PORT     指定端口（默认 8000）
#   --host HOST     指定监听地址（默认 0.0.0.0）
#   --prod          使用 Gunicorn 生产模式启动
#   --without-reranker 不安装重排序依赖（默认会安装）

set -euo pipefail

# ── 颜色 ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
die()   { fail "$@"; exit 1; }

# ── 参数解析 ──────────────────────────────────────────────────
SKIP_OLLAMA=false
PORT=8000
HOST="0.0.0.0"
PROD=false
WITH_RERANKER=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ollama) SKIP_OLLAMA=true; shift ;;
        --port)        PORT="$2"; shift 2 ;;
        --host)        HOST="$2"; shift 2 ;;
        --prod)        PROD=true; shift ;;
        --without-reranker) WITH_RERANKER=false; shift ;;
        -h|--help)
            echo "用法: ./start.sh [选项]"
            echo "  --skip-ollama   跳过 Ollama 检查"
            echo "  --port PORT     指定端口（默认 8000）"
            echo "  --host HOST     指定监听地址（默认 0.0.0.0）"
            echo "  --prod          使用 Gunicorn 生产模式启动"
            echo "  --without-reranker 不安装重排序依赖（默认会安装）"
            exit 0 ;;
        *) die "未知参数: $1（使用 --help 查看帮助）" ;;
    esac
done

# ── 切换到项目根目录 ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "  RAG Web System 一键启动"
echo "=========================================="
echo ""

# ── 1. 检查 Python ────────────────────────────────────────────
info "检查 Python 版本..."

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

[[ -z "$PYTHON" ]] && die "需要 Python 3.10+，请先安装。"
ok "Python $($PYTHON --version 2>&1 | awk '{print $2}')"

# ── 2. 创建/激活虚拟环境 ──────────────────────────────────────
VENV_DIR=".venv"

if [[ ! -d "$VENV_DIR" ]]; then
    info "创建虚拟环境..."
    $PYTHON -m venv "$VENV_DIR"
    ok "虚拟环境已创建：$VENV_DIR"
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
    VENV_VER=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    VENV_MAJOR=$(echo "$VENV_VER" | cut -d. -f1)
    VENV_MINOR=$(echo "$VENV_VER" | cut -d. -f2)
    if [[ "$VENV_MAJOR" -lt 3 || "$VENV_MINOR" -lt 10 ]]; then
        warn "检测到旧虚拟环境 Python $VENV_VER，正在重建为 $($PYTHON --version 2>&1 | awk '{print $2}')"
        rm -rf "$VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        ok "虚拟环境已重建：$VENV_DIR"
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "虚拟环境已激活"

# ── 3. 安装依赖 ───────────────────────────────────────────────
MARKER="$VENV_DIR/.deps_installed"
TOML_HASH=$(md5sum pyproject.toml 2>/dev/null || md5 -q pyproject.toml 2>/dev/null || echo "unknown")

if [[ ! -f "$MARKER" ]] || [[ "$(cat "$MARKER" 2>/dev/null)" != "$TOML_HASH" ]]; then
    info "安装项目依赖（首次或 pyproject.toml 变更时）..."
    pip install --quiet --upgrade pip
    pip install --quiet -e .
    echo "$TOML_HASH" > "$MARKER"
    ok "依赖安装完成"
else
    ok "依赖已是最新"
fi

if [[ "$WITH_RERANKER" == "true" ]]; then
    RERANKER_MARKER="$VENV_DIR/.deps_reranker_installed"
    if [[ ! -f "$RERANKER_MARKER" ]] || [[ "$(cat "$RERANKER_MARKER" 2>/dev/null)" != "$TOML_HASH" ]]; then
        info "安装重排序依赖（sentence-transformers）..."
        pip install --quiet -e ".[reranker]"
        echo "$TOML_HASH" > "$RERANKER_MARKER"
        ok "重排序依赖安装完成"
    else
        ok "重排序依赖已是最新"
    fi
fi

# ── 4. 配置 .env ──────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        warn ".env 文件不存在，已从 .env.example 复制。请编辑 .env 配置 LLM_BASE_URL 和 LLM_API_KEY。"
    else
        die ".env 和 .env.example 均不存在，无法启动。"
    fi
else
    ok ".env 配置文件就绪"
fi

APP_API_TOKEN=$(grep -E "^APP_API_TOKEN=" .env 2>/dev/null | cut -d= -f2- || echo "")
if [[ -n "${APP_API_TOKEN}" ]]; then
    info "已启用 API Token 认证（请求 /ingest /query /evaluate 需 Bearer Token）"
fi

# ── 5. 检查 Ollama ────────────────────────────────────────────
EMBEDDING_MODEL=$(grep -E "^EMBEDDING_MODEL_NAME=" .env 2>/dev/null | cut -d= -f2 || echo "bge-m3")
EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge-m3}"
OLLAMA_URL=$(grep -E "^OLLAMA_BASE_URL=" .env 2>/dev/null | cut -d= -f2 || echo "http://localhost:11434")
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

if [[ "$SKIP_OLLAMA" == "true" ]]; then
    warn "跳过 Ollama 检查（--skip-ollama）"
else
    info "检查 Ollama 服务..."

    # 尝试启动 Ollama（如果未运行）
    if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        if command -v ollama &>/dev/null; then
            info "Ollama 未运行，尝试后台启动..."
            ollama serve &>/dev/null &
            OLLAMA_PID=$!
            # 等待 Ollama 就绪（最多 15 秒）
            for i in $(seq 1 15); do
                if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
                    break
                fi
                sleep 1
            done
            if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
                ok "Ollama 服务已启动（PID: $OLLAMA_PID）"
            else
                warn "Ollama 启动超时。Embedding 功能可能不可用，请手动运行: ollama serve"
            fi
        else
            warn "Ollama 未安装或未运行（${OLLAMA_URL}）。Embedding 功能不可用。"
            warn "安装 Ollama: https://ollama.com/download"
        fi
    else
        ok "Ollama 服务运行中（${OLLAMA_URL}）"
    fi

    # 检查 Embedding 模型
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        if curl -sf "${OLLAMA_URL}/api/tags" | grep -q "$EMBEDDING_MODEL"; then
            ok "Embedding 模型 ${EMBEDDING_MODEL} 已就绪"
        else
            info "拉取 Embedding 模型 ${EMBEDDING_MODEL}..."
            if ollama pull "$EMBEDDING_MODEL" 2>/dev/null; then
                ok "模型 ${EMBEDDING_MODEL} 拉取完成"
            else
                warn "模型拉取失败，请手动运行: ollama pull ${EMBEDDING_MODEL}"
            fi
        fi
    fi
fi

# ── 6. 创建数据目录 ───────────────────────────────────────────
mkdir -p data vector_store
ok "数据目录就绪"

# ── 7. 检查端口 ───────────────────────────────────────────────
if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    warn "端口 ${PORT} 已被占用"
    EXISTING_PID=$(lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null | head -1)
    if [[ -n "$EXISTING_PID" ]]; then
        warn "占用进程 PID: ${EXISTING_PID}（可用 kill $EXISTING_PID 终止，或 --port 指定其他端口）"
    fi
    die "端口冲突，请释放端口 ${PORT} 或使用 --port 指定其他端口。"
fi

# ── 8. 启动服务 ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  启动 RAG Web System"
echo "=========================================="
echo ""
info "监听地址: http://${HOST}:${PORT}"
info "Gradio UI: http://127.0.0.1:${PORT}/ui"
info "API 文档:  http://127.0.0.1:${PORT}/docs"
info "按 Ctrl+C 停止服务"
echo ""

if [[ "$PROD" == "true" ]]; then
    info "模式: 生产（Gunicorn）"
    exec gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b "${HOST}:${PORT}" app.main:app
else
    info "模式: 开发（Uvicorn + 热重载）"
    exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
fi
