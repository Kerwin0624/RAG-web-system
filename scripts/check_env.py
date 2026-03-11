"""RAG Web System 环境检测脚本

跨平台运行：python scripts/check_env.py
检查项：Python 版本、核心依赖包、Ollama 连接、Embedding 模型、LLM 配置、目录权限。
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

if os.name == "nt":
    os.system("")


def header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_python_version() -> bool:
    header("Python 环境")
    ver = sys.version_info
    ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    if ver >= (3, 10):
        print(f"  {PASS} Python {ver_str} (>= 3.10)")
        return True
    print(f"  {FAIL} Python {ver_str} — 需要 3.10+")
    return False


def check_core_packages() -> tuple[int, int]:
    header("核心依赖包")
    packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("gradio", "gradio"),
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_ollama", "langchain-ollama"),
        ("langchain_chroma", "langchain-chroma"),
        ("langchain_text_splitters", "langchain-text-splitters"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("pypdf", "pypdf"),
        ("rank_bm25", "rank-bm25"),
        ("jieba", "jieba"),
        ("ragas", "ragas"),
        ("datasets", "datasets"),
    ]
    passed = 0
    failed = 0
    for import_name, pip_name in packages:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {PASS} {pip_name} ({ver})")
            passed += 1
        except ImportError:
            print(f"  {FAIL} {pip_name} — pip install {pip_name}")
            failed += 1
    return passed, failed


def check_optional_packages() -> None:
    header("可选依赖包")
    optionals = [
        ("sentence_transformers", "sentence-transformers", "reranker 重排序"),
        ("magic_pdf", "magic-pdf", "MinerU PDF 解析"),
        ("pytest", "pytest", "测试框架"),
        ("httpx", "httpx", "HTTP 测试客户端"),
        ("ruff", "ruff", "代码检查"),
    ]
    for import_name, pip_name, desc in optionals:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {PASS} {pip_name} ({ver}) — {desc}")
        except ImportError:
            print(f"  {WARN} {pip_name} 未安装 — {desc}")


def check_ollama() -> bool:
    header("Ollama 服务")

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.environ.get("EMBEDDING_MODEL_NAME", "bge-m3")

    if shutil.which("ollama"):
        print(f"  {PASS} ollama CLI 可用")
    else:
        print(f"  {WARN} ollama CLI 未在 PATH 中找到")

    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            print(f"  {PASS} Ollama 服务可达 ({ollama_url})")
            print(f"  {INFO} 已拉取模型: {', '.join(models) if models else '(无)'}")

            model_found = any(model_name in m for m in models)
            if model_found:
                print(f"  {PASS} Embedding 模型 '{model_name}' 已就绪")
            else:
                print(f"  {FAIL} Embedding 模型 '{model_name}' 未找到")
                print(f"         运行: ollama pull {model_name}")
                return False
            return True
    except (urllib.error.URLError, OSError) as exc:
        print(f"  {FAIL} 无法连接 Ollama ({ollama_url}): {exc}")
        print(f"         请确保 Ollama 正在运行: ollama serve")
        return False


def check_env_file() -> bool:
    header("环境配置 (.env)")

    env_path = Path(".env")
    example_path = Path(".env.example")

    if not env_path.exists():
        if example_path.exists():
            print(f"  {FAIL} .env 文件不存在")
            copy_cmd = "copy .env.example .env" if os.name == "nt" else "cp .env.example .env"
            print(f"         运行: {copy_cmd}")
        else:
            print(f"  {FAIL} .env 和 .env.example 均不存在")
        return False

    print(f"  {PASS} .env 文件存在")

    env_vars: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    required = ["LLM_BASE_URL", "LLM_MODEL"]
    all_ok = True
    for key in required:
        val = env_vars.get(key, "")
        env_val = os.environ.get(key, "")
        effective = env_val or val
        if not effective or effective in ("", "replace-with-your-key", "https://your-llm-endpoint/v1"):
            print(f"  {FAIL} {key} 未配置或为占位符")
            all_ok = False
        else:
            display = effective[:30] + "..." if len(effective) > 30 else effective
            print(f"  {PASS} {key} = {display}")

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or env_vars.get("LLM_API_KEY", "")
    if not api_key or api_key == "replace-with-your-key":
        print(f"  {WARN} LLM_API_KEY 未设置（部分模型可能不需要）")
    else:
        masked = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
        print(f"  {PASS} LLM_API_KEY = {masked}")

    return all_ok


def check_directories() -> bool:
    header("目录权限")
    dirs = {
        "data": Path(os.environ.get("DATA_DIR", "./data")),
        "vector_store": Path(os.environ.get("VECTOR_STORE_DIR", "./vector_store")),
    }
    all_ok = True
    for name, path in dirs.items():
        if path.exists():
            if os.access(str(path), os.W_OK):
                print(f"  {PASS} {name}: {path} (可写)")
            else:
                print(f"  {FAIL} {name}: {path} (不可写)")
                all_ok = False
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  {PASS} {name}: {path} (已创建)")
            except OSError as exc:
                print(f"  {FAIL} {name}: {path} (创建失败: {exc})")
                all_ok = False
    return all_ok


def check_system_info() -> None:
    header("系统信息")
    print(f"  {INFO} 操作系统: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  {INFO} Python 路径: {sys.executable}")
    print(f"  {INFO} 工作目录: {os.getcwd()}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  {INFO} pip: {result.stdout.strip()}")
    except Exception:
        pass


def main() -> None:
    print("\n" + "=" * 60)
    print("  RAG Web System 环境检测")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    check_system_info()

    ok = check_python_version()
    results.append(("Python 版本", ok))

    passed, failed = check_core_packages()
    results.append(("核心依赖", failed == 0))

    check_optional_packages()

    ok = check_env_file()
    results.append(("环境配置", ok))

    ok = check_ollama()
    results.append(("Ollama 服务", ok))

    ok = check_directories()
    results.append(("目录权限", ok))

    header("检测结果汇总")
    all_pass = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n  \033[92m所有检查通过！可以启动服务。\033[0m")
        print(f"  运行: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    else:
        print(f"\n  \033[91m部分检查未通过，请根据上述提示修复后重试。\033[0m")

    print()


if __name__ == "__main__":
    main()
