from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import time
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException, status
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.errors import EvaluationError, GenerationError, IngestError, RetrievalError
from app.core.logging import setup_logging
from app.eval.reporting import export_metrics
from app.eval.ragas_eval import RagasEvaluator
from app.eval.testset_runner import TestsetRunner
from app.rag.bm25_store import BM25Store
from app.rag.generator import LLMGenerator
from app.rag.ingest import IngestService
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import RerankerService
from app.rag.retriever import RetrieverService
from app.rag.vector_store import VectorStoreManager
from app.schemas.rag import EvalRequest, EvalResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.web.gradio_ui import build_gradio_ui

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_report_dir(report_dir: str | None) -> Path:
    base_dir = settings.reports_dir.resolve()
    if not report_dir:
        return base_dir
    if report_dir in (".", "reports", "./reports"):
        return base_dir
    target = Path(report_dir).expanduser()
    if not target.is_absolute():
        target = base_dir / target
    target = target.resolve()
    if not _is_subpath(target, base_dir):
        raise EvaluationError(f"报告目录必须位于 {base_dir} 下")
    return target


@dataclass
class AppServices:
    ingest_service: IngestService
    pipeline: RAGPipeline
    evaluator: RagasEvaluator
    testset_runner: TestsetRunner


def _build_services() -> AppServices:
    vector_store_manager = VectorStoreManager(settings=settings)
    bm25_store = BM25Store(corpus_path=settings.vector_store_dir / "bm25_corpus.pkl")
    reranker_service = RerankerService(model_name=settings.reranker_model)
    ingest_service = IngestService(
        settings=settings,
        vector_store_manager=vector_store_manager,
        bm25_store=bm25_store,
    )
    retriever_service = RetrieverService(
        settings=settings,
        vector_store_manager=vector_store_manager,
        bm25_store=bm25_store,
        reranker=reranker_service,
    )
    generator_service = LLMGenerator(settings=settings)
    pipeline = RAGPipeline(retriever=retriever_service, generator=generator_service)
    eval_llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )
    eval_embeddings = OllamaEmbeddings(
        model=settings.embedding_model_name,
        base_url=settings.ollama_base_url,
    )
    evaluator = RagasEvaluator(llm=eval_llm, embeddings=eval_embeddings)
    testset_runner = TestsetRunner(pipeline=pipeline, evaluator=evaluator)
    return AppServices(
        ingest_service=ingest_service,
        pipeline=pipeline,
        evaluator=evaluator,
        testset_runner=testset_runner,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = _build_services()
    logger.info("应用服务初始化完成")
    yield


app = FastAPI(title=settings.app_name, version="0.4.0", lifespan=lifespan)


def _get_services() -> AppServices:
    return app.state.services


def _ingest_impl(req: IngestRequest) -> IngestResponse:
    services = _get_services()
    count, sources = services.ingest_service.ingest(
        paths=req.paths,
        directory=req.directory,
        recursive=req.recursive,
        pdf_loader=req.pdf_loader,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        chunk_separators=req.chunk_separators,
    )
    return IngestResponse(chunks_indexed=count, sources=sources)


def _query_impl(req: QueryRequest) -> QueryResponse:
    services = _get_services()
    result = services.pipeline.ask(
        question=req.question,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        search_type=req.search_type,
        vector_weight=req.vector_weight,
        bm25_weight=req.bm25_weight,
        reranker_enabled=req.reranker_enabled,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        model=req.model,
    )
    return QueryResponse(**result)


def _evaluate_impl(req: EvalRequest) -> EvalResponse:
    services = _get_services()
    metrics = services.evaluator.evaluate_samples(req.samples)
    report_dir = _resolve_report_dir(req.report_dir)
    report_files = export_metrics(metrics, report_dir) if req.save_report else None
    return EvalResponse(metrics=metrics, report_files=report_files)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    if settings.app_api_token and request.url.path.startswith(("/ingest", "/query", "/evaluate")):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if token != settings.app_api_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Unauthorized"},
            )
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Process-Time-MS"] = str(elapsed_ms)
    logger.info("%s %s -> %sms", request.method, request.url.path, elapsed_ms)
    return response


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        return _ingest_impl(req)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected ingest error")
        raise HTTPException(status_code=500, detail="Ingest failed") from exc


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return _query_impl(req)
    except (RetrievalError, GenerationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected query error")
        raise HTTPException(status_code=500, detail="Query failed") from exc


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest) -> EvalResponse:
    try:
        return _evaluate_impl(req)
    except EvaluationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected evaluation error")
        raise HTTPException(status_code=500, detail="Evaluation failed") from exc


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


ui = build_gradio_ui(
    query_handler=lambda request: _query_impl(request).model_dump(),
    eval_handler=lambda request: _evaluate_impl(request).model_dump(),
    ingest_handler=lambda request: _ingest_impl(request).model_dump(),
    testset_eval_handler=lambda **kwargs: _get_services().testset_runner.run(**kwargs),
)
app = gr.mount_gradio_app(app, ui, path="/ui")
