from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.core.errors import IngestError
from app.rag.bm25_store import BM25Store
from app.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def _load_pdf_with_mineru(path: Path) -> list[Document]:
    try:
        import magic_pdf.model as model_config
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    except ImportError as exc:
        raise IngestError(
            "MinerU (magic-pdf) 未安装。请执行: pip install 'rag-web-system[mineru]'"
        ) from exc

    model_config.__use_inside_model__ = True
    pdf_bytes = path.read_bytes()
    ds = PymuDocDataset(pdf_bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir = os.path.join(tmpdir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(image_dir)

        try:
            parse_method = ds.classify()
            if parse_method == model_config.SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            md_content = pipe_result.get_markdown(image_dir)
        except Exception as exc:
            raise IngestError(f"MinerU PDF 解析失败: {exc}") from exc

    return [Document(page_content=md_content, metadata={"source": str(path)})]


class IngestService:
    def __init__(
        self,
        settings: Settings,
        vector_store_manager: VectorStoreManager,
        bm25_store: BM25Store,
    ):
        self._settings = settings
        self._vector_store_manager = vector_store_manager
        self._bm25_store = bm25_store

    def ingest(
        self,
        paths: list[str] | None = None,
        directory: str | None = None,
        recursive: bool = True,
        pdf_loader: str = "pypdf",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunk_separators: list[str] | None = None,
    ) -> tuple[int, list[str]]:
        docs: list[Document] = []
        sources: list[str] = []

        if paths:
            for path in paths:
                file_path = Path(path)
                docs.extend(self._load_one(file_path, pdf_loader))
                sources.append(str(file_path))

        if directory:
            directory_path = Path(directory)
            if not directory_path.exists():
                raise IngestError(f"目录不存在: {directory}")
            pattern = "**/*" if recursive else "*"
            for file_path in directory_path.glob(pattern):
                if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                docs.extend(self._load_one(file_path, pdf_loader))
                sources.append(str(file_path))

        if not docs:
            return 0, []

        splitter = self._build_splitter(chunk_size, chunk_overlap, chunk_separators)
        split_docs = self._split_docs(docs, splitter)

        count = self._vector_store_manager.add_documents(split_docs)
        self._bm25_store.add_documents(split_docs)

        logger.info("Ingested %d chunks from %d sources", count, len(sources))
        return count, sorted(set(sources))

    def _load_one(self, path: Path, pdf_loader: str = "pypdf") -> list[Document]:
        if not path.exists():
            raise IngestError(f"文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise IngestError(f"不支持的文件格式: {suffix}")

        if suffix == ".pdf":
            if pdf_loader == "mineru":
                return _load_pdf_with_mineru(path)
            return PyPDFLoader(str(path)).load()

        return TextLoader(str(path), encoding="utf-8").load()

    def _build_splitter(
        self,
        chunk_size: int | None,
        chunk_overlap: int | None,
        separators: list[str] | None,
    ) -> RecursiveCharacterTextSplitter:
        size = chunk_size or self._settings.chunk_size
        overlap = chunk_overlap or self._settings.chunk_overlap
        seps = separators if separators else DEFAULT_SEPARATORS
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=seps,
        )

    def _split_docs(
        self, docs: list[Document], splitter: RecursiveCharacterTextSplitter
    ) -> list[Document]:
        chunks = splitter.split_documents(docs)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
        return chunks
