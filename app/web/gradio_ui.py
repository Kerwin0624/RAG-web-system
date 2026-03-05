from __future__ import annotations

import json
from typing import Callable

import gradio as gr

from app.schemas.rag import EvalRequest, EvalSample, IngestRequest, QueryRequest

SEPARATOR_CHOICES = [
    ("\\n\\n（双换行）", "\n\n"),
    ("\\n（换行）", "\n"),
    ("。（句号）", "。"),
    ("！（感叹号）", "！"),
    ("？（问号）", "？"),
    ("；（分号）", "；"),
    ("，（逗号）", "，"),
    ("空格", " "),
    ("无（空字符串）", ""),
]

SEPARATOR_LABELS = [label for label, _ in SEPARATOR_CHOICES]
SEPARATOR_MAP = {label: value for label, value in SEPARATOR_CHOICES}
DEFAULT_SEPARATOR_LABELS = ["\\n\\n（双换行）", "\\n（换行）", "空格", "无（空字符串）"]


def build_gradio_ui(
    query_handler: Callable[[QueryRequest], dict],
    eval_handler: Callable[[EvalRequest], dict],
    ingest_handler: Callable[[IngestRequest], dict],
    testset_eval_handler: Callable[..., dict] | None = None,
) -> gr.Blocks:
    with gr.Blocks(title="RAG 智能问答系统") as demo:
        gr.Markdown("# RAG 智能问答系统")

        # ── Tab 1: 知识库构建 ──────────────────────────────
        with gr.Tab("知识库构建"):
            gr.Markdown("上传或指定本地文件路径，将文档向量化入库后即可在问答中使用。")

            with gr.Row():
                pdf_loader = gr.Dropdown(
                    label="PDF 解析引擎",
                    choices=["pypdf", "mineru"],
                    value="pypdf",
                    info="MinerU 解析更精准，需额外安装 magic-pdf",
                )

            with gr.Row():
                upload_files = gr.File(
                    label="上传文档（支持 .txt / .md / .pdf）",
                    file_types=[".txt", ".md", ".pdf"],
                    file_count="multiple",
                    type="filepath",
                )
            manual_paths = gr.Textbox(
                label="本地文件绝对路径（可选，多行，每行一个路径）",
                placeholder=r"例如：D:\Edge Download\RAG痛点分析与优化方案实践_Advanced RAG.pdf",
                lines=3,
            )
            directory = gr.Textbox(
                label="目录入库（可选）",
                placeholder="例如：./data 或 D:\\docs",
            )
            recursive = gr.Checkbox(label="递归扫描子目录", value=True)

            gr.Markdown("### 分块设置")
            with gr.Row():
                chunk_size = gr.Slider(
                    label="分块大小",
                    value=800,
                    minimum=100,
                    maximum=4000,
                    step=50,
                    info="每个文本块的最大字符数",
                )
                chunk_overlap = gr.Slider(
                    label="分块重叠",
                    value=120,
                    minimum=0,
                    maximum=1000,
                    step=10,
                    info="相邻文本块之间的重叠字符数",
                )

            separator_checkboxes = gr.CheckboxGroup(
                label="分块分隔符（按优先级排列，从上到下依次尝试）",
                choices=SEPARATOR_LABELS,
                value=DEFAULT_SEPARATOR_LABELS,
                info="选择用于切分文本的分隔符",
            )

            ingest_btn = gr.Button("开始入库", variant="primary")
            ingest_result = gr.Textbox(label="入库结果", lines=8)

            def _ingest(files, manual, dir_path, rec, loader, c_size, c_overlap, sep_labels):
                try:
                    paths: list[str] = []
                    if files:
                        if isinstance(files, str):
                            paths.append(files)
                        else:
                            for f in files:
                                if f:
                                    paths.append(str(f))
                    if manual:
                        for line in str(manual).splitlines():
                            line = line.strip()
                            if line:
                                paths.append(line)

                    separators = None
                    if sep_labels:
                        separators = [SEPARATOR_MAP[lbl] for lbl in sep_labels if lbl in SEPARATOR_MAP]

                    req = IngestRequest(
                        paths=paths,
                        directory=dir_path or None,
                        recursive=bool(rec),
                        pdf_loader=loader,
                        chunk_size=int(c_size),
                        chunk_overlap=int(c_overlap),
                        chunk_separators=separators,
                    )
                    res = ingest_handler(req)
                    sources = "\n".join(res.get("sources", []))
                    return f"已索引分块数: {res.get('chunks_indexed', 0)}\n数据源:\n{sources or '（无）'}"
                except Exception as exc:
                    return f"入库失败: {exc}"

            ingest_btn.click(
                _ingest,
                inputs=[
                    upload_files, manual_paths, directory, recursive,
                    pdf_loader, chunk_size, chunk_overlap, separator_checkboxes,
                ],
                outputs=[ingest_result],
            )

        # ── Tab 2: 问答 ──────────────────────────────
        with gr.Tab("问答"):
            question = gr.Textbox(label="问题", placeholder="请输入问题", lines=3)

            gr.Markdown("### 检索设置")
            with gr.Row():
                top_k = gr.Slider(label="检索数量（Top-K）", value=4, minimum=1, maximum=20, step=1)
                score_threshold = gr.Slider(
                    label="相似度阈值",
                    value=0.25,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                )
                search_type = gr.Dropdown(
                    label="检索方式",
                    choices=[
                        ("向量相似度", "similarity"),
                        ("关键词检索（BM25）", "bm25"),
                        ("混合检索", "hybrid"),
                        ("MMR 多样性检索", "mmr"),
                    ],
                    value="similarity",
                )

            with gr.Row(visible=False) as weight_row:
                vector_weight = gr.Slider(
                    label="向量权重",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="混合检索中向量相似度的权重",
                )
                bm25_weight = gr.Slider(
                    label="关键词权重",
                    value=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="混合检索中 BM25 关键词的权重",
                )

            def _on_search_type_change(s_type):
                is_hybrid = s_type == "hybrid"
                return gr.update(visible=is_hybrid)

            search_type.change(
                _on_search_type_change,
                inputs=[search_type],
                outputs=[weight_row],
            )

            def _on_vector_weight_change(vw):
                return round(1.0 - vw, 2)

            vector_weight.change(
                _on_vector_weight_change,
                inputs=[vector_weight],
                outputs=[bm25_weight],
            )

            with gr.Row():
                reranker_enabled = gr.Checkbox(label="启用重排序", value=False, info="使用交叉编码器对检索结果重新排序")

            gr.Markdown("### 生成设置")
            with gr.Row():
                temperature = gr.Slider(label="温度（Temperature）", value=0.2, minimum=0.0, maximum=2.0, step=0.05)
                max_tokens = gr.Slider(label="最大令牌数", value=512, minimum=32, maximum=4096, step=32)
                top_p = gr.Slider(label="采样阈值（Top-P）", value=1.0, minimum=0.1, maximum=1.0, step=0.05)

            ask_btn = gr.Button("生成回答", variant="primary")
            answer = gr.Textbox(label="回答", lines=8)
            citations = gr.JSON(label="引用片段")
            with gr.Row():
                latency = gr.Textbox(label="耗时（毫秒）")
                cache_tag = gr.Textbox(label="缓存命中")

            def _ask(q, k, st, s_type, vw, bw, rerank, temp, m_toks, tp):
                try:
                    req = QueryRequest(
                        question=q,
                        top_k=int(k),
                        score_threshold=float(st),
                        search_type=s_type,
                        vector_weight=float(vw),
                        bm25_weight=float(bw),
                        reranker_enabled=bool(rerank),
                        temperature=float(temp),
                        max_tokens=int(m_toks),
                        top_p=float(tp),
                    )
                    res = query_handler(req)
                    return (
                        res["answer"],
                        res["citations"],
                        str(res["elapsed_ms"]),
                        "是" if res.get("from_cache") else "否",
                    )
                except Exception as exc:
                    return f"请求失败: {exc}", [], "-", "-"

            ask_btn.click(
                _ask,
                inputs=[
                    question, top_k, score_threshold, search_type,
                    vector_weight, bm25_weight, reranker_enabled,
                    temperature, max_tokens, top_p,
                ],
                outputs=[answer, citations, latency, cache_tag],
            )

        # ── Tab 3: 标准答案评测 ──────────────────────────────
        if testset_eval_handler is not None:
            with gr.Tab("标准答案评测"):
                gr.Markdown(
                    "### 端到端 RAG 评测\n"
                    "系统会逐条将测试问题发送给 RAG 管线，将**实际回答**与 `rag_eval_set.json` 中的**标准答案**对比，"
                    "并可选运行 RAGAS 自动评估。"
                )

                ts_path = gr.Textbox(
                    label="评测集路径",
                    value="data/rag_eval_set.json",
                    info="默认使用项目自带测试集，也可填入其他 JSON 文件路径",
                )

                gr.Markdown("### 检索设置")
                with gr.Row():
                    ts_top_k = gr.Slider(label="检索数量（Top-K）", value=4, minimum=1, maximum=20, step=1)
                    ts_threshold = gr.Slider(label="相似度阈值", value=0.25, minimum=0.0, maximum=1.0, step=0.01)
                    ts_search_type = gr.Dropdown(
                        label="检索方式",
                        choices=[
                            ("向量相似度", "similarity"),
                            ("关键词检索（BM25）", "bm25"),
                            ("混合检索", "hybrid"),
                            ("MMR 多样性检索", "mmr"),
                        ],
                        value="similarity",
                    )

                with gr.Row(visible=False) as ts_weight_row:
                    ts_vw = gr.Slider(label="向量权重", value=0.7, minimum=0.0, maximum=1.0, step=0.05)
                    ts_bw = gr.Slider(label="关键词权重", value=0.3, minimum=0.0, maximum=1.0, step=0.05)

                def _on_ts_search_change(s_type):
                    return gr.update(visible=s_type == "hybrid")

                ts_search_type.change(_on_ts_search_change, inputs=[ts_search_type], outputs=[ts_weight_row])

                ts_vw.change(lambda vw: round(1.0 - vw, 2), inputs=[ts_vw], outputs=[ts_bw])

                with gr.Row():
                    ts_reranker = gr.Checkbox(label="启用重排序", value=False)

                gr.Markdown("### 生成设置")
                with gr.Row():
                    ts_temp = gr.Slider(label="温度", value=0.2, minimum=0.0, maximum=2.0, step=0.05)
                    ts_max_tok = gr.Slider(label="最大令牌数", value=512, minimum=32, maximum=4096, step=32)
                    ts_top_p = gr.Slider(label="Top-P", value=1.0, minimum=0.1, maximum=1.0, step=0.05)

                with gr.Row():
                    ts_run_ragas = gr.Checkbox(
                        label="运行 RAGAS 评估指标（耗时较长）",
                        value=False,
                        info="需要 LLM 额外调用，20 条约需数分钟",
                    )

                ts_btn = gr.Button("开始评测", variant="primary")

                ts_status = gr.Textbox(label="评测状态", interactive=False)

                gr.Markdown("### 逐题结果")
                ts_table = gr.Dataframe(
                    headers=["ID", "问题", "标准答案", "模型回答", "类型", "难度", "检索数", "耗时(ms)", "状态"],
                    label="逐题对比",
                    wrap=True,
                    column_widths=[60, 160, 200, 200, 60, 60, 60, 70, 60],
                )

                gr.Markdown("### RAGAS 评估指标")
                ts_metrics = gr.JSON(label="RAGAS Metrics")

                gr.Markdown("### 完整详情（JSON）")
                ts_detail = gr.JSON(label="全部逐题详情（含完整文本）")

                def _truncate(s: str, limit: int = 100) -> str:
                    return s[:limit] + "…" if len(s) > limit else s

                def _run_testset(path, s_type, k, st, vw, bw, rerank, temp, m_toks, tp, do_ragas):
                    try:
                        res = testset_eval_handler(
                            testset_path=path,
                            search_type=s_type,
                            top_k=int(k),
                            score_threshold=float(st),
                            vector_weight=float(vw),
                            bm25_weight=float(bw),
                            reranker_enabled=bool(rerank),
                            temperature=float(temp),
                            max_tokens=int(m_toks),
                            top_p=float(tp),
                            run_ragas=bool(do_ragas),
                        )
                        status_msg = (
                            f"评测完成！共 {res['total_questions']} 题，"
                            f"成功 {res['successful']}，失败 {res['failed']}，"
                            f"总耗时 {res['total_elapsed_ms']}ms"
                        )
                        rows = [
                            [
                                r["id"],
                                _truncate(r["question"], 80),
                                _truncate(r["gold_answer"], 120),
                                _truncate(r["model_answer"], 120),
                                r["question_type"],
                                r["difficulty"],
                                r["contexts_count"],
                                r["elapsed_ms"],
                                r["status"],
                            ]
                            for r in res["per_question"]
                        ]
                        metrics = res.get("ragas_metrics", {}) or {}
                        detail = res["per_question"]
                        return status_msg, rows, metrics, detail
                    except Exception as exc:
                        return f"评测失败: {exc}", [], {}, []

                ts_btn.click(
                    _run_testset,
                    inputs=[
                        ts_path, ts_search_type, ts_top_k, ts_threshold,
                        ts_vw, ts_bw, ts_reranker,
                        ts_temp, ts_max_tok, ts_top_p, ts_run_ragas,
                    ],
                    outputs=[ts_status, ts_table, ts_metrics, ts_detail],
                )

        # ── Tab 4: 自定义评估 ──────────────────────────────
        with gr.Tab("自定义评估"):
            gr.Markdown("按 JSON 输入评估样本数组，每个元素包含 question / answer / ground_truth / contexts。")
            eval_input = gr.Textbox(label="评估样本（JSON）", lines=12)
            save_report = gr.Checkbox(label="导出评估报告（JSON/CSV）", value=False)
            report_dir = gr.Textbox(label="报告目录", value="reports")
            eval_btn = gr.Button("执行评估", variant="primary")
            eval_output = gr.JSON(label="评估指标")

            def _evaluate(payload: str, save: bool, out_dir: str):
                try:
                    rows = json.loads(payload)
                    samples = [EvalSample(**row) for row in rows]
                    req = EvalRequest(samples=samples, save_report=save, report_dir=out_dir)
                    return eval_handler(req)
                except Exception as exc:
                    return {"status": "error", "reason": str(exc)}

            eval_btn.click(_evaluate, inputs=[eval_input, save_report, report_dir], outputs=[eval_output])

    return demo
