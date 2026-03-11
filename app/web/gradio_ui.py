from __future__ import annotations

import csv
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from app.schemas.rag import EvalRequest, EvalSample, IngestRequest, QueryRequest

logger = logging.getLogger(__name__)

# ── 常量 ────────────────────────────────────────────────────

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

MODEL_CHOICES = [
    ("Kimi K2.5", "kimi-k2.5"),
    ("MiniMax M2.5", "MiniMax/MiniMax-M2.5"),
    ("GLM-5", "glm-5"),
    ("Qwen 3.5 Plus", "qwen3.5-plus"),
    ("Qwen 3.5 Flash", "qwen3.5-flash"),
    ("自定义模型", "__custom__"),
]

SEARCH_TYPE_CHOICES = [
    ("向量相似度", "similarity"),
    ("关键词检索（BM25）", "bm25"),
    ("混合检索", "hybrid"),
    ("MMR 多样性检索", "mmr"),
]
TEMP_FILE_KEEP_SECONDS = 24 * 60 * 60


# ── 辅助函数 ────────────────────────────────────────────────

def _resolve_model(dropdown_val: str, custom_val: str) -> str | None:
    """从下拉选择和自定义输入框中解析出最终模型名。"""
    if dropdown_val == "__custom__":
        return custom_val.strip() or None
    return dropdown_val or None


def _build_ragas_chart(metrics: dict[str, float]) -> Any:
    """用 matplotlib 绘制 RAGAS 柱状图 + 雷达图，返回 figure。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    label_map = {
        "context_precision": "上下文精度",
        "context_recall": "上下文召回率",
        "faithfulness": "忠实度",
        "answer_relevancy": "答案相关性",
        "retrieval_f1": "检索F1",
    }
    display_keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "retrieval_f1"]
    labels = []
    values = []
    for k in display_keys:
        if k in metrics:
            labels.append(label_map.get(k, k))
            values.append(metrics[k])

    if not values:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

    ax1 = axes[0]
    bars = ax1.bar(labels, values, color=colors[: len(labels)], width=0.6, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("分数")
    ax1.set_title("RAGAS 评估指标")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    ax2 = fig.add_subplot(122, polar=True)
    axes[1].set_visible(False)
    ax2.plot(angles_closed, values_closed, "o-", linewidth=2, color="#4E79A7")
    ax2.fill(angles_closed, values_closed, alpha=0.25, color="#4E79A7")
    ax2.set_thetagrids(np.degrees(angles), labels)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("雷达图", pad=20)

    fig.tight_layout(pad=3.0)
    return fig


def _cleanup_temp_files(prefix: str, suffix: str) -> None:
    """清理过期临时导出文件，避免长期运行时堆积。"""
    now = datetime.now().timestamp()
    tmp_dir = Path(tempfile.gettempdir())
    for file in tmp_dir.glob(f"{prefix}*{suffix}"):
        try:
            if now - file.stat().st_mtime > TEMP_FILE_KEEP_SECONDS:
                file.unlink(missing_ok=True)
        except OSError:
            logger.warning("清理临时文件失败: %s", file, exc_info=True)


def _save_chart_to_file(fig: Any) -> str | None:
    """将 matplotlib figure 保存为临时 PNG 文件，返回路径。"""
    if fig is None:
        return None
    try:
        _cleanup_temp_files("ragas_chart_", ".png")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="ragas_chart_")
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        return tmp.name
    except Exception:
        logger.warning("保存图表文件失败", exc_info=True)
        return None


def _save_json_to_file(data: Any, prefix: str = "export") -> str | None:
    """将数据保存为临时 JSON 文件。"""
    try:
        _cleanup_temp_files(f"{prefix}_", ".json")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, prefix=f"{prefix}_", mode="w", encoding="utf-8",
        )
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        return tmp.name
    except Exception:
        logger.warning("保存 JSON 文件失败", exc_info=True)
        return None


def _save_csv_to_file(rows: list[dict], prefix: str = "export") -> str | None:
    """将 dict 列表保存为临时 CSV 文件。"""
    if not rows:
        return None
    try:
        _cleanup_temp_files(f"{prefix}_", ".csv")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix=f"{prefix}_", mode="w", encoding="utf-8-sig", newline="",
        )
        writer = csv.DictWriter(tmp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        tmp.close()
        return tmp.name
    except Exception:
        logger.warning("保存 CSV 文件失败", exc_info=True)
        return None


# ── UI 构建 ─────────────────────────────────────────────────

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
                    info="成本：pypdf 无额外成本；MinerU 需本地算力，入库略慢。效率：MinerU 解析更精准、表格/公式保留更好，检索质量更高。",
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
                    info="成本：块越大，单次检索上下文越多，可能增加 LLM token 消耗。效率：800 左右兼顾语义完整与检索精度；过大易含噪声，过小易断句。",
                )
                chunk_overlap = gr.Slider(
                    label="分块重叠",
                    value=120,
                    minimum=0,
                    maximum=1000,
                    step=10,
                    info="成本：重叠多则块数略增，向量存储与检索略增。效率：约 15% 重叠可减少边界断句，提升召回；过大重复多、冗余增加。",
                )

            separator_checkboxes = gr.CheckboxGroup(
                label="分块分隔符（按优先级排列，从上到下依次尝试）",
                choices=SEPARATOR_LABELS,
                value=DEFAULT_SEPARATOR_LABELS,
                info="成本：无额外成本。效率：分隔符决定切分质量，双换行/句号等利于保持语义完整，从而提升检索与回答质量。",
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
            gr.Markdown("### 模型选择")
            with gr.Row():
                qa_model = gr.Dropdown(
                    label="大模型",
                    choices=MODEL_CHOICES,
                    value="kimi-k2.5",
                    info="成本：不同厂商/型号按 token 计费差异大，Plus 类通常比 Flash 贵、质量更好。效率：大模型响应速度与并发能力不同，影响首 token 与总耗时。",
                )
                qa_custom_model = gr.Textbox(
                    label="自定义模型名称",
                    placeholder="输入百炼平台模型 API 名称",
                    visible=False,
                )

            def _on_qa_model_change(val):
                return gr.update(visible=(val == "__custom__"))

            qa_model.change(_on_qa_model_change, inputs=[qa_model], outputs=[qa_custom_model])

            question = gr.Textbox(label="问题", placeholder="请输入问题", lines=3)

            gr.Markdown("### 检索设置")
            with gr.Row():
                top_k = gr.Slider(
                    label="检索数量（Top-K）",
                    value=4,
                    minimum=1,
                    maximum=20,
                    step=1,
                    info="成本：K 越大送入 LLM 的上下文越多，token 消耗与费用上升。效率：K 大召回更全但延迟与噪音增加；4–8 为常用折中。",
                )
                score_threshold = gr.Slider(
                    label="相似度阈值",
                    value=0.25,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    info="成本：阈值越高过滤越多，实际送入 LLM 的 token 越少，成本与耗时降低。效率：过高易漏检，过低易带入无关片段影响答案质量。",
                )
                search_type = gr.Dropdown(
                    label="检索方式",
                    choices=SEARCH_TYPE_CHOICES,
                    value="similarity",
                    info="成本：向量≈BM25＜混合＜MMR；混合/MMR 计算更多。效率：混合兼顾语义+关键词；MMR 多样性好；纯向量适合通用语义问句。",
                )

            with gr.Row(visible=False) as weight_row:
                vector_weight = gr.Slider(
                    label="向量权重",
                    value=0.7,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="成本：无直接变化。效率：向量高偏语义理解，BM25 高偏精确词匹配；0.7/0.3 为常见平衡，可按业务调。",
                )
                bm25_weight = gr.Slider(
                    label="关键词权重",
                    value=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="成本：无直接变化。效率：与向量权重互补；术语、代码、专有名词多时可适当提高。",
                )

            def _on_search_type_change(s_type):
                return gr.update(visible=s_type == "hybrid")

            search_type.change(_on_search_type_change, inputs=[search_type], outputs=[weight_row])

            def _on_vector_weight_change(vw):
                return round(1.0 - vw, 2)

            vector_weight.change(_on_vector_weight_change, inputs=[vector_weight], outputs=[bm25_weight])

            with gr.Row():
                reranker_enabled = gr.Checkbox(
                    label="启用重排序",
                    value=False,
                    info="成本：需多取候选再精排，检索阶段耗时与算力增加；不增加 LLM 费用。效率：显著提升 Top-K 内排序质量，适合对准确率要求高的场景。",
                )

            gr.Markdown("### 生成设置")
            with gr.Row():
                temperature = gr.Slider(
                    label="温度（Temperature）",
                    value=0.2,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    info="成本：不直接改变计费。效率：低温度回答更稳定、更贴上下文，RAG 推荐 0.1–0.3；高温度创意多但易偏题。",
                )
                max_tokens = gr.Slider(
                    label="最大令牌数",
                    value=512,
                    minimum=32,
                    maximum=4096,
                    step=32,
                    info="成本：按输出 token 计费，上限越大单次回答可能越贵。效率：设足可避免截断，过大则冗长且耗时长；512 适合多数问答。",
                )
                top_p = gr.Slider(
                    label="采样阈值（Top-P）",
                    value=1.0,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    info="成本：不直接改变计费。效率：1.0 不截断词表；降低可减少随机性、回答更集中，与低 temperature 类似。",
                )

            ask_btn = gr.Button("生成回答", variant="primary")
            answer = gr.Textbox(label="回答", lines=8)
            citations = gr.JSON(label="引用片段")
            with gr.Row():
                latency = gr.Textbox(label="耗时（毫秒）")
                cache_tag = gr.Textbox(label="缓存命中")

            def _ask(q, k, st, s_type, vw, bw, rerank, temp, m_toks, tp, model_dd, model_custom):
                try:
                    model = _resolve_model(model_dd, model_custom)
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
                        model=model,
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
                    qa_model, qa_custom_model,
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

                gr.Markdown("### 模型选择")
                with gr.Row():
                    ts_model = gr.Dropdown(
                        label="大模型",
                        choices=MODEL_CHOICES,
                        value="kimi-k2.5",
                        info="成本：不同厂商/型号按 token 计费差异大。效率：影响每条题的响应速度与评测总耗时。",
                    )
                    ts_custom_model = gr.Textbox(
                        label="自定义模型名称",
                        placeholder="输入百炼平台模型 API 名称",
                        visible=False,
                    )

                ts_model.change(
                    lambda v: gr.update(visible=(v == "__custom__")),
                    inputs=[ts_model], outputs=[ts_custom_model],
                )

                ts_path = gr.Textbox(
                    label="评测集路径",
                    value="data/rag_eval_set.json",
                    info="默认使用项目自带测试集，也可填入其他 JSON 文件路径",
                )

                gr.Markdown("### 检索设置")
                with gr.Row():
                    ts_top_k = gr.Slider(
                        label="检索数量（Top-K）",
                        value=4,
                        minimum=1,
                        maximum=20,
                        step=1,
                        info="成本：K 越大单题 token 越多，批量评测总成本上升。效率：影响每题检索与生成耗时。",
                    )
                    ts_threshold = gr.Slider(
                        label="相似度阈值",
                        value=0.25,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        info="成本：阈值高则送入 LLM 的上下文少，单题与总成本降低。效率：过高易漏检，影响评测得分。",
                    )
                    ts_search_type = gr.Dropdown(
                        label="检索方式",
                        choices=SEARCH_TYPE_CHOICES,
                        value="similarity",
                        info="成本：向量＜混合＜MMR。效率：混合/MMR 检索更耗时但可能提升召回与评测指标。",
                    )

                with gr.Row(visible=False) as ts_weight_row:
                    ts_vw = gr.Slider(
                        label="向量权重",
                        value=0.7,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="成本：无直接变化。效率：与关键词权重互补，影响检索排序与评测表现。",
                    )
                    ts_bw = gr.Slider(
                        label="关键词权重",
                        value=0.3,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="成本：无直接变化。效率：术语多时可适当提高，改善检索与答案质量。",
                    )

                ts_search_type.change(
                    lambda s: gr.update(visible=s == "hybrid"),
                    inputs=[ts_search_type], outputs=[ts_weight_row],
                )
                ts_vw.change(lambda vw: round(1.0 - vw, 2), inputs=[ts_vw], outputs=[ts_bw])

                with gr.Row():
                    ts_reranker = gr.Checkbox(
                        label="启用重排序",
                        value=False,
                        info="成本：检索阶段耗时增加，不增加 LLM 费用。效率：提升排序质量，评测时更易反映真实能力。",
                    )

                gr.Markdown("### 生成设置")
                with gr.Row():
                    ts_temp = gr.Slider(
                        label="温度",
                        value=0.2,
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        info="成本：不直接改变计费。效率：低温度结果更稳定，便于评测对比。",
                    )
                    ts_max_tok = gr.Slider(
                        label="最大令牌数",
                        value=512,
                        minimum=32,
                        maximum=4096,
                        step=32,
                        info="成本：上限越大单题可能越贵，批量评测总成本上升。效率：足够即可，过大增加耗时。",
                    )
                    ts_top_p = gr.Slider(
                        label="Top-P",
                        value=1.0,
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        info="成本：不直接改变计费。效率：与温度配合，影响回答稳定性与评测一致性。",
                    )

                with gr.Row():
                    ts_run_ragas = gr.Checkbox(
                        label="运行 RAGAS 评估指标（耗时较长）",
                        value=False,
                        info="成本：RAGAS 需额外 LLM 调用（生成与评估），显著增加 token 消耗与费用。效率：20 条约需数分钟，适合做深度质量评估时开启。",
                    )

                gr.Markdown("### 指标目标（RAGAS 得分低于目标时将给出优化建议）")
                with gr.Row():
                    ts_target_faithfulness = gr.Number(
                        label="Faithfulness 目标",
                        value=0.85,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="忠实度：回答是否严格基于上下文",
                    )
                    ts_target_answer_relevancy = gr.Number(
                        label="Answer Relevancy 目标",
                        value=0.8,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="答案相关性：回答与问题的匹配度",
                    )
                    ts_target_context_recall = gr.Number(
                        label="Context Recall 目标",
                        value=0.8,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="上下文召回：相关文档是否被检索到",
                    )
                    ts_target_context_precision = gr.Number(
                        label="Context Precision 目标",
                        value=0.7,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        info="上下文精度：检索片段中相关内容的占比",
                    )

                ts_btn = gr.Button("开始评测", variant="primary")

                ts_status = gr.Textbox(label="评测状态", interactive=False)

                # ── 参数快照 ──
                gr.Markdown("### 本次评测参数")
                ts_params_json = gr.JSON(label="参数快照")
                ts_params_file = gr.File(label="下载参数快照（JSON）", interactive=False)

                # ── 逐题结果 ──
                gr.Markdown("### 逐题结果")
                ts_table = gr.Dataframe(
                    headers=["ID", "问题", "标准答案", "模型回答", "类型", "难度", "检索数", "耗时(ms)", "状态"],
                    label="逐题对比（点击任意行查看完整内容）",
                    wrap=True,
                    column_widths=[60, 160, 200, 200, 60, 60, 60, 70, 60],
                )

                # ── 逐题完整预览 ──
                ts_full_data = gr.State([])
                gr.Markdown("### 选中题目详情预览")
                with gr.Row():
                    detail_question = gr.Textbox(label="问题", interactive=False, lines=3)
                with gr.Row():
                    detail_gold = gr.Textbox(label="标准答案", interactive=False, lines=8)
                    detail_model = gr.Textbox(label="模型回答", interactive=False, lines=8)
                with gr.Row():
                    detail_meta = gr.Textbox(label="元信息（类型 / 难度 / 检索数 / 耗时 / 状态）", interactive=False)

                def _on_row_select(evt: gr.SelectData, full_data):
                    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    if not full_data or idx >= len(full_data):
                        return "", "", "", ""
                    r = full_data[idx]
                    meta = f"类型: {r.get('question_type', '')} | 难度: {r.get('difficulty', '')} | 检索数: {r.get('contexts_count', '')} | 耗时: {r.get('elapsed_ms', '')}ms | 状态: {r.get('status', '')}"
                    return r.get("question", ""), r.get("gold_answer", ""), r.get("model_answer", ""), meta

                ts_table.select(
                    _on_row_select,
                    inputs=[ts_full_data],
                    outputs=[detail_question, detail_gold, detail_model, detail_meta],
                )

                # ── RAGAS 评估指标 ──
                gr.Markdown("### RAGAS 评估指标")
                ts_metrics = gr.JSON(label="RAGAS Metrics")
                ts_ragas_plot = gr.Plot(label="RAGAS 可视化图表")
                gr.Markdown("### 优化方向建议")
                ts_ragas_suggestions = gr.Markdown(
                    value="完成 RAGAS 评测并勾选「运行 RAGAS 评估指标」后，将根据各维度得分给出优化建议。",
                )
                with gr.Row():
                    ts_chart_file = gr.File(label="下载 RAGAS 图表（PNG）", interactive=False)
                    ts_ragas_detail_file = gr.File(label="下载 RAGAS 完整详情（JSON）", interactive=False)

                # ── 完整详情 ──
                gr.Markdown("### 完整详情（JSON）")
                ts_detail = gr.JSON(label="全部逐题详情（含完整文本）")

                def _truncate(s: str, limit: int = 100) -> str:
                    return s[:limit] + "…" if len(s) > limit else s

                def _run_testset(
                    path, s_type, k, st, vw, bw, rerank, temp, m_toks, tp, do_ragas,
                    t_fa, t_ar, t_cr, t_cp,
                    c_size, c_overlap, sep_labels,
                    model_dd, model_custom,
                ):
                    try:
                        model = _resolve_model(model_dd, model_custom)
                        target_fa = float(t_fa) if t_fa is not None else 0.85
                        target_ar = float(t_ar) if t_ar is not None else 0.8
                        target_cr = float(t_cr) if t_cr is not None else 0.8
                        target_cp = float(t_cp) if t_cp is not None else 0.7

                        # --- 构建参数快照 ---
                        seps_display = sep_labels if sep_labels else []
                        search_type_display = dict(SEARCH_TYPE_CHOICES).get(s_type, s_type) if isinstance(s_type, str) else s_type
                        params_snapshot = {
                            "模型": model or "(默认)",
                            "分块大小": int(c_size),
                            "分块重叠": int(c_overlap),
                            "分块分隔符": seps_display,
                            "Top-K": int(k),
                            "相似度阈值": float(st),
                            "检索方式": s_type,
                            "向量权重": float(vw),
                            "关键词权重": float(bw),
                            "启用重排序": "是" if rerank else "否",
                            "温度": float(temp),
                            "最大令牌数": int(m_toks),
                            "Top-P": float(tp),
                            "运行RAGAS": "是" if do_ragas else "否",
                            "指标目标": {"Faithfulness": target_fa, "Answer Relevancy": target_ar, "Context Recall": target_cr, "Context Precision": target_cp},
                            "评测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        params_file = _save_json_to_file(params_snapshot, prefix="params")

                        # --- 执行评测 ---
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
                            model=model,
                            ragas_target_faithfulness=target_fa,
                            ragas_target_answer_relevancy=target_ar,
                            ragas_target_context_recall=target_cr,
                            ragas_target_context_precision=target_cp,
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
                        full_data = res["per_question"]

                        # --- RAGAS 指标处理 ---
                        raw_ragas = res.get("ragas_metrics", {}) or {}
                        metrics_display = {}
                        chart_fig = None
                        chart_path = None
                        ragas_detail_path = None

                        if "summary" in raw_ragas:
                            metrics_display = raw_ragas["summary"]
                            chart_fig = _build_ragas_chart(metrics_display)
                            chart_path = _save_chart_to_file(chart_fig)
                            ragas_detail_path = _save_json_to_file(raw_ragas, prefix="ragas_detail")
                        elif "status" not in raw_ragas and raw_ragas:
                            metrics_display = raw_ragas
                        else:
                            metrics_display = raw_ragas

                        suggestions_md = "完成 RAGAS 评测并勾选「运行 RAGAS 评估指标」后，将根据各维度得分给出优化建议。"
                        if "optimization_suggestions" in raw_ragas:
                            sugg = raw_ragas["optimization_suggestions"]
                            suggestions_md = "#### 优化方向建议\n\n" + "\n\n".join(sugg)

                        detail = res["per_question"]

                        return (
                            status_msg,
                            params_snapshot,
                            params_file,
                            rows,
                            full_data,
                            metrics_display,
                            chart_fig,
                            chart_path,
                            ragas_detail_path,
                            suggestions_md,
                            detail,
                        )
                    except Exception as exc:
                        return (
                            f"评测失败: {exc}",
                            {},
                            None,
                            [],
                            [],
                            {},
                            None,
                            None,
                            None,
                            "",
                            [],
                        )

                ts_btn.click(
                    _run_testset,
                    inputs=[
                        ts_path, ts_search_type, ts_top_k, ts_threshold,
                        ts_vw, ts_bw, ts_reranker,
                        ts_temp, ts_max_tok, ts_top_p, ts_run_ragas,
                        ts_target_faithfulness, ts_target_answer_relevancy,
                        ts_target_context_recall, ts_target_context_precision,
                        chunk_size, chunk_overlap, separator_checkboxes,
                        ts_model, ts_custom_model,
                    ],
                    outputs=[
                        ts_status,
                        ts_params_json,
                        ts_params_file,
                        ts_table,
                        ts_full_data,
                        ts_metrics,
                        ts_ragas_plot,
                        ts_chart_file,
                        ts_ragas_detail_file,
                        ts_ragas_suggestions,
                        ts_detail,
                    ],
                )

        # ── Tab 4: 自定义评估 ──────────────────────────────
        with gr.Tab("自定义评估"):
            gr.Markdown("按 JSON 输入评估样本数组，每个元素包含 question / answer / ground_truth / contexts。")
            eval_input = gr.Textbox(label="评估样本（JSON）", lines=12)
            save_report = gr.Checkbox(label="导出评估报告（JSON/CSV）", value=False)
            report_dir = gr.Textbox(label="报告目录（相对 REPORTS_DIR）", value="")
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
