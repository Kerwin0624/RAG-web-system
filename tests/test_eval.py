from app.eval.dataset_builder import EvalDatasetBuilder
from app.eval.ragas_eval import RagasEvaluator
from app.schemas.rag import EvalSample


def test_dataset_builder():
    sample = EvalSample(
        question="什么是RAG？",
        answer="RAG 是检索增强生成。",
        ground_truth="RAG 是一种结合检索和生成的方法。",
        contexts=["RAG 将检索与生成结合。"],
    )
    dataset = EvalDatasetBuilder.build([sample])
    assert len(dataset) == 1


def test_ragas_evaluator_returns_dict():
    sample = EvalSample(
        question="什么是RAG？",
        answer="RAG 是检索增强生成。",
        ground_truth="RAG 是一种结合检索和生成的方法。",
        contexts=["RAG 将检索与生成结合。"],
    )
    evaluator = RagasEvaluator()
    result = evaluator.evaluate_samples([sample])
    assert isinstance(result, dict)
