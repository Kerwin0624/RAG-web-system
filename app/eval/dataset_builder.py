from __future__ import annotations

from datasets import Dataset

from app.schemas.rag import EvalSample


class EvalDatasetBuilder:
    @staticmethod
    def build(samples: list[EvalSample]) -> Dataset:
        records = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        return Dataset.from_dict(records)
