from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.errors import GenerationError


SYSTEM_PROMPT = """你是一个严谨的 RAG 问答助手，必须严格遵守以下规则：
1. 仅根据提供的「上下文」作答，不得使用上下文之外的任何知识或臆造内容。
2. 若上下文为空、或与问题无关、或不足以回答问题，你必须仅回复：「根据知识库现有资料无法回答该问题，请仅就知识库内已有内容提问。」
3. 禁止回答与知识库无关的通用问题（如闲聊、编程、时事等），该类问题一律回复上述固定话术。
4. 回答尽量简洁，优先使用中文。不得以「作为 AI」等口吻扩展或补充知识库外信息。
"""


class LLMGenerator:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "问题：{question}\n\n上下文：\n{context}\n\n请输出最终答案。"),
            ]
        )
        self._default_llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self._parser = StrOutputParser()

    @lru_cache(maxsize=16)
    def _get_llm(self, model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            timeout=self._settings.llm_timeout,
            max_retries=self._settings.llm_max_retries,
        )

    # 无上下文时的固定回复，避免调用 API 被滥用
    NO_CONTEXT_REPLY = "知识库中暂无与您问题相关的内容，请仅就知识库内已有资料提问。"

    def generate(
        self,
        question: str,
        contexts: list[str],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        model: str | None = None,
    ) -> str:
        if not contexts:
            return self.NO_CONTEXT_REPLY
        context_block = "\n\n---\n\n".join(contexts)
        base_llm = self._get_llm(model) if model else self._default_llm
        llm = base_llm.bind(
            temperature=temperature if temperature is not None else self._settings.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._settings.default_max_tokens,
            top_p=top_p if top_p is not None else self._settings.default_top_p,
        )
        chain = self._prompt | llm | self._parser
        try:
            return chain.invoke({"question": question, "context": context_block})
        except Exception as exc:
            raise GenerationError(f"Generation failed: {exc}") from exc
