from __future__ import annotations

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.errors import GenerationError


SYSTEM_PROMPT = """你是一个严谨的RAG问答助手。
请根据提供的上下文回答问题：
1. 仅使用上下文中的事实，不要臆造。
2. 如果上下文不足，请明确说明"根据现有资料无法确定"。
3. 回答尽量简洁，并优先使用中文。
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

    def generate(
        self,
        question: str,
        contexts: list[str],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        model: str | None = None,
    ) -> str:
        context_block = "\n\n---\n\n".join(contexts) if contexts else "无可用上下文。"
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
