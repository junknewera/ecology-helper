"""Генератор ответов на базе квантованной модели Qwen."""

from typing import Any, Dict

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class QwenGenerator:
    """Генерирует ответы на основе квантованной LLM Qwen 0.5B."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        n_ctx: int = 2048,
        n_threads: int = 4,
    ) -> None:
        """
        Инициализирует генератор с квантованной моделью.

        Args:
            model_name: HuggingFace репозиторий с моделью
            n_ctx: Размер контекстного окна (количество токенов)
            n_threads: Количество потоков CPU для инференса
        """
        # Скачиваем квантованную модель (Q4_K_M = 4-bit)
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            resume_download=True,
        )

        # Загружаем модель через llama.cpp (работает на CPU)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(
        self,
        query: str,
        context: str = "",
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Генерирует ответ на запрос с опциональным контекстом.

        Args:
            query: Вопрос пользователя
            context: Релевантные документы из RAG (опционально)
            max_tokens: Максимальная длина ответа
            temperature: Креативность (0 = детерминировано, 1 = случайно)
            top_p: Nucleus sampling параметр

        Returns:
            Сгенерированный текст ответа
        """
        # Формируем промпт в формате Qwen (ChatML)
        if context:
            # RAG режим: используем найденные документы как контекст
            system_msg = (
                "You are a mental health assistant. "
                "Provide 4-6 clear, actionable points. "
                "Be specific and encouraging."
            )
            prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        else:
            # Baseline режим: генерация без контекста
            system_msg = (
                "You are a mental health assistant. "
                "Provide 4-6 clear, actionable points. "
                "Be specific and encouraging."
            )
            prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
Question: {query}<|im_end|>
<|im_start|>assistant
"""

        # Генерируем ответ
        output: Dict[str, Any] = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>"],  # Останавливаемся на специальном токене
        )

        return output["choices"][0]["text"].strip()
