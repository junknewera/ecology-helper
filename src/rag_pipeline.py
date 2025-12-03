"""RAG pipeline для семантического поиска по базе знаний."""

import json
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, util


class RagPipeline:
    """Извлекает релевантные документы из базы знаний по запросу."""

    def __init__(
        self,
        data_path: str = "data/qa_dataset.json",
        index_path: str = "data/e5_index.pt",
        device: Optional[str] = None,
    ) -> None:
        """
        Инициализирует RAG pipeline с предвычисленными эмбеддингами.

        Args:
            data_path: Путь к JSON файлу с вопросами и ответами
            index_path: Путь к файлу с сохраненными эмбеддингами
            device: Устройство (cuda/cpu), автоопределение если None
        """
        # Загружаем базу знаний (вопросы и ответы)
        with open(data_path, encoding="utf-8") as f:
            self.qa: List[Dict[str, Any]] = json.load(f)

        # Загружаем предвычисленные эмбеддинги и модель
        cache = torch.load(index_path, map_location="cpu")
        self.model = SentenceTransformer(cache["model_name"], device=device)
        self.emb: torch.Tensor = cache["embeddings"]

        # Переносим эмбеддинги на нужное устройство
        if device:
            self.emb = self.emb.to(device)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Находит top-K наиболее релевантных документов для запроса.

        Args:
            query: Запрос пользователя
            top_k: Количество документов для возврата

        Returns:
            Список словарей с документами и метаданными
            (score, question, answer)
        """
        # Кодируем запрос с префиксом "query:" (требование модели E5)
        q = self.model.encode(
            f"query: {query}",
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        # Семантический поиск: косинусное сходство с базой эмбеддингов
        hits = util.semantic_search(q, self.emb, top_k=top_k)[0]

        # Формируем результат с полной информацией
        out = []
        for h in hits:
            item = self.qa[h["corpus_id"]]
            out.append(
                {
                    "score": float(h["score"]),  # Релевантность (0-1)
                    "id": item.get("id", h["corpus_id"]),
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
        return out
