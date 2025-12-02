import json, torch
from sentence_transformers import SentenceTransformer, util

class RagPipeline:
    def __init__(self, data_path="data/qa_dataset.json", index_path="data/e5_index.pt", device=None):
        self.qa = json.load(open(data_path, encoding="utf-8"))
        cache = torch.load(index_path, map_location="cpu")
        self.model = SentenceTransformer(cache["model_name"], device=device)
        self.emb = cache["embeddings"]
        if device:
            self.emb = self.emb.to(device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def retrieve(self, query, top_k=3):
        q = self.model.encode(f"query: {query}", convert_to_tensor=True, normalize_embeddings=True)
        hits = util.semantic_search(q, self.emb, top_k=top_k)[0]
        out = []
        for h in hits:
            item = self.qa[h["corpus_id"]]
            out.append({
                "score": float(h["score"]),
                "id": item.get("id", h["corpus_id"]),
                "question": item["question"],
                "answer": item["answer"],
            })
        return out
