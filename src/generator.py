from llama_cpp import Llama
from huggingface_hub import hf_hub_download


class QwenGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct-GGUF", n_ctx=2048, n_threads=4):
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def generate(self, query, context="", max_tokens=256, temperature=0.7, top_p=0.9):
        if context:
            prompt = f"""<|im_start|>system
You are a mental health assistant. Provide 4-6 clear, actionable points. Be specific and encouraging.<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
You are a mental health assistant. Provide 4-6 clear, actionable points. Be specific and encouraging.<|im_end|>
<|im_start|>user
Question: {query}<|im_end|>
<|im_start|>assistant
"""

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>"]
        )
        return output["choices"][0]["text"].strip()
