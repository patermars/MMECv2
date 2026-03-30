import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "ProsusAI/finbert"
_tokenizer = None
_model = None


def _get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
    return _tokenizer, _model


@torch.no_grad()
def encode_utterance(text: str, device: str = "cuda") -> torch.Tensor:
    tokenizer, model = _get_model_and_tokenizer()
    model.to(device).eval()
    tokens  = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=512).to(device)
    outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()


@torch.no_grad()
def encode_batch(texts: list[str], device: str = "cuda",
                 batch_size: int = 32) -> np.ndarray:
    tokenizer, model = _get_model_and_tokenizer()
    model.to(device).eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt",
                          truncation=True, max_length=512,
                          padding=True).to(device)
        outputs = model(**tokens)
        cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embs)

    return np.concatenate(all_embeddings, axis=0)
