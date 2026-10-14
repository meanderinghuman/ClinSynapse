"""MedCPT encoders and cross-encoder (Jin et al., 2023), trained on PubMed search logs."""

import threading
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"

# One lock for every model on the accelerator (MedCPT and the NLI judge): concurrent use of the
# MPS device from several threads crashes Metal ("command encoder is already encoding").
ACCELERATOR_LOCK = threading.RLock()


def _device(preferred: str) -> str:
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _locked(fn):
    def inner(*args, **kwargs):
        with ACCELERATOR_LOCK:
            return fn(*args, **kwargs)

    return inner


@lru_cache
def _load(name: str, device: str, cross: bool = False):
    tok = AutoTokenizer.from_pretrained(name)
    cls = AutoModelForSequenceClassification if cross else AutoModel
    dev = _device(device)
    model = cls.from_pretrained(name).to(dev).eval()
    if dev != "cpu":
        model = model.half()  # ~2x faster on MPS/CUDA; CLS embeddings are cast back to fp32
    return tok, model


@_locked
@torch.no_grad()
def encode_queries(queries: list[str], device: str = "mps") -> np.ndarray:
    tok, model = _load(QUERY_ENCODER, device)
    enc = tok(queries, truncation=True, padding=True, max_length=64, return_tensors="pt")
    out = model(**enc.to(model.device)).last_hidden_state[:, 0, :]
    return out.float().cpu().numpy()


@_locked
@torch.no_grad()
def encode_articles(
    titles: list[str], texts: list[str], device: str = "mps", batch_size: int = 64
) -> np.ndarray:
    tok, model = _load(ARTICLE_ENCODER, device)
    chunks = []
    for i in range(0, len(texts), batch_size):
        pairs = [[t, a] for t, a in zip(titles[i : i + batch_size], texts[i : i + batch_size])]
        enc = tok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt")
        chunks.append(model(**enc.to(model.device)).last_hidden_state[:, 0, :].float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 768), dtype=np.float32)


@_locked
@torch.no_grad()
def cross_scores(query: str, texts: list[str], device: str = "mps", batch_size: int = 16) -> list[float]:
    tok, model = _load(CROSS_ENCODER, device, cross=True)
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        pairs = [[query, t] for t in texts[i : i + batch_size]]
        enc = tok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt")
        scores.extend(model(**enc.to(model.device)).logits.squeeze(-1).float().cpu().tolist())
    return scores
