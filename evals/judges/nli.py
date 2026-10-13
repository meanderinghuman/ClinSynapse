"""NLI judge: premise = cited abstract(s), hypothesis = claim; supported if P(entailment) >= threshold."""

import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinsynapse.retrieval.medcpt import ACCELERATOR_LOCK, _device

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


class NLIJudge:
    name = "nli"

    def __init__(self, model: str = MODEL, threshold: float = 0.5, device: str = "mps"):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.dev = _device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.dev).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]
        self.threshold = threshold

    @torch.no_grad()
    def entailment_prob(self, claim: str, evidence: str) -> float:
        enc = self.tok(evidence, claim, truncation="only_first", max_length=512, return_tensors="pt")
        with ACCELERATOR_LOCK:
            logits = self.model(**enc.to(self.dev)).logits[0]
        return float(torch.softmax(logits.float(), -1)[self.entail])

    def support_prob(self, claim: str, evidence: str) -> float:
        """Max entailment over the whole evidence and 3-sentence windows (SummaC-style).

        Whole-abstract premises miss claims stated in one sentence; windows miss claims that
        combine sentences. Taking the max keeps both.
        """
        sents = [s for s in _SPLIT.split(evidence) if s.strip()]
        windows = [" ".join(sents[i : i + 3]) for i in range(max(1, len(sents) - 2))]
        return max(self.entailment_prob(claim, p) for p in [evidence, *windows])

    def supported(self, claim: str, evidence: str) -> bool:
        return self.support_prob(claim, evidence) >= self.threshold
