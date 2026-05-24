"""Multilingual sentiment classifier wrapper.

Uses cardiffnlp/twitter-xlm-roberta-base-sentiment by default.
Loads lazily so that pure-SQL queries don't pay the import cost.
"""
from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm


LABELS = ["negative", "neutral", "positive"]


@dataclass
class SentimentPrediction:
    label: str
    score: float        # confidence of predicted label, 0..1
    signed: float       # -score / 0 / +score for trajectory math


class SentimentClassifier:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentPrediction]:
        results: list[SentimentPrediction] = []
        total = len(texts)
        pbar = tqdm(range(0, total, batch_size), desc="sentiment", unit="batch")
        for i in pbar:
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                idx = int(row.argmax())
                label = LABELS[idx]
                score = float(row[idx])
                signed = -score if label == "negative" else (score if label == "positive" else 0.0)
                results.append(SentimentPrediction(label=label, score=score, signed=signed))
            pbar.set_postfix(rows=f"{min(i + batch_size, total)}/{total}")
        return results
