"""Text decoding evaluation metrics."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:
    from bert_score import score as bert_score
except ImportError:  # pragma: no cover - optional dependency
    bert_score = None


def _tokenize(text: str) -> List[str]:
    return text.strip().split()


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[m, n])


def _bleu_score(decoded: str, reference: str, max_order: int = 4) -> float:
    cand = _tokenize(decoded)
    ref = _tokenize(reference)
    if not cand:
        return 0.0
    weights = [1.0 / max_order] * max_order
    precision_log = 0.0
    for n in range(1, max_order + 1):
        cand_ngrams = {}
        for i in range(len(cand) - n + 1):
            ngram = tuple(cand[i : i + n])
            cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
        ref_ngrams = {}
        for i in range(len(ref) - n + 1):
            ngram = tuple(ref[i : i + n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
        overlap = 0
        for ngram, count in cand_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))
        total = max(len(cand) - n + 1, 1)
        precision = (overlap + 1) / (total + 1)
        precision_log += weights[n - 1] * np.log(precision)
    bp = 1.0 if len(cand) > len(ref) else np.exp(1 - len(ref) / max(len(cand), 1))
    return float(bp * np.exp(precision_log))


def _meteor(decoded: str, reference: str) -> float:
    cand = _tokenize(decoded)
    ref = _tokenize(reference)
    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0
    matches = sum(1 for token in cand if token in ref)
    precision = matches / len(cand) if cand else 0.0
    recall = matches / len(ref) if ref else 0.0
    if precision + recall == 0:
        return 0.0
    fmean = (10 * precision * recall) / (recall + 9 * precision)
    frag = 0
    ref_set = set(ref)
    streak = 0
    for token in cand:
        if token in ref_set:
            streak += 1
        else:
            if streak > 0:
                frag += 1
            streak = 0
    if streak > 0:
        frag += 1
    penalty = 0.5 * (frag / matches) if matches else 0.0
    return (1 - penalty) * fmean


def eval_text_list(
    decoded: Sequence[str],
    reference: Sequence[str],
    bert_model: str = "bert-base-uncased",
) -> dict:
    """Compute aggregate text metrics for paired decoded/reference lists."""
    if len(decoded) != len(reference):
        raise ValueError("decoded and reference must be the same length")
    wers: List[float] = []
    bleus: List[float] = []
    meteors: List[float] = []
    for hyp, ref in zip(decoded, reference):
        ref_tokens = _tokenize(ref)
        hyp_tokens = _tokenize(hyp)
        if ref_tokens:
            distance = _levenshtein(ref_tokens, hyp_tokens)
            wers.append(distance / len(ref_tokens))
        else:
            wers.append(float(len(hyp_tokens) > 0))
        bleus.append(_bleu_score(hyp, ref))
        meteors.append(_meteor(hyp, ref))
    bert_f1 = np.full(len(decoded), np.nan, dtype=float)
    if bert_score is not None and decoded:
        try:
            _, _, F = bert_score(list(decoded), list(reference), model_type=bert_model)
            bert_f1 = F.detach().cpu().numpy()
        except Exception:
            pass
    return {
        "wer": float(np.nanmean(wers)) if wers else float("nan"),
        "bleu": float(np.nanmean(bleus)) if bleus else float("nan"),
        "meteor": float(np.nanmean(meteors)) if meteors else float("nan"),
        "bertscore_f1": float(np.nanmean(bert_f1)) if np.isfinite(bert_f1).any() else float("nan"),
    }


def identification_matrix(
    decoded_windows: Sequence[str],
    reference_windows: Sequence[str],
    bert_model: str = "bert-base-uncased",
) -> Tuple[np.ndarray, float]:
    """Return BERTScore matrix and mean percentile along the diagonal."""
    if bert_score is None:
        raise ImportError("bert_score package required for identification_matrix")
    n_dec = len(decoded_windows)
    n_ref = len(reference_windows)
    scores = np.zeros((n_dec, n_ref), dtype=float)
    for i, hyp in enumerate(decoded_windows):
        hyps = [hyp] * n_ref
        try:
            _, _, F = bert_score(hyps, list(reference_windows), model_type=bert_model)
            scores[i] = F.detach().cpu().numpy()
        except Exception:
            scores[i] = np.nan
    diag_percentiles: List[float] = []
    for i in range(min(n_dec, n_ref)):
        row = scores[i]
        value = row[i]
        if not np.isfinite(value):
            continue
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        percentile = float(np.mean(finite <= value))
        diag_percentiles.append(percentile)
    diag_pct = float(np.mean(diag_percentiles)) if diag_percentiles else float("nan")
    return scores, diag_pct
