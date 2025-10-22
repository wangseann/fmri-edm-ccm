"""Language-model guided beam search with encoding priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError as exc:
    torch = None  # type: ignore
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _TRANSFORMERS_ERROR = exc
else:
    _TRANSFORMERS_ERROR = None


EncodingScorer = Callable[[str, Sequence[float], np.ndarray], float]


@dataclass
class _BeamState:
    input_ids: "torch.Tensor"
    text: str
    lm_score: float
    score: float
    finished: bool = False


class BeamDecoder:
    """Wraps a causal LM and fuses scores with encoding evidence."""

    def __init__(
        self,
        lm_name: str,
        beam_size: int = 5,
        topk_next: int = 10,
        alpha_encoding: float = 0.6,
        max_tokens_per_window: int = 25,
        device: Optional[str] = None,
    ):
        """Initialize the decoder and load language-model resources."""
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for BeamDecoder") from _TRANSFORMERS_ERROR
        if torch is None:
            raise ImportError("PyTorch is required for BeamDecoder") from _TORCH_ERROR
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModelForCausalLM.from_pretrained(lm_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.beam_size = int(beam_size)
        self.topk_next = int(topk_next)
        self.alpha_encoding = float(alpha_encoding)
        self.max_tokens = int(max_tokens_per_window)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode_prompt(self, prompt_text: str) -> "torch.Tensor":
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.size(-1) == 0:
            fallback_tokens = self.tokenizer.encode(
                self.tokenizer.eos_token or " ",
                add_special_tokens=False,
            )
            if not fallback_tokens:
                fallback_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
                fallback_tokens = [fallback_id]
            input_ids = torch.tensor([fallback_tokens], device=self.device)
        return input_ids

    def step(self, input_ids: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return top-k next token ids and log-probs for each batch item."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(log_probs, k=self.topk_next, dim=-1)
        return topk.indices.detach(), topk.values.detach()

    def decode_window(
        self,
        prompt_text: str,
        enc_scorer: EncodingScorer,
        tr_grid_s: Sequence[float],
        pred_target_TR: np.ndarray,
    ) -> str:
        input_ids = self._encode_prompt(prompt_text)
        initial_state = _BeamState(
            input_ids=input_ids[0],
            text=self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
            lm_score=0.0,
            score=0.0,
        )
        beams: List[_BeamState] = [initial_state]
        best_state = initial_state
        for _ in range(self.max_tokens):
            new_beams: List[_BeamState] = []
            for state in beams:
                if state.finished:
                    new_beams.append(state)
                    continue
                seq = state.input_ids.unsqueeze(0)
                token_ids, log_probs = self.step(seq)
                for tok_id, log_p in zip(token_ids[0], log_probs[0]):
                    tok_id = tok_id.view(1)
                    next_ids = torch.cat([state.input_ids, tok_id.to(self.device)], dim=0)
                    decoded = self.tokenizer.decode(next_ids, skip_special_tokens=True)
                    lm_score = state.lm_score + float(log_p.cpu())
                    enc_score = enc_scorer(decoded, tr_grid_s, pred_target_TR)
                    fused = (1 - self.alpha_encoding) * lm_score + self.alpha_encoding * enc_score
                    finished = bool(tok_id.item() == self.tokenizer.eos_token_id)
                    new_beams.append(
                        _BeamState(
                            input_ids=next_ids,
                            text=decoded,
                            lm_score=lm_score,
                            score=fused,
                            finished=finished,
                        )
                    )
            if not new_beams:
                break
            new_beams.sort(key=lambda st: st.score, reverse=True)
            beams = new_beams[: self.beam_size]
            if beams and beams[0].score > best_state.score:
                best_state = beams[0]
            if all(st.finished for st in beams):
                break
        return best_state.text
