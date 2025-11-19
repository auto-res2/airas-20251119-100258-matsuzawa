"""src/preprocess.py
Dataset handling + *correct* label alignment (question tokens \u2192 masked, answer
tokens \u2192 target).
"""
from __future__ import annotations

from functools import partial
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Q-A helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def encode_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def decode_answer(ans: str) -> str:
    import fractions
    import re

    text = ans.strip().replace("$", "")
    text = re.sub(r"[^0-9A-Za-z./-]", "", text)
    try:
        if "/" in text:
            return str(float(fractions.Fraction(text)))
        return str(float(text))
    except Exception:
        return text.lower()


# ---------------------------------------------------------------------------
# Data module ---------------------------------------------------------------
# ---------------------------------------------------------------------------

class GSM8KDataModule:
    """Prepares train + hold-out loaders with strict label separation."""

    def __init__(self, ds_cfg, tokenizer, mode: str):
        self.cfg = ds_cfg
        self.tok = tokenizer
        self.mode = mode

    # ------------------------------------------------------------------
    # public                                                             
    # ------------------------------------------------------------------

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        ds = load_dataset("gsm8k", "main", cache_dir=".cache/")
        train = ds["train"]
        hold = train.select(range(200))
        if self.mode == "trial":
            train = train.select(range(16))
        proc = partial(self._process_example)
        train = train.map(proc, remove_columns=train.column_names)
        hold = hold.map(proc, remove_columns=hold.column_names)
        cols = ["input_ids", "attention_mask", "labels"]
        train.set_format(type="torch", columns=cols)
        hold.set_format(type="torch", columns=cols)
        collate = partial(self._collate, pad_id=self.tok.pad_token_id)
        dl_train = DataLoader(
            train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=self.cfg.drop_last,
            collate_fn=collate,
        )
        dl_hold = DataLoader(hold, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collate)
        return dl_train, dl_hold

    # ------------------------------------------------------------------
    # example processing                                                 
    # ------------------------------------------------------------------

    def _process_example(self, ex: Dict) -> Dict:
        prompt_ids = self.tok(
            encode_prompt(ex["question"]),
            truncation=True,
            max_length=self.cfg.max_length - 128,  # leave room for answer
            add_special_tokens=True,
        ).input_ids
        # answer tokens (append EOS)
        ans_ids: List[int] = self.tok(" " + ex["answer"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        input_ids = prompt_ids + ans_ids
        if len(input_ids) > self.cfg.max_length:
            # truncate from start (keep answer intact)
            overflow = len(input_ids) - self.cfg.max_length
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + ans_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ------------------------------------------------------------------
    # collator                                                          
    # ------------------------------------------------------------------

    @staticmethod
    def _collate(batch, *, pad_id: int):  # type: ignore
        out = {}
        keys = batch[0].keys()
        for k in keys:
            pad_val = pad_id if k == "input_ids" else 0 if k == "attention_mask" else -100
            seqs = [torch.tensor(b[k], dtype=torch.long) for b in batch]
            out[k] = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_val)
        return out
