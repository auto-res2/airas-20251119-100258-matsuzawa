"""src/train.py
Single-run trainer with Optuna isolation fixed and trial/full auto-config.
All metrics logged to WandB; no stdout JSON artefacts are produced.
"""
from __future__ import annotations

import gc
import os
import random
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.profiler import ProfilerActivity, profile
from tqdm.auto import tqdm

import wandb

try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover – CI may drop optuna when n_trials==0
    optuna = None

# ---------------------------------------------------------------------------
# local  imports (relative ‑ works under Hydra)
# ---------------------------------------------------------------------------
from src.model import LayerController, build_model
from src.preprocess import GSM8KDataModule, decode_answer, encode_prompt

CACHE = ".cache/"
# ---------------------------------------------------------------------------
# utility helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def approx_backward_flops(model: torch.nn.Module) -> int:
    """Cheap analytical estimate: 6 × trainable parameter count."""
    return 6 * sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# profiler-based FLOP calibration (one real step) ----------------------------
# ---------------------------------------------------------------------------

def _measure_real_backward_flops(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> int:
    for p in model.parameters():  # clear grads so profiler counts backward ops
        p.grad = None
    with profile(activities=[ProfilerActivity.CUDA], with_flops=True) as prof:
        loss = model(**batch).loss
        loss.backward()
    total = 0
    for evt in prof.key_averages():
        if getattr(evt, "flops", None):
            total += evt.flops
    return int(total)


# ---------------------------------------------------------------------------
# Trainer --------------------------------------------------------------------
# ---------------------------------------------------------------------------

class Trainer:
    """Handles one full experimental run (optionally after Optuna sweep)."""

    def __init__(self, cfg: DictConfig, save_dir: Path):
        self.cfg = cfg
        self.run_cfg = cfg.run           # alias – will mutate after Optuna
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_global_seed(42)

        # build tokenizer once – reused across proxy and main run ----------
        from transformers import AutoTokenizer  # local import (avoids boot-time cost if not needed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.run_cfg.model.name, cache_dir=CACHE)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        # ------------------------------------------------------------------
        # optional Optuna hyper-parameter optimisation ---------------------
        # ------------------------------------------------------------------
        self._maybe_optuna()

        # ------------------------------------------------------------------
        # data loaders (use *post-Optuna* hyper-params) --------------------
        # ------------------------------------------------------------------
        self._build_dataloaders()
        # limit batches in trial mode for ultrafast smoke test
        if cfg.mode == "trial":
            self.dl_train = list(islice(self.dl_train, 2))

        # ------------------------------------------------------------------
        # final model / optimiser / controller -----------------------------
        # ------------------------------------------------------------------
        self._build_model_and_optimiser()

        # bookkeeping ------------------------------------------------------
        self.global_step = 0
        self.flop_scale: float | None = None
        self.budget_ok_counter = 0
        self.wandb_run = None
        self.wall_t0 = time.time()

    # ======================================================================
    # public interface                                                      
    # ======================================================================

    def run(self) -> None:
        self._init_wandb()
        self._train_loop()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    # ------------------------------------------------------------------
    # data-module -------------------------------------------------------
    # ------------------------------------------------------------------

    def _build_dataloaders(self) -> None:
        dm = GSM8KDataModule(self.run_cfg.dataset, self.tokenizer, mode=self.cfg.mode)
        self.dl_train, self.dl_hold = dm.get_dataloaders()

    # ------------------------------------------------------------------
    # model / optimiser -------------------------------------------------
    # ------------------------------------------------------------------

    def _build_model_and_optimiser(self) -> None:
        self.model, _ = build_model(self.run_cfg.model)
        self.model.to(self.device)

        tr_cfg = self.run_cfg.training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tr_cfg.base_learning_rate,
            betas=tuple(tr_cfg.betas),
            eps=tr_cfg.eps,
            weight_decay=tr_cfg.weight_decay,
        )
        self.controller = LayerController(self.model, self.cfg)
        self.training_cfg = tr_cfg

    # ------------------------------------------------------------------
    # Optuna sweep (fixed) ---------------------------------------------
    # ------------------------------------------------------------------

    def _maybe_optuna(self) -> None:
        n_trials = int(self.run_cfg.optuna.n_trials)
        if n_trials <= 1:
            return  # nothing to do
        if optuna is None:
            raise RuntimeError("Optuna requested but the package is unavailable.")
        print(f"[Optuna] starting sweep with {n_trials} trials …")

        study = optuna.create_study(direction=self.run_cfg.optuna.direction)

        # helper: inject trial suggestion into a *copy* of cfg ------------
        def _suggest(trial_: "optuna.Trial", cfg_copy: DictConfig) -> None:  # type: ignore
            for dotted, space in self.run_cfg.optuna.search_space.items():
                if space["type"] == "loguniform":
                    val = trial_.suggest_float(dotted, space["low"], space["high"], log=True)
                elif space["type"] == "uniform":
                    val = trial_.suggest_float(dotted, space["low"], space["high"])
                elif space["type"] == "int":
                    val = trial_.suggest_int(dotted, space["low"], space["high"])
                elif space["type"] == "categorical":
                    val = trial_.suggest_categorical(dotted, space["choices"])
                else:
                    raise ValueError(space["type"])
                OmegaConf.update(cfg_copy, f"run.{dotted}", val, merge=False)

        # fast EM/TFLOP proxy -------------------------------------------
        def _objective(trial_: "optuna.Trial") -> float:  # type: ignore
            cfg_copy = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
            _suggest(trial_, cfg_copy)
            run_c = cfg_copy.run
            # build small model & data -----------------------------------
            model, _ = build_model(run_c.model)
            model.to(self.device)
            dm_tmp = GSM8KDataModule(run_c.dataset, self.tokenizer, mode=self.cfg.mode)
            dl_train_tmp, dl_hold_tmp = dm_tmp.get_dataloaders()
            max_steps = max(10, int(0.05 * run_c.training.max_steps))
            optim_tmp = torch.optim.AdamW(
                model.parameters(),
                lr=run_c.training.base_learning_rate,
                betas=tuple(run_c.training.betas),
                eps=run_c.training.eps,
                weight_decay=run_c.training.weight_decay,
            )
            model.train()
            steps = 0
            for batch in dl_train_tmp:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                optim_tmp.step()
                optim_tmp.zero_grad(set_to_none=True)
                steps += 1
                if steps >= max_steps:
                    break
            # quick hold-out EM for proxy score ---------------------------
            hold_batch = next(iter(dl_hold_tmp))
            em = self._quick_exact_match(model, hold_batch)
            proxy_flops = approx_backward_flops(model) * max_steps / 1e12
            # clean up                                                   
            del model, optim_tmp, dl_train_tmp, dl_hold_tmp
            torch.cuda.empty_cache(); gc.collect()
            return em / (proxy_flops + 1e-12)

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

        print("[Optuna] best value:", study.best_value)
        # merge best params back into main cfg ---------------------------
        for dotted, val in study.best_trial.params.items():
            OmegaConf.update(self.cfg, f"run.{dotted}", val, merge=False)
        # refresh references
        self.run_cfg = self.cfg.run
        print("[Optuna] best hyper-parameters injected into cfg – proceeding with full training.")

    # ------------------------------------------------------------------
    # WandB init --------------------------------------------------------
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        if self.cfg.wandb.mode == "disabled":
            return  # offline/trial mode
        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.project,
            id=self.run_cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(self.cfg, resolve=True),
            mode=self.cfg.wandb.mode,
        )
        print(f"[WandB] URL → {self.wandb_run.url}")

    # ------------------------------------------------------------------
    # main optimisation loop -------------------------------------------
    # ------------------------------------------------------------------

    def _train_loop(self) -> None:
        pbar = tqdm(total=self.training_cfg.max_steps, dynamic_ncols=True, desc="train")
        while self.global_step < self.training_cfg.max_steps:
            for raw_batch in self.dl_train:
                if self.global_step >= self.training_cfg.max_steps:
                    break
                loss_val = self._train_step(raw_batch)
                pbar.update(1)
        pbar.close()
        # final evaluation ----------------------------------------------
        em_test = self._evaluate_gsm8k_test(sample_only=self.cfg.mode == "trial")
        tflops = self.controller.cumulative_flops / 1e12
        primary = em_test / (tflops if tflops > 0 else 1.0)
        summary = {
            "exact_match_test": em_test,
            "total_backward_tflops": tflops,
            "compute_normalised_accuracy": primary,
            "budget_compliance_rate": self.budget_ok_counter / max(1, self.global_step),
            "wallclock_seconds": time.time() - self.wall_t0,
        }
        for k, v in summary.items():
            print(f"{k:30s}: {v:.6f}" if isinstance(v, float) else f"{k:30s}: {v}")
            if self.wandb_run is not None:
                self.wandb_run.summary[k] = v

    # ------------------------------------------------------------------
    # one optimisation step -------------------------------------------
    # ------------------------------------------------------------------

    def _train_step(self, raw_batch: Dict[str, torch.Tensor]) -> float:
        batch = {k: v.to(self.device) for k, v in raw_batch.items()}

        # calibrate real FLOPs at step-0 ---------------------------------
        if self.global_step == 0:
            real_flop = _measure_real_backward_flops(self.model, batch)
            self.flop_scale = real_flop / max(1, approx_backward_flops(self.model))
            loss = self.model(**batch).loss / self.run_cfg.dataset.grad_accumulation_steps
            loss.backward()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss / self.run_cfg.dataset.grad_accumulation_steps
            loss.backward()
            real_flop = int((self.flop_scale or 1.0) * approx_backward_flops(self.model))

        update_now = (self.global_step + 1) % self.run_cfg.dataset.grad_accumulation_steps == 0
        loss_item = loss.item() * self.run_cfg.dataset.grad_accumulation_steps  # rescale for logging

        if update_now:
            total_grad_norm = torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in self.model.parameters() if p.grad is not None)
            ).item()
            self.controller.update(loss_item, real_flop, total_grad_norm)
            self.controller.scale_gradients()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.controller.last_budget_ok:
                self.budget_ok_counter += 1

        # logging --------------------------------------------------------
        if self.wandb_run is not None:
            metrics = self._collect_metrics(loss_item, real_flop)
            self.wandb_run.log(metrics, step=self.global_step)
        self.global_step += 1
        return loss_item

    # ------------------------------------------------------------------
    # metric collection ------------------------------------------------
    # ------------------------------------------------------------------

    def _collect_metrics(self, train_loss: float, step_flops: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "step": self.global_step,
            "train_loss": train_loss,
            "backward_flops": step_flops,
            "cumulative_flops": self.controller.cumulative_flops,
            "budget_ok": float(self.controller.last_budget_ok),
            "g_t": self.controller.g_t,
            "num_full": self.controller.mode_counts.get("FULL", 0),
            "num_lora": self.controller.mode_counts.get("LORA", 0),
            "num_frozen": self.controller.mode_counts.get("FROZEN", 0),
        }
        hold_int = getattr(self.run_cfg.logging, "holdout_interval_steps", 50)
        if self.global_step % hold_int == 0:
            batch = next(iter(self.dl_hold))
            data["exact_match_holdout"] = self._quick_exact_match(self.model, batch)
        return data

    # ------------------------------------------------------------------
    # evaluation helpers ----------------------------------------------
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quick_exact_match(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
        model.eval()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=32,
        )
        preds = self.tokenizer.batch_decode(out_ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        gold_ids = batch["labels"].clone()
        gold_ids[gold_ids == -100] = self.tokenizer.pad_token_id
        gold = self.tokenizer.batch_decode(gold_ids, skip_special_tokens=True)
        correct = sum(decode_answer(p) == decode_answer(g) for p, g in zip(preds, gold))
        model.train()
        return correct / len(preds)

    @torch.no_grad()
    def _evaluate_gsm8k_test(self, *, sample_only: bool = False) -> float:
        from datasets import load_dataset

        self.model.eval()
        ds = load_dataset("gsm8k", "main", split="test", cache_dir=CACHE)
        if sample_only:
            ds = ds.select(range(20))
        correct = 0
        for ex in tqdm(ds, desc="eval:test", leave=False):
            prompt = encode_prompt(ex["question"])
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**enc, max_new_tokens=32)[0][enc.input_ids.shape[1]:]
            pred = self.tokenizer.decode(out, skip_special_tokens=True)
            if decode_answer(pred) == decode_answer(ex["answer"]):
                correct += 1
        self.model.train()
        return correct / len(ds)


# ---------------------------------------------------------------------------
# Hydra entry ----------------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def _main(cfg: DictConfig) -> None:  # type: ignore
    # resolve run-config file ---------------------------------------------
    runs_dir = Path(__file__).resolve().parent.parent / "config" / "runs"
    run_file = runs_dir / f"{cfg.run}.yaml"
    if not run_file.exists():
        raise FileNotFoundError(f"Unknown run-id '{cfg.run}'. Expected file: {run_file}")
    run_cfg = OmegaConf.load(run_file)
    OmegaConf.update(cfg, "run", run_cfg, merge=False)

    # apply mode-specific patches -----------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        OmegaConf.update(cfg, "optuna.n_trials", 0, merge=True)
        OmegaConf.update(cfg, "run.optuna.n_trials", 0, merge=True)
        OmegaConf.update(cfg, "run.training.max_steps", 2, merge=True)
        OmegaConf.update(cfg, "run.logging.holdout_interval_steps", 1, merge=True)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    save_root = Path(cfg.results_dir).expanduser()
    trainer = Trainer(cfg, save_root / cfg.run.run_id)
    trainer.run()


if __name__ == "__main__":
    _main()
