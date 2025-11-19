"""src/model.py
Model factory + EcoHiCaLRT layer controller (unchanged except cosmetic)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

CACHE = ".cache/"

# ---------------------------------------------------------------------------
# model builder -------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(cfg.name, cache_dir=CACHE)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        cache_dir=CACHE,
        torch_dtype=torch.float16 if cfg.precision == "fp16" else torch.float32,
    )
    if getattr(cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if "LoRA" in cfg.parameter_update_modes and getattr(cfg, "lora_rank", 0) > 0:
        l_cfg = LoraConfig(r=cfg.lora_rank, lora_alpha=32, bias="none", dropout=0.05)
        model = get_peft_model(model, l_cfg)
        model.print_trainable_parameters()
    return model, tok

# ---------------------------------------------------------------------------
# Layer controller (identical to previous submission) -----------------------
# ---------------------------------------------------------------------------

class LayerController:
    """Hierarchical PID controller with compute-budget gate (EcoHiCaLRT)."""

    def __init__(self, model: torch.nn.Module, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.train_cfg = cfg.run.training
        self.has_lora = "LoRA" in cfg.run.model.parameter_update_modes

        self.layers: List[torch.nn.Module] = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        n = len(self.layers)
        self.mode: List[str] = ["FULL"] * n
        self.mode_patience: List[int] = [0] * n
        self.mode_counts: Dict[str, int] = defaultdict(int)

        self.r_star = [1e-3] * n
        self.r_ema = [0.0] * n
        self.f_scale = [1.0] * n
        self.i_r = [0.0] * n

        self.g_t = 1.0
        self.prev_loss: float | None = None
        self.int_e = 0.0
        self.prev_d = 0.0

        self.use_budget = hasattr(self.train_cfg, "compute_budget_target_ratio")
        self.F_target = None
        self.F_ema = 0.0
        self.i_c = 0.0
        if self.use_budget:
            self.F_target = self.train_cfg.compute_budget_target_ratio * self._baseline_flops()

        self.last_flops = 0.0
        self.cumulative_flops = 0.0
        self.last_budget_ok = True

    # ============================= public ==============================

    def update(self, loss_val: float, step_flops: int, grad_norm_total: float) -> None:
        self.last_flops = step_flops
        self.cumulative_flops += step_flops

        # compute-budget loop -------------------------------------------
        b_t = 1.0
        if self.use_budget and self.F_target is not None:
            self.F_ema = step_flops if self.F_ema == 0 else 0.95 * self.F_ema + 0.05 * step_flops
            e_c = (self.F_ema - self.F_target) / self.F_target
            self.i_c = 0.9 * self.i_c + e_c
            b_t = max(self.train_cfg.compute_gate.clamp_min, min(self.train_cfg.compute_gate.clamp_max, 1 - 0.4 * e_c - 0.05 * self.i_c))
            self.last_budget_ok = self.F_ema <= self.F_target
        else:
            self.last_budget_ok = True

        # global PID -----------------------------------------------------
        gains_g = self.train_cfg.pid_gains.global
        if self.prev_loss is None:
            self.prev_loss = loss_val
        d_loss = (loss_val - self.prev_loss) / (abs(self.prev_loss) + 1e-12)
        self.int_e = 0.95 * self.int_e + d_loss
        d_term = d_loss - self.prev_d
        delta = gains_g.Kp * d_loss + gains_g.Ki * self.int_e + gains_g.Kd * d_term
        self.g_t = max(0.5, min(1.5, self.g_t * (1 + delta)))
        self.prev_d = d_loss
        self.prev_loss = loss_val

        # layer-level PI + mode gates ------------------------------------
        gains_l = self.train_cfg.pid_gains.layer
        for idx, layer in enumerate(self.layers):
            params = [p for p in layer.parameters() if p.grad is not None]
            if not params:
                self.r_ema[idx] = 0.9 * self.r_ema[idx]
                continue
            grad = torch.cat([p.grad.detach().view(-1) for p in params])
            weight = torch.cat([p.detach().view(-1) for p in params])
            r_now = (grad.norm() / (weight.norm() + 1e-12)).item()
            self.r_ema[idx] = 0.9 * self.r_ema[idx] + 0.1 * r_now
            err = (self.r_ema[idx] - self.r_star[idx]) / (self.r_star[idx] + 1e-12)
            self.i_r[idx] = 0.9 * self.i_r[idx] + err
            adj = gains_l.Kp * err + gains_l.Ki * self.i_r[idx]
            self.f_scale[idx] = min(10.0, max(0.1, self.f_scale[idx] * (1 + adj)))

            if self.has_lora:
                thr = self.train_cfg.compute_gate.thresholds
                self.mode_patience[idx] += 1
                if self.mode[idx] == "FULL" and self.r_ema[idx] < thr.full_to_lora * b_t * self.r_star[idx] and self.mode_patience[idx] >= thr.patience_full_to_lora:
                    self._to_lora(idx)
                elif self.mode[idx] == "LORA" and self.r_ema[idx] < thr.lora_to_frozen * b_t * self.r_star[idx] and self.mode_patience[idx] >= thr.patience_lora_to_frozen:
                    self._freeze(idx)
                elif self.mode[idx] in {"FROZEN", "LORA"} and self.r_ema[idx] > thr.recover_to_full * b_t * self.r_star[idx]:
                    self._to_full(idx)
            else:
                thr = self.train_cfg.freeze_controller.thresholds
                self.mode_patience[idx] += 1
                if self.mode[idx] == "FULL" and self.r_ema[idx] < thr.full_to_frozen * self.r_star[idx] and self.mode_patience[idx] >= thr.patience_full_to_frozen:
                    self._freeze(idx)
                elif self.mode[idx] == "FROZEN" and self.r_ema[idx] > thr.recover_to_full * self.r_star[idx]:
                    self._to_full(idx)
        self._update_mode_counts()

    def scale_gradients(self) -> None:
        for idx, layer in enumerate(self.layers):
            if self.mode[idx] == "FROZEN":
                continue
            factor = self.g_t * self.f_scale[idx]
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad.mul_(factor)

    # ---------------- internal helpers ---------------------------------

    def _freeze(self, idx: int) -> None:
        for p in self.layers[idx].parameters():
            p.requires_grad = False
        self.mode[idx] = "FROZEN"
        self.mode_patience[idx] = 0

    def _to_lora(self, idx: int) -> None:
        if not self.has_lora:
            return
        for n, p in self.layers[idx].named_parameters():
            p.requires_grad = "lora_" in n
        self.mode[idx] = "LORA"
        self.mode_patience[idx] = 0

    def _to_full(self, idx: int) -> None:
        for p in self.layers[idx].parameters():
            p.requires_grad = True
        self.mode[idx] = "FULL"
        self.mode_patience[idx] = 0

    def _baseline_flops(self) -> float:
        return 6 * sum(p.numel() for p in self.model.parameters()) * self.cfg.run.training.max_steps

    def _update_mode_counts(self) -> None:
        self.mode_counts = defaultdict(int)
        for m in self.mode:
            self.mode_counts[m] += 1
