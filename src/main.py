"""src/main.py – thin wrapper that launches src.train via subprocess."""
from __future__ import annotations

import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):  # type: ignore
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("[MAIN] launching subprocess:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
