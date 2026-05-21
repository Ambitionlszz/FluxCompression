#!/usr/bin/env python3
"""
Lightweight Stage2 hyperparameter tuner.

This script intentionally sits outside train_stage2.py. It launches short
Stage2 runs with sampled hyperparameters, parses eval metrics from stdout, and
writes a ranked summary plus a recommended full-training shell script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


EVAL_RE = re.compile(
    r"\[eval\s+(?P<step>\d+)\]\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"bpp=(?P<bpp>[-+0-9.eE]+)\s+"
    r"psnr=(?P<psnr>[-+0-9.eE]+)\s+"
    r"lpips=(?P<lpips>[-+0-9.eE]+)"
    r"(?:\s+dists=(?P<dists>[-+0-9.eE]+))?"
)
TRAIN_RE = re.compile(
    r"\[step\s+(?P<step>\d+)\].*?"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"bpp=(?P<bpp>[-+0-9.eE]+)\s+"
    r"psnr=(?P<psnr>[-+0-9.eE]+).*?"
    r"lpips=(?P<lpips>[-+0-9.eE]+).*?"
    r"dists=(?P<dists>[-+0-9.eE]+).*?"
    r"g=(?P<g_loss>[-+0-9.eE]+)\s+"
    r"d_r=(?P<d_real>[-+0-9.eE]+)\s+"
    r"d_f=(?P<d_fake>[-+0-9.eE]+)"
)


DEFAULT_SPACE = {
    "lambda_rate": [0.1, 0.2, 0.3, 0.5, 0.8],
    "lambda_l2": [0.5, 1.0, 2.0],
    "lambda_lpips": [0.5, 1.0, 1.5],
    "lambda_dists": [0.1, 0.2, 0.5],
    "lambda_gan": [0.03, 0.05, 0.1, 0.2],
    "lr": [1e-5, 2e-5, 5e-5],
    "lr_disc": [5e-6, 1e-5, 2e-5, 5e-5],
    "grad_clip": [0.5, 1.0, 2.0],
    "train_schedule_steps": [50, 100, 150],
    "guidance": [0.8, 1.0, 1.2],
    "lr_decay_steps": ["10000,20000,35000", "5000,10000,15000", "20000,40000,70000"],
    "lr_decay_values": ["2e-5,1e-5,1e-6", "1e-5,5e-6,1e-6", "5e-5,2e-5,1e-5"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune FluxCodec Stage2 hyperparameters")

    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--steps_per_trial", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000000)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--refine_top_k", type=int, default=3, help="Run longer validation for top-k coarse trials")
    p.add_argument("--refine_steps", type=int, default=10000)
    p.add_argument("--refine_eval_every", type=int, default=2000)
    p.add_argument("--refine_eval_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num_processes", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--accelerate", type=str, default="accelerate")
    p.add_argument("--train_script", type=str, default="train_stage2.py")
    p.add_argument("--output_dir", type=str, default="./outputs/fluxcodec_stage2_tuning")
    p.add_argument("--space_json", type=str, default="", help="Optional JSON file overriding the search space")
    p.add_argument("--dry_run", action="store_true")

    # Scoring. Larger is better. BPP is reported but not used by default.
    p.add_argument("--psnr_weight", type=float, default=0.05)
    p.add_argument("--lpips_weight", type=float, default=40.0)
    p.add_argument("--dists_weight", type=float, default=20.0)
    p.add_argument("--instability_weight", type=float, default=2.0)

    # Paths copied from train_stage2.sh defaults.
    p.add_argument("--train_root", type=str, default="/data2/luosheng/code/flux2/datasets/train")
    p.add_argument("--val_root", type=str, default="/data2/luosheng/data/Datasets/Kodak")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    p.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    p.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    p.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")
    p.add_argument("--elic_ckpt", type=str, default="/data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth")
    p.add_argument("--dinov2_repo", type=str, default="/data2/luosheng/hf_models/hub/dinov2")
    p.add_argument("--dinov2_weights", type=str, default="/data2/luosheng/hf_models/hub/dinov2/dinov2_vitb14_reg4_pretrain.pth")

    # Fixed training args.
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_schedule_steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--gan_loss_type", type=str, default="multilevel_sigmoid_s")
    p.add_argument("--disc_cv_type", type=str, default="dinov2_reg")
    p.add_argument("--lr_decay_steps", type=str, default="10000,20000,35000")
    p.add_argument("--lr_decay_values", type=str, default="2e-5,1e-5,1e-6")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--codec_ch_emd", type=int, default=128)
    p.add_argument("--codec_channel", type=int, default=320)
    p.add_argument("--codec_channel_out", type=int, default=128)
    p.add_argument("--codec_num_slices", type=int, default=5)
    p.add_argument("--use_aux_encoder", type=int, default=1)
    p.add_argument("--use_aux_decoder", type=int, default=1)

    return p.parse_args()


def load_space(path: str) -> dict[str, list[Any]]:
    if not path:
        return DEFAULT_SPACE
    with open(path, "r", encoding="utf-8") as f:
        space = json.load(f)
    if not isinstance(space, dict):
        raise ValueError("space_json must contain an object mapping arg names to value lists")
    return space


def sample_params(space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in space.items()}


def as_cli_args(options: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def build_train_options(
    args: argparse.Namespace,
    trial_dir: Path,
    params: dict[str, Any],
    max_steps: int | None = None,
    eval_every: int | None = None,
    eval_batches: int | None = None,
) -> dict[str, Any]:
    base = {
        "train_root": args.train_root,
        "val_root": args.val_root,
        "output_dir": str(trial_dir),
        "stage1_ckpt": args.stage1_ckpt,
        "model_name": args.model_name,
        "flux_ckpt": args.flux_ckpt,
        "ae_ckpt": args.ae_ckpt,
        "qwen_ckpt": args.qwen_ckpt,
        "elic_ckpt": args.elic_ckpt,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_steps": max_steps if max_steps is not None else args.steps_per_trial,
        "grad_clip": args.grad_clip,
        "train_schedule_steps": args.train_schedule_steps,
        "guidance": args.guidance,
        "gan_loss_type": args.gan_loss_type,
        "disc_cv_type": args.disc_cv_type,
        "dinov2_repo": args.dinov2_repo,
        "dinov2_weights": args.dinov2_weights,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_decay_values": args.lr_decay_values,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "codec_ch_emd": args.codec_ch_emd,
        "codec_channel": args.codec_channel,
        "codec_channel_out": args.codec_channel_out,
        "codec_num_slices": args.codec_num_slices,
        "use_aux_encoder": args.use_aux_encoder,
        "use_aux_decoder": args.use_aux_decoder,
        "log_every": min(20, eval_every if eval_every is not None else args.eval_every),
        "eval_every": eval_every if eval_every is not None else args.eval_every,
        "save_every": args.save_every,
        "eval_batches": eval_batches if eval_batches is not None else args.eval_batches,
        "use_tensorboard": True,
        "save_log_file": True,
    }
    base.update(params)
    return base


def build_command(args: argparse.Namespace, train_options: dict[str, Any]) -> list[str]:
    return [
        args.accelerate,
        "launch",
        "--mixed_precision",
        args.mixed_precision,
        "--num_processes",
        str(args.num_processes),
        args.train_script,
        *as_cli_args(train_options),
    ]


def parse_metrics(log_path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    evals: list[dict[str, float]] = []
    trains: list[dict[str, float]] = []
    if not log_path.exists():
        return trains, evals
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            trains.append({k: float(v) for k, v in m.groupdict(default="0").items()})
            continue
        m = EVAL_RE.search(line)
        if m:
            evals.append({k: float(v) for k, v in m.groupdict(default="0").items()})
    return trains, evals


def instability_penalty(trains: list[dict[str, float]]) -> float:
    if not trains:
        return 1.0
    last = trains[-min(10, len(trains)) :]
    penalty = 0.0
    for item in last:
        d_real = abs(item.get("d_real", 0.0))
        d_fake = abs(item.get("d_fake", 0.0))
        g_loss = abs(item.get("g_loss", 0.0))
        if d_real > 10.0 or d_fake > 10.0 or g_loss > 10.0:
            penalty += 1.0
        if d_real < 1e-5 and d_fake < 1e-5:
            penalty += 0.5
    return penalty / max(len(last), 1)


def score_trial(args: argparse.Namespace, trains: list[dict[str, float]], evals: list[dict[str, float]]) -> dict[str, Any]:
    if not evals:
        return {
            "status": "no_eval",
            "score": -1e9,
            "instability": instability_penalty(trains),
        }
    best = max(
        evals,
        key=lambda m: (
            args.psnr_weight * m["psnr"]
            - args.lpips_weight * m["lpips"]
            - args.dists_weight * m.get("dists", 0.0)
        ),
    )
    instability = instability_penalty(trains)
    score = (
        args.psnr_weight * best["psnr"]
        - args.lpips_weight * best["lpips"]
        - args.dists_weight * best.get("dists", 0.0)
        - args.instability_weight * instability
    )
    return {
        "status": "ok",
        "score": score,
        "instability": instability,
        "best_eval": best,
        "last_eval": evals[-1],
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "rank",
        "trial",
        "phase",
        "status",
        "score",
        "best_step",
        "best_bpp",
        "best_psnr",
        "best_lpips",
        "best_dists",
        "instability",
        "lambda_rate",
        "lambda_l2",
        "lambda_lpips",
        "lambda_dists",
        "lambda_gan",
        "lr",
        "lr_disc",
        "grad_clip",
        "train_schedule_steps",
        "guidance",
        "lr_decay_steps",
        "lr_decay_values",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def write_best_script(path: Path, args: argparse.Namespace, best: dict[str, Any]) -> None:
    params = best["params"]
    full_options = build_train_options(args, Path("./outputs/fluxcodec_stage2_best"), params)
    full_options["max_steps"] = 100000
    full_options["output_dir"] = "./outputs/fluxcodec_stage2_best"
    full_options["save_every"] = 10000
    full_options["eval_every"] = 2000
    cmd = build_command(args, full_options)
    content = "\n".join(
        [
            "#!/bin/bash",
            "set -e",
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}",
            shell_join(cmd),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def run_trial(
    args: argparse.Namespace,
    trial_id: int | str,
    params: dict[str, Any],
    phase: str = "coarse",
    max_steps: int | None = None,
    eval_every: int | None = None,
    eval_batches: int | None = None,
) -> dict[str, Any]:
    root = Path(args.output_dir).resolve()
    trial_name = f"trial_{trial_id:03d}" if isinstance(trial_id, int) else f"trial_{trial_id}"
    trial_dir = root / phase / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    train_options = build_train_options(
        args,
        trial_dir,
        params,
        max_steps=max_steps,
        eval_every=eval_every,
        eval_batches=eval_batches,
    )
    cmd = build_command(args, train_options)
    log_path = trial_dir / "stdout.log"

    write_json(
        trial_dir / "trial_config.json",
        {"trial": trial_id, "phase": phase, "params": params, "train_options": train_options},
    )
    (trial_dir / "command.sh").write_text(shell_join(cmd) + "\n", encoding="utf-8")

    print(f"\n=== {phase.capitalize()} trial {trial_id} ===")
    print(json.dumps(params, indent=2))
    print(shell_join(cmd))

    if args.dry_run:
        return {
            "trial": trial_id,
            "phase": phase,
            "trial_dir": str(trial_dir),
            "params": params,
            "status": "dry_run",
            "score": 0.0,
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    return_code = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = proc.wait()

    trains, evals = parse_metrics(log_path)
    scored = score_trial(args, trains, evals)
    result = {
        "trial": trial_id,
        "phase": phase,
        "trial_dir": str(trial_dir),
        "params": params,
        "return_code": return_code,
        "train_metrics": trains,
        "eval_metrics": evals,
        **scored,
    }
    if return_code != 0 and result["status"] == "ok":
        result["status"] = "failed_after_eval"
    elif return_code != 0:
        result["status"] = "failed"
    write_json(trial_dir / "result.json", result)
    print(f"{phase.capitalize()} trial {trial_id}: status={result['status']} score={result['score']:.6f}")
    return result


def flatten_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(results, key=lambda r: r.get("score", -1e9), reverse=True)
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(ranked, start=1):
        best = result.get("best_eval") or {}
        params = result.get("params") or {}
        rows.append(
            {
                "rank": rank,
                "trial": result.get("trial"),
                "phase": result.get("phase"),
                "status": result.get("status"),
                "score": result.get("score"),
                "best_step": best.get("step"),
                "best_bpp": best.get("bpp"),
                "best_psnr": best.get("psnr"),
                "best_lpips": best.get("lpips"),
                "best_dists": best.get("dists"),
                "instability": result.get("instability"),
                **params,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    space = load_space(args.space_json)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "search_space.json", space)
    write_json(out / "tuner_config.json", vars(args))

    coarse_results: list[dict[str, Any]] = []
    for trial_id in range(args.trials):
        params = sample_params(space, rng)
        result = run_trial(args, trial_id, params, phase="coarse")
        coarse_results.append(result)
        rows = flatten_summary(coarse_results)
        write_json(out / "coarse_best_trials.json", rows)
        write_csv_summary(out / "coarse_best_trials.csv", rows)

    coarse_rows = flatten_summary(coarse_results)
    write_json(out / "best_trials.json", coarse_rows)
    write_csv_summary(out / "best_trials.csv", coarse_rows)

    ok_coarse = [
        r for r in sorted(coarse_results, key=lambda r: r.get("score", -1e9), reverse=True)
        if r.get("best_eval")
    ]

    refine_results: list[dict[str, Any]] = []
    if args.refine_top_k > 0 and ok_coarse:
        selected = ok_coarse[: args.refine_top_k]
        write_json(
            out / "refine_selected.json",
            [
                {
                    "coarse_trial": item["trial"],
                    "coarse_score": item["score"],
                    "params": item["params"],
                    "best_eval": item.get("best_eval"),
                }
                for item in selected
            ],
        )
        for refine_id, item in enumerate(selected):
            result = run_trial(
                args,
                f"{refine_id:03d}_from_{item['trial']}",
                item["params"],
                phase="refine",
                max_steps=args.refine_steps,
                eval_every=args.refine_eval_every,
                eval_batches=args.refine_eval_batches,
            )
            result["coarse_trial"] = item["trial"]
            refine_results.append(result)
            refine_rows = flatten_summary(refine_results)
            write_json(out / "refine_best_trials.json", refine_rows)
            write_csv_summary(out / "refine_best_trials.csv", refine_rows)

    final_results = refine_results if refine_results else coarse_results
    rows = flatten_summary(final_results)
    write_json(out / "best_trials.json", rows)
    write_csv_summary(out / "best_trials.csv", rows)
    ok_final = [r for r in sorted(final_results, key=lambda r: r.get("score", -1e9), reverse=True) if r.get("best_eval")]
    if ok_final:
        write_best_script(out / "best_train_stage2.sh", args, ok_final[0])

    print("\n=== Best trials ===")
    for row in rows[:5]:
        print(
            f"rank={row['rank']} trial={row['trial']} score={float(row['score']):.6f} "
            f"bpp={row.get('best_bpp')} psnr={row.get('best_psnr')} "
            f"lpips={row.get('best_lpips')} dists={row.get('best_dists')} "
            f"params={{lambda_rate:{row.get('lambda_rate')}, lambda_gan:{row.get('lambda_gan')}, "
            f"lr:{row.get('lr')}, lr_disc:{row.get('lr_disc')}}}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
