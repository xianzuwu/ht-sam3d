import os
import json
import sys
import time
import torch
import argparse
from datetime import datetime
from itertools import cycle
from torch.utils.data import DataLoader

# =========================================================
# 环境与硬件补丁
# =========================================================
CACHE_DIR = "/data/L202500204/Projects/sam-3d-objects/.cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["TORCH_HUB_DIR"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR


def apply_hardware_patch():
    if not torch.cuda.is_available():
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f">>> Detected GPU: {gpu_name}")

    if "H100" in gpu_name or "H200" in gpu_name:
        print(">>> Applying H100 Stability Patch: Using native spconv & sdpa")
        os.environ["SPCONV_ALGO"] = "native"
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
        os.environ["TORCH_ALLOW_TF32"] = "0"


apply_hardware_patch()

from trellis.models.sam3d_adapter import SAM3DStructureFlowAdapter
from trellis.trainers.flow_matching.sam3d_trainer import SAM3DFlowMatchingTrainer
from trellis.datasets.sam3d_distill_dataset import SAM3DDistillDataset


class TeeLogger:
    def __init__(self, filepath):
        self.file = open(filepath, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def deep_update(base: dict, extra: dict):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_experiment_dir(base_output_dir: str, exp_name: str) -> str:
    run_name = f"{get_timestamp()}_{exp_name}"
    run_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)
    return run_dir


def save_json(filepath: str, obj: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--token_dir",
        type=str,
        default="/data/L202500204/Projects/trellis-sam-3d-objects/notebook/gt_tokens",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/data/L202500204/Projects/trellis-sam-3d-objects/notebook/images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sam3d_distill_experiment",
    )
    parser.add_argument("--exp_name", type=str, default="sam3d_stage1")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    gpu_name = torch.cuda.get_device_name(0)
    is_h100 = ("H100" in gpu_name) or ("H200" in gpu_name)

    # =========================================================
    # 默认配置
    # =========================================================
    model_cfg = {
        "in_channels": 8,
        "model_channels": 1024,
        "out_channels": 8,
        "num_blocks": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 1,
        "resolution": 16,
        "dino_model": "dinov2_vitl14",
        "use_fp16": not is_h100,
        "use_bf16": is_h100,
        "use_checkpoint": True,
    }

    dataset_cfg = {
        "token_dir": args.token_dir,
        "image_dir": args.image_dir,
        "image_size": 518,
    }

    trainer_cfg = {
        "optimizer": {
            "name": "AdamW",
            "args": {
                "lr": 1e-4,
                "weight_decay": 0.05,
                "betas": [0.9, 0.95],
            },
        },
        "lr_scheduler": {
            "name": "LinearWarmupLRScheduler",
            "args": {
                "warmup_steps": 100,
            },
        },
        "max_steps": args.num_steps,
        "i_save": 1000,
        "i_sample": 1000,
        "i_log": 10,
        "batch_size": args.batch_size,
    }

    # =========================================================
    # 从 JSON 覆盖配置
    # =========================================================
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            ext_cfg = json.load(f)

        ext_model_args = ext_cfg.get("models", {}).get("denoiser", {}).get("args", {})
        if ext_model_args:
            deep_update(model_cfg, ext_model_args)

        ext_dataset_args = ext_cfg.get("dataset", {}).get("args", {})
        if ext_dataset_args:
            deep_update(dataset_cfg, ext_dataset_args)

        ext_trainer_args = ext_cfg.get("trainer", {}).get("args", {})
        if ext_trainer_args:
            if "scheduler" in ext_trainer_args and "lr_scheduler" not in ext_trainer_args:
                ext_trainer_args["lr_scheduler"] = ext_trainer_args.pop("scheduler")
            deep_update(trainer_cfg, ext_trainer_args)

    # 强制按硬件覆盖精度设置
    model_cfg["use_fp16"] = not is_h100
    model_cfg["use_bf16"] = is_h100

    # =========================================================
    # 实验目录与日志
    # =========================================================
    run_dir = build_experiment_dir(args.output_dir, args.exp_name)
    log_path = os.path.join(run_dir, "log.txt")
    resolved_cfg_path = os.path.join(run_dir, "config_resolved.json")

    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    try:
        print("=" * 80)
        print(f"Run directory: {run_dir}")
        print(f"Log file: {log_path}")
        print(f"Resolved config: {resolved_cfg_path}")
        print("=" * 80)

        save_json(
            resolved_cfg_path,
            {
                "exp_name": args.exp_name,
                "model_cfg": model_cfg,
                "dataset_cfg": dataset_cfg,
                "trainer_cfg": trainer_cfg,
            },
        )

        print(">>> Final model config:", model_cfg)
        print(">>> Final dataset config:", dataset_cfg)
        print(">>> Final trainer config:", trainer_cfg)

        # =========================================================
        # 初始化模型
        # =========================================================
        print(">>> Initializing SAM3D Adapter Model...")
        model = SAM3DStructureFlowAdapter(**model_cfg).cuda()

        # =========================================================
        # 初始化数据集
        # =========================================================
        print(">>> Building dataset...")
        dataset = SAM3DDistillDataset(**dataset_cfg)

        dataloader = DataLoader(
            dataset,
            batch_size=trainer_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
        )

        # =========================================================
        # 初始化 Trainer
        # =========================================================
        print(">>> Initializing trainer...")
        trainer = SAM3DFlowMatchingTrainer(
            models={"denoiser": model},
            dataset=dataset,
            output_dir=run_dir,
            load_dir=None,
            step=0,
            batch_size=trainer_cfg["batch_size"],
            max_steps=trainer_cfg["max_steps"],
            optimizer=trainer_cfg["optimizer"],
            lr_scheduler=trainer_cfg.get("lr_scheduler", None),
            i_log=trainer_cfg.get("i_log", 10),
            i_save=trainer_cfg.get("i_save", 1000),
            i_sample=trainer_cfg.get("i_sample", 1000),
        )

        # =========================================================
        # 手动训练循环
        # =========================================================
        model.train()
        optimizer = trainer.optimizer
        lr_scheduler = getattr(trainer, "lr_scheduler", None)

        # bf16/H100 分支不需要 GradScaler；fp16 分支使用新接口
        scaler = None
        if not is_h100:
            scaler = torch.amp.GradScaler("cuda", enabled=True)

        autocast_dtype = torch.bfloat16 if is_h100 else torch.float16
        train_iter = cycle(dataloader)

        print(">>> Starting Distillation Training...")
        train_start = time.time()

        for step in range(1, trainer_cfg["max_steps"] + 1):
            trainer.step = step
            step_start = time.time()

            batch = next(train_iter)
            batch = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", enabled=True, dtype=autocast_dtype):
                losses, _ = trainer.training_losses(**batch)
                loss = losses["loss"]

            optimizer.zero_grad(set_to_none=True)

            if is_h100:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % trainer_cfg.get("i_log", 10) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - train_start
                step_time = time.time() - step_start

                log_obj = {
                    "time": {
                        "step": step_time,
                        "elapsed": elapsed,
                    },
                    "loss": {},
                    "status": {},
                    "lr": current_lr,
                    "step": step,
                }

                for k, v in losses.items():
                    try:
                        log_obj["loss"][k] = float(v.item())
                    except Exception:
                        pass

                print(json.dumps(log_obj, ensure_ascii=False))

            if step % trainer_cfg.get("i_save", 1000) == 0:
                ckpt_path = os.path.join(run_dir, "ckpts", f"checkpoint_{step:06d}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f">>> Saved checkpoint: {ckpt_path}")

        print(">>> Training finished!")

    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
