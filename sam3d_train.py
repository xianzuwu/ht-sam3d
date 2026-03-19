import os
import torch
import argparse
import json
from torch.utils.data import DataLoader
from itertools import cycle
from easydict import EasyDict

# =========================================================
# 🔥 第一部分：环境与硬件补丁 (保持不变)
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
    if not torch.cuda.is_available(): return
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

# =========================================================
# 第二部分：主逻辑
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--token_dir", type=str, default="/data/L202500204/Projects/trellis-sam-3d-objects/notebook/gt_tokens")
    parser.add_argument("--image_dir", type=str, default="/data/L202500204/Projects/trellis-sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567")
    parser.add_argument("--output_dir", type=str, default="outputs/sam3d_distill_experiment")
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--num_steps", type=int, default=1000)
    args = parser.parse_args()

    # 1. 模型配置
    is_h100 = "H100" in torch.cuda.get_device_name(0)
    model_cfg = {
        "in_channels": 8, "model_channels": 1024, "out_channels": 8,
        "num_blocks": 24, "num_heads": 16, "mlp_ratio": 4,
        "patch_size": 1, "resolution": 16, "dino_model": "dinov2_vitl14",
        "use_fp16": not is_h100, "use_bf16": is_h100, "use_checkpoint": True
    }

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            ext_cfg = json.load(f)
            if 'model' in ext_cfg: model_cfg.update(ext_cfg['model'])
            model_cfg["use_fp16"], model_cfg["use_bf16"] = not is_h100, is_h100

    print(">>> Initializing SAM3D Adapter Model...")
    model = SAM3DStructureFlowAdapter(**model_cfg).cuda()
    
    # 2. 数据集 (请确保已经手动删除了 dataset.py 里的 path 参数)
    dataset = SAM3DDistillDataset(token_dir=args.token_dir, image_dir=args.image_dir, image_size=518)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 3. 核心修复：适配新版 Trainer API
    # ---------------------------------------------------------
    # 🔥 关键点 A：打平配置结构，去掉 "solver" 嵌套
    trainer_cfg = EasyDict({
        "optimizer": {"name": "AdamW", "args": {"lr": 1e-4, "weight_decay": 0.05}},
        "scheduler": {"name": "cosine_with_restart", "args": {"warmup_steps": 100}}
    })

    # 🔥 关键点 B：按照新版 Trainer.__init__ 的强制要求传参
    # 你在 A6000 上的旧版代码可能只需要传 cfg，但现在不行了
    trainer = SAM3DFlowMatchingTrainer(
        cfg=trainer_cfg,              # 只包含优化器和调度器配置
        models={"denoiser": model},
        dataset=dataset,
        dataloaders={"train": dataloader},
        output_dir=args.output_dir,   # 显式传入输出目录
        batch_size=args.batch_size,   # 显式传入 Batch Size
        max_steps=args.num_steps,     # 显式传入最大步数
        load_dir=None,                # 如果有预训练权重路径写在这
        step=0,
        device_mesh=None 
    )

    # 4. 训练循环 (适配 H100 BF16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    train_iter = cycle(dataloader)
    scaler = torch.cuda.amp.GradScaler(enabled=not is_h100) 

    print(">>> Starting Distillation Training...")
    for step in range(1, args.num_steps + 1):
        batch = next(train_iter)
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16 if is_h100 else torch.float16):
            losses, _ = trainer.training_losses(batch)
            loss = losses["loss"]
        
        optimizer.zero_grad()
        if is_h100:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if step % 10 == 0:
            print(f"[Step {step:04d}/{args.num_steps}] Loss: {loss.item():.6f}")

        if step % 500 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_{step:06d}.pt"))

    print(">>> Training finished!")

if __name__ == "__main__":
    main()