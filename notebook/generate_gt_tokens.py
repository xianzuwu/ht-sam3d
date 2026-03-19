import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

CACHE_DIR = "/data/L202500204/Projects/sam-3d-objects/.cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
os.environ["TORCH_HUB_DIR"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR

# === 核心修复：把项目根目录加入到 Python 环境变量中 ===
# 这样 inference.py 就能顺利找到外层的 sam3d_objects 包了
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # notebook/ 目录
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # trellis-sam-3d-objects/ 根目录

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 现在可以安全导入了
from inference import Inference

class GTTokenExtractor(Inference):
    """
    继承自 Inference 类。
    直接调用 Pipeline 内部组件获取原始 Log Space Latents 作为 GT。
    """
    def __init__(self, config_file: str, compile: bool = False):
        super().__init__(config_file, compile)
        # 确保所有模型处于 eval 模式
        if hasattr(self._pipeline, 'models'):
            for name, model in self._pipeline.models.items():
                if hasattr(model, 'eval'):
                    model.eval()

    @torch.no_grad()
    def extract_and_save_tokens(self, image_np, mask_np, output_path_base):
        """
        接收 numpy 数组格式的 image 和 mask，提取并保存 tokens
        """
        image_input = self.merge_mask_to_rgba(image_np, mask_np)

        try:
            # (A) 计算 Pointmap
            pointmap_dict = self._pipeline.compute_pointmap(image_input, pointmap=None)
            pointmap = pointmap_dict["pointmap"]
            
            # (B) 预处理
            ss_input_dict = self._pipeline.preprocess_image(
                image_input, 
                self._pipeline.ss_preprocessor, 
                pointmap=pointmap
            )
            
            # (C) 获取原始的 Generator 输出
            raw_results = self._pipeline.sample_sparse_structure(
                ss_input_dict,
                inference_steps=50,   
                use_distillation=False 
            )
            
            # 提取并保存所需的 GT 数据
            gt_data = {}
            target_keys = [
                "shape", 
                "6drotation_normalized", 
                "translation", 
                "scale", 
                "translation_scale"
            ]
            
            for k in target_keys:
                if k in raw_results:
                    gt_data[k] = raw_results[k].detach().cpu()
            
            if gt_data:
                save_path = f"{output_path_base}.pt"
                torch.save(gt_data, save_path)
                return True
            else:
                return False

        except Exception as e:
            print(f"[Error] Failed to process token extraction: {e}")
            return False

def process_scene(extractor, scene_dir, output_dir):
    """处理单个场景文件夹"""
    image_path = os.path.join(scene_dir, "image.png")
    if not os.path.exists(image_path):
        print(f"[Skip] No image.png found in {scene_dir}")
        return 0, 0

    print(f"\nProcessing scene: {os.path.basename(scene_dir)}")
    image_np = np.array(Image.open(image_path).convert("RGB"))
    
    mask_files = [f for f in os.listdir(scene_dir) if f.endswith('.png') and f[:-4].isdigit()]
    mask_files.sort(key=lambda x: int(x[:-4])) 
    
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    
    for mask_name in tqdm(mask_files, desc="Extracting Masks"):
        mask_path = os.path.join(scene_dir, mask_name)
        mask_np = np.array(Image.open(mask_path).convert("L")) > 0
        
        mask_id = mask_name[:-4] 
        output_base = os.path.join(output_dir, f"mask_{mask_id}") 
        
        if extractor.extract_and_save_tokens(image_np, mask_np, output_base):
            success_count += 1
            
    return success_count, len(mask_files)

def main():
    parser = argparse.ArgumentParser(description="Extract GT Tokens for Trellis")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing image.png and numbered masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pt files")
    args = parser.parse_args()

    print("Loading model pipeline...")
    # 注意：这里的路径要以当前运行脚本的相对位置为准
    extractor = GTTokenExtractor(config_file=args.config)
    
    if os.path.exists(os.path.join(args.input_dir, "image.png")):
        succ, total = process_scene(extractor, args.input_dir, args.output_dir)
        print(f"Done! Successfully extracted {succ}/{total} GT tokens.")
    else:
        scene_dirs = [os.path.join(args.input_dir, d) for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        total_succ, total_masks = 0, 0
        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)
            scene_out_dir = os.path.join(args.output_dir, scene_name)
            succ, total = process_scene(extractor, scene_dir, scene_out_dir)
            total_succ += succ
            total_masks += total
        print(f"All Done! Successfully extracted {total_succ}/{total_masks} GT tokens across {len(scene_dirs)} scenes.")

if __name__ == "__main__":
    main()