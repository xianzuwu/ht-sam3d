import os
import torch
import numpy as np
import glob  # ✅ 必须导入
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SAM3DDistillDataset(Dataset):
    def __init__(self, token_dir, image_dir, image_size=518):
        self.token_dir = token_dir
        self.image_dir = image_dir
        self.image_size = image_size
        
        # [H100-FIX] 使用 glob 递归搜索所有子目录下的 .pt 文件
        search_path = os.path.join(token_dir, "**/*.pt")
        # glob 返回的是完整路径
        self.token_files = glob.glob(search_path, recursive=True)
        self.token_files.sort()
        
        if len(self.token_files) == 0:
            raise FileNotFoundError(f"No .pt files found in {token_dir} or its subdirectories!")
            
        print(f"[Dataset] Found {len(self.token_files)} GT token files.")
        
        self.loads = np.ones(len(self.token_files), dtype=np.float32)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.token_files)

    def __getitem__(self, idx):
        # 1. 直接获取完整路径，不再用 os.path.join 拼接
        token_path = self.token_files[idx]
        
        # 加载预提取的 GT Tokens
        gt_data = torch.load(token_path, map_location='cpu')
        
        # 2. 寻找对应的图片逻辑优化
        # 假设 token 路径是: .../gt_tokens/FOLDER_NAME/mask_x.pt
        # 对应的图片路径是: .../images/FOLDER_NAME/image.png
        
        # 获取子文件夹名称（例如 shutterstock_stylish_kidsroom_1640806567）
        scene_folder = os.path.basename(os.path.dirname(token_path))
        
        # 定义可能的图片文件名 (通常是 image.png 或与文件夹同名)
        potential_img_names = ["image.png", "image.jpg", f"{scene_folder}.png", f"{scene_folder}.jpg"]
        
        img_path = None
        for name in potential_img_names:
            test_path = os.path.join(self.image_dir, scene_folder, name)
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        if img_path is None:
            # 回退方案：如果子文件夹里没找到，尝试你原来的逻辑（找 mask_x.png）
            token_filename = os.path.basename(token_path)
            img_path = os.path.join(self.image_dir, scene_folder, token_filename.replace('.pt', '.png'))
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Could not find matching image for {token_path} in {self.image_dir}")
            
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        
        return {
            "cond_projs": image_tensor,
            "gt_tokens": gt_data['tokens'],
            "indices": gt_data['indices']
        }