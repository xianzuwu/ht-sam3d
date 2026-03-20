import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, default_collate
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm


class SAM3DDistillDataset(Dataset):
    """
    按当前一阶段 ss 训练逻辑整理的数据集：
      - shape latent 不做标准化
      - scale 使用 torch.log
      - translation_scale 保留返回并使用 torch.log
        （按当前项目设定，trainer 中参与 loss）
    """

    def __init__(self, token_dir, image_dir, image_size=518):
        self.token_dir = token_dir
        self.image_dir = image_dir
        self.image_size = image_size

        search_path = os.path.join(token_dir, "**", "*.pt")
        self.token_files = glob.glob(search_path, recursive=True)
        self.token_files.sort()

        if len(self.token_files) == 0:
            raise FileNotFoundError(
                f"No .pt files found in {token_dir} or its subdirectories."
            )

        self.loads = np.ones(len(self.token_files), dtype=np.float32)
        self.value_range = (0.0, 1.0)

        print(f"[Dataset] Found {len(self.token_files)} GT token files.")
        self.stats = self._compute_stats()

    @staticmethod
    def collate_fn(batch, **kwargs):
        return default_collate(batch)

    def __len__(self):
        return len(self.token_files)

    @staticmethod
    def _maybe_squeeze_batch_dim(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim > 0 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x

    def _compute_stats(self):
        print(f"\n[Dataset] Scanning {len(self.token_files)} files to compute stats...")

        shape_accum = []
        scale_log_accum = []
        trans_scale_log_accum = []

        for token_path in tqdm(self.token_files, desc="Compute dataset stats"):
            try:
                gt_data = torch.load(token_path, map_location="cpu", weights_only=False)

                if "shape" in gt_data:
                    shape_accum.append(gt_data["shape"].float().flatten())

                if "scale" in gt_data:
                    scale = torch.clamp(gt_data["scale"].float().flatten(), min=1e-6)
                    scale_log_accum.append(torch.log(scale))

                if "translation_scale" in gt_data:
                    ts = torch.clamp(
                        gt_data["translation_scale"].float().flatten(), min=1e-6
                    )
                    trans_scale_log_accum.append(torch.log(ts))

            except Exception as e:
                print(f"[Dataset] Failed loading stats from {token_path}: {e}")

        stats = {}
        if len(shape_accum) > 0:
            all_shapes = torch.cat(shape_accum, dim=0)
            stats["shape_mean"] = all_shapes.mean().item()
            stats["shape_std"] = max(all_shapes.std().item(), 1e-6)

        if len(scale_log_accum) > 0:
            all_scale_log = torch.cat(scale_log_accum, dim=0)
            stats["scale_log_mean"] = all_scale_log.mean().item()
            stats["scale_log_std"] = max(all_scale_log.std().item(), 1e-6)

        if len(trans_scale_log_accum) > 0:
            all_ts_log = torch.cat(trans_scale_log_accum, dim=0)
            stats["ts_log_mean"] = all_ts_log.mean().item()
            stats["ts_log_std"] = max(all_ts_log.std().item(), 1e-6)

        print("[Dataset] Stats ready (for inspection only):")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")
        print()

        return stats

    def visualize_sample(self, sample):
        if isinstance(sample, dict):
            if "rgb_image" in sample:
                return {"input_image": sample["rgb_image"]}
            if "image" in sample:
                return {"input_image": sample["image"]}
        return {"input_image": torch.zeros(1, 3, self.image_size, self.image_size)}

    def get_crop_bbox(self, mask_np):
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)

        if not np.any(rows) or not np.any(cols):
            return 0, 0, mask_np.shape[1], mask_np.shape[0]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # PIL crop 右下边界是开区间，所以这里 +1
        return cmin, rmin, cmax + 1, rmax + 1

    def preprocess_image_tensor(self, pil_image):
        arr = np.array(pil_image)

        if arr.shape[-1] == 3:
            alpha = np.ones_like(arr[..., 0], dtype=np.uint8) * 255
            arr = np.dstack([arr, alpha])

        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        img = x[:3, ...]
        mask = x[3:4, ...]

        _, h, w = img.shape
        if h != w:
            diff = abs(h - w)
            p1 = diff // 2
            p2 = diff - p1
            padding = (p1, p2, 0, 0) if h > w else (0, 0, p1, p2)
            img = F.pad(img, padding, value=1.0)
            mask = F.pad(mask, padding, value=0.0)

        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        img = img * mask + (1 - mask)
        return img, mask

    def _find_image_path(self, token_path):
        token_filename = os.path.basename(token_path)
        scene_folder = os.path.basename(os.path.dirname(token_path))
        file_id = os.path.splitext(token_filename)[0]

        candidates = [
            os.path.join(self.image_dir, f"{file_id}.png"),
            os.path.join(self.image_dir, f"{file_id}.jpg"),
            os.path.join(self.image_dir, scene_folder, f"{file_id}.png"),
            os.path.join(self.image_dir, scene_folder, f"{file_id}.jpg"),
            os.path.join(self.image_dir, scene_folder, "image.png"),
            os.path.join(self.image_dir, scene_folder, "image.jpg"),
            os.path.join(self.image_dir, scene_folder, f"{scene_folder}.png"),
            os.path.join(self.image_dir, scene_folder, f"{scene_folder}.jpg"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Image not found for token {token_path}. Tried: {candidates}"
        )

    def __getitem__(self, idx):
        token_path = self.token_files[idx]
        gt_data = torch.load(token_path, map_location="cpu", weights_only=False)

        required = [
            "shape",
            "6drotation_normalized",
            "translation",
            "scale",
            "translation_scale",
        ]
        missing = [k for k in required if k not in gt_data]
        if missing:
            raise KeyError(
                f"Token file {token_path} missing keys: {missing}. "
                f"Available keys: {list(gt_data.keys())}"
            )

        # shape：一阶段 ss 不做标准化
        shape_raw = self._maybe_squeeze_batch_dim(gt_data["shape"])

        rot_token = self._maybe_squeeze_batch_dim(gt_data["6drotation_normalized"])
        trans_token = self._maybe_squeeze_batch_dim(gt_data["translation"])

        # scale：使用 log-space
        scale_raw = self._maybe_squeeze_batch_dim(gt_data["scale"])
        scale_token = torch.log(torch.clamp(scale_raw, min=1e-6))

        # translation_scale：保留并参与 loss，也使用 log-space
        t_scale_raw = self._maybe_squeeze_batch_dim(gt_data["translation_scale"])
        t_scale_token = torch.log(torch.clamp(t_scale_raw, min=1e-6))

        img_path = self._find_image_path(token_path)
        pil_image = Image.open(img_path).convert("RGBA")

        global_img_tensor, global_mask_tensor = self.preprocess_image_tensor(pil_image)

        mask_np = np.array(pil_image)[:, :, 3] > 128
        cmin, rmin, cmax, rmax = self.get_crop_bbox(mask_np)
        width, height = cmax - cmin, rmax - rmin
        pad = int(max(width, height) * 0.1)

        cmin = max(0, cmin - pad)
        rmin = max(0, rmin - pad)
        cmax = min(pil_image.width, cmax + pad)
        rmax = min(pil_image.height, rmax + pad)

        crop_pil = pil_image.crop((cmin, rmin, cmax, rmax))
        local_img_tensor, local_mask_tensor = self.preprocess_image_tensor(crop_pil)

        return {
            "x_0": shape_raw,
            "6drotation_normalized": rot_token,
            "translation": trans_token,
            "scale": scale_token,
            "translation_scale": t_scale_token,
            "image": local_img_tensor,
            "mask": local_mask_tensor,
            "rgb_image": global_img_tensor,
            "rgb_image_mask": global_mask_tensor,
        }