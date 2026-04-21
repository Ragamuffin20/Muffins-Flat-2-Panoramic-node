import torch
import torch.nn.functional as F


class MaskedOutpaintGuideFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "fill_mode": (["edge_spread", "neutral_gray"], {"default": "edge_spread"}),
                "iterations": ("INT", {"default": 96, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(self, images, mask, fill_mode, iterations):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        B, H, W, C = images.shape
        device = images.device
        dtype = images.dtype

        if mask.ndim == 4:
            mask = mask[..., 0]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1)
        mask = mask[:B].to(device=device, dtype=torch.float32)

        if mask.shape[1] != H or mask.shape[2] != W:
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)

        hole = (mask > 0.5).unsqueeze(1)
        img = images.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).clamp(0, 1)

        if fill_mode == "neutral_gray":
            filled = torch.where(hole, torch.full_like(img, 0.5), img)
            return (filled.permute(0, 2, 3, 1).to(dtype=dtype),)

        valid = (~hole).to(dtype=torch.float32)
        filled = img.clone()
        kernel = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
        rgb_kernel = kernel.repeat(3, 1, 1, 1)

        for _ in range(int(iterations)):
            neighbor_count = F.conv2d(valid, kernel, padding=1)
            can_fill = (valid < 0.5) & (neighbor_count > 0)
            if not bool(can_fill.any()):
                break

            neighbor_sum = F.conv2d(filled * valid, rgb_kernel, padding=1, groups=3)
            avg = neighbor_sum / neighbor_count.clamp_min(1e-6)
            filled = torch.where(can_fill.expand(-1, 3, -1, -1), avg, filled)
            valid = torch.where(can_fill, torch.ones_like(valid), valid)

        if bool((valid < 0.5).any()):
            filled = torch.where((valid < 0.5).expand(-1, 3, -1, -1), torch.full_like(filled, 0.5), filled)

        return (filled.permute(0, 2, 3, 1).to(dtype=dtype).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {"MaskedOutpaintGuideFill": MaskedOutpaintGuideFill}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskedOutpaintGuideFill": "Masked Outpaint Guide Fill"}
