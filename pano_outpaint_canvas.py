import numpy as np
import torch
import torch.nn.functional as F


def _round_to_multiple(x: int, multiple: int):
    return int(np.ceil(max(1, int(x)) / multiple) * multiple)


def _base_target_size(vr_format: str, longest_side: int):
    side = _round_to_multiple(longest_side, 64)
    if vr_format in ("padded_360_equirect_2_1", "panorama_2_1"):
        return side, _round_to_multiple(int(np.ceil(side / 2.0)), 64)
    return side, side


def _scaled_target_size(base_w: int, base_h: int, outpaint_scale: float):
    scale = max(1.0, float(outpaint_scale))
    return _round_to_multiple(int(base_w * scale), 64), _round_to_multiple(int(base_h * scale), 64)


def _padding_mask(batch, out_h, out_w, left, top, resized_w, resized_h, feather_pixels, device):
    feather_pixels = max(0, int(feather_pixels))
    mask = torch.ones(
        (batch, out_h, out_w),
        dtype=torch.float32,
        device=device,
    )

    right = left + resized_w
    bottom = top + resized_h
    mask[:, top:bottom, left:right] = 0.0
    if feather_pixels <= 0:
        return mask

    xs = torch.arange(out_w, dtype=torch.float32, device=device).view(1, -1)
    ys = torch.arange(out_h, dtype=torch.float32, device=device).view(-1, 1)
    distance_to_edge = torch.minimum(
        torch.minimum(xs - left, (right - 1) - xs),
        torch.minimum(ys - top, (bottom - 1) - ys),
    )

    inside = distance_to_edge >= 0
    inner_feather = torch.clamp(1.0 - distance_to_edge / float(feather_pixels), 0.0, 1.0)
    feather = torch.where(inside, inner_feather, torch.ones_like(inner_feather))
    return feather.unsqueeze(0).repeat(batch, 1, 1)


class PanoOutpaintCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vr_format": (
                    ["vr180_equirect_1_1", "padded_360_equirect_2_1"],
                    {"default": "vr180_equirect_1_1"},
                ),
                "longest_side": ("INT", {"default": 704, "min": 256, "max": 16384, "step": 64}),
                "source_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "outpaint_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("canvas_images", "padding_mask", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(self, images, vr_format, longest_side, source_scale, outpaint_scale=2.0, mask_feather=0):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        batch, in_h, in_w, _ = images.shape
        base_w, base_h = _base_target_size(vr_format, longest_side)
        out_w, out_h = _scaled_target_size(base_w, base_h, outpaint_scale)

        # Fit the source once into the ratio-locked base canvas, then only grow
        # the gray outpaint area around it.
        fit = min(base_w / in_w, base_h / in_h) * float(source_scale)
        fit = max(fit, 1e-6)
        resized_w = max(1, min(out_w, int(round(in_w * fit))))
        resized_h = max(1, min(out_h, int(round(in_h * fit))))
        left = (out_w - resized_w) // 2
        top = (out_h - resized_h) // 2

        resized = F.interpolate(
            images.movedim(-1, 1),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

        canvas = torch.full(
            (batch, out_h, out_w, 3),
            0.5,
            dtype=images.dtype,
            device=images.device,
        )
        canvas[:, top : top + resized_h, left : left + resized_w, :] = resized

        mask = _padding_mask(
            batch,
            out_h,
            out_w,
            left,
            top,
            resized_w,
            resized_h,
            mask_feather,
            images.device,
        )

        info = (
            f"OK: input={in_w}x{in_h} format={vr_format} "
            f"base={base_w}x{base_h} canvas={out_w}x{out_h} source={resized_w}x{resized_h} "
            f"offset=({left},{top}) source_scale={float(source_scale):.2f} "
            f"outpaint_scale={float(outpaint_scale):.2f} mask_feather={int(mask_feather)} padding=gray"
        )
        return (canvas.clamp(0, 1), mask, info)


NODE_CLASS_MAPPINGS = {"PanoOutpaintCanvas": PanoOutpaintCanvas}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoOutpaintCanvas": "Pano Outpaint Canvas"}
