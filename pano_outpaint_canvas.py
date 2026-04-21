import numpy as np
import torch
import torch.nn.functional as F


def _round_to_multiple(x: int, multiple: int):
    return int(np.ceil(max(1, int(x)) / multiple) * multiple)


def _base_target_size(output_projection: str, canvas_width: int, canvas_height: int):
    if output_projection == "fisheye_1_1":
        side = _round_to_multiple(max(int(canvas_width), int(canvas_height)), 64)
        return side, side

    height = max(int(canvas_height), int(np.ceil(int(canvas_width) / 2.0)))
    height = _round_to_multiple(height, 64)
    return height * 2, height


def _scaled_target_size(base_w: int, base_h: int, outpaint_scale: float):
    scale = max(1.0, float(outpaint_scale))
    return _round_to_multiple(int(base_w * scale), 64), _round_to_multiple(int(base_h * scale), 64)


class PanoOutpaintCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_projection": (
                    ["panorama_2_1", "fisheye_1_1"],
                    {"default": "panorama_2_1"},
                ),
                "canvas_width": ("INT", {"default": 1280, "min": 256, "max": 16384, "step": 64}),
                "canvas_height": ("INT", {"default": 640, "min": 256, "max": 16384, "step": 64}),
                "source_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "outpaint_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("canvas_images", "padding_mask", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(self, images, output_projection, canvas_width, canvas_height, source_scale, outpaint_scale=1.0):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        batch, in_h, in_w, _ = images.shape
        base_w, base_h = _base_target_size(output_projection, canvas_width, canvas_height)
        out_w, out_h = _scaled_target_size(base_w, base_h, outpaint_scale)

        # Fit the source to the unscaled base canvas first. outpaint_scale then
        # grows the black outpaint area around that fitted source instead of
        # enlarging the source along with the canvas.
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

        canvas = torch.zeros(
            (batch, out_h, out_w, 3),
            dtype=images.dtype,
            device=images.device,
        )
        canvas[:, top : top + resized_h, left : left + resized_w, :] = resized

        mask = torch.ones(
            (batch, out_h, out_w),
            dtype=torch.float32,
            device=images.device,
        )
        mask[:, top : top + resized_h, left : left + resized_w] = 0.0

        info = (
            f"OK: input={in_w}x{in_h} projection={output_projection} "
            f"base={base_w}x{base_h} canvas={out_w}x{out_h} source={resized_w}x{resized_h} "
            f"offset=({left},{top}) source_scale={float(source_scale):.2f} "
            f"outpaint_scale={float(outpaint_scale):.2f}"
        )
        return (canvas.clamp(0, 1), mask, info)


NODE_CLASS_MAPPINGS = {"PanoOutpaintCanvas": PanoOutpaintCanvas}
NODE_DISPLAY_NAME_MAPPINGS = {"PanoOutpaintCanvas": "Pano Outpaint Canvas"}
