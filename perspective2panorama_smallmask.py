import math
import numpy as np
import torch
import torch.nn.functional as F


def _round_to_multiple(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


class Perspective2PanoramaSmallMask:
    """
    Variant of Perspective2Panorama that makes the HOLE/MASK smaller
    WITHOUT crop+resize. It does this by changing the projection scale
    (effective focal length), so the source fills more of the pano canvas directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),

                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.01}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -89.9, "max": 89.9, "step": 0.01}),
                "vfov": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 179.0, "step": 0.01}),

                # Canvas sizing
                "auto_canvas": ("BOOLEAN", {"default": False}),
                "canvas_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.01}),
                "pano_width": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 64}),
                "pano_height": ("INT", {"default": 1024, "min": 256, "max": 16384, "step": 64}),

                # THIS is the main knob:
                # < 1.0 => content appears bigger in the pano => mask/hole becomes smaller
                # > 1.0 => content appears smaller => mask/hole becomes bigger
                "projection_scale": ("FLOAT", {"default": 0.70, "min": 0.20, "max": 2.00, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("panorama_images", "mask", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(
        self,
        images,
        roll,
        pitch,
        vfov,
        auto_canvas,
        canvas_scale,
        pano_width,
        pano_height,
        projection_scale,
    ):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        device = images.device
        dtype = images.dtype

        B, H, W, C = images.shape

        # Output canvas size
        if bool(auto_canvas):
            pano_w = _round_to_multiple(int(W * float(canvas_scale)), 64)
            pano_h = _round_to_multiple(int(H * float(canvas_scale)), 64)
        else:
            pano_w = int(pano_width)
            pano_h = int(pano_height)

        # Build pano lon/lat grid (numpy -> torch)
        ys = np.linspace(0, pano_h - 1, pano_h, dtype=np.float32)
        xs = np.linspace(0, pano_w - 1, pano_w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)

        lon = (xv / pano_w) * (2.0 * np.pi) - np.pi
        lat = (yv / pano_h) * np.pi - (np.pi / 2.0)

        cos_lat = np.cos(lat)
        dirs = np.stack(
            [
                cos_lat * np.sin(lon),
                np.sin(lat),
                cos_lat * np.cos(lon),
            ],
            axis=-1,
        ).astype(np.float32)  # [pano_h,pano_w,3]

        # Rotation (roll Z, pitch X)
        rz = np.deg2rad(float(roll))
        rx = np.deg2rad(float(pitch))
        cz, sz = np.cos(rz), np.sin(rz)
        cx, sx = np.cos(rx), np.sin(rx)

        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        R = (Rx @ Rz).astype(np.float32)

        # Intrinsics from vertical FOV + image height
        vfov_rad = np.deg2rad(float(vfov))
        focal = H / (2.0 * np.tan(vfov_rad / 2.0))

        # projection_scale is the "mask shrink" knob:
        # smaller => focal smaller => more pano directions map into the image => smaller hole/mask
        focal = focal * float(projection_scale)

        cx0 = (W - 1) / 2.0
        cy0 = (H - 1) / 2.0

        # Rotate dirs into camera coords
        vcam = dirs @ R.T
        x = vcam[..., 0]
        y = vcam[..., 1]
        z = vcam[..., 2]

        eps = 1e-6
        valid = z > eps

        u = focal * (x / (z + eps)) + cx0
        v = focal * (y / (z + eps)) + cy0

        in_bounds = (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))
        content_mask = valid & in_bounds  # True where we have real pixels

        # Build grid_sample grid in normalized coords [-1, 1]
        # grid[...,0] = x(u), grid[...,1] = y(v)
        # align_corners=True makes mapping exact for pixel coordinates when normalizing with (W-1)/(H-1)
        u_norm = (u / (W - 1)) * 2.0 - 1.0
        v_norm = (v / (H - 1)) * 2.0 - 1.0

        grid = np.stack([u_norm, v_norm], axis=-1).astype(np.float32)  # [pano_h,pano_w,2]
        grid_t = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
        grid_t = grid_t.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,pano_h,pano_w,2]

        # Prepare input for grid_sample: IMAGE is [B,H,W,3] -> [B,3,H,W]
        img = images.detach()
        if img.dtype != torch.float32:
            img = img.float()
        img_nchw = img.permute(0, 3, 1, 2).clamp(0, 1)

        # Sample (no resizing of the source, just remapping into pano canvas)
        pano_nchw = F.grid_sample(
            img_nchw,
            grid_t,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # [B,3,pano_h,pano_w]

        pano = pano_nchw.permute(0, 2, 3, 1).to(device=device, dtype=dtype).clamp(0, 1)

        # Hole mask: white = to inpaint/fill, black = keep
        hole_mask = (~content_mask).astype(np.float32)  # [pano_h,pano_w]
        hole_mask_b = np.repeat(hole_mask[None, ...], B, axis=0)  # [B,pano_h,pano_w]
        mask_t = torch.from_numpy(hole_mask_b).to(device=device, dtype=torch.float32)

        info = (
            f"OK: input={W}x{H} roll={roll:.2f} pitch={pitch:.2f} vfov={vfov:.2f} "
            f"output={pano_w}x{pano_h} auto_canvas={bool(auto_canvas)} scale={float(canvas_scale):.2f} "
            f"projection_scale={float(projection_scale):.2f} (lower = bigger center, smaller mask)"
        )

        return (pano, mask_t, info)


NODE_CLASS_MAPPINGS = {
    "Perspective2PanoramaSmallMask": Perspective2PanoramaSmallMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Perspective2PanoramaSmallMask": "Perspective → Panorama (Small Mask, No Resize)",
}
