import numpy as np
import torch


def _round_to_multiple(x: int, m: int) -> int:
    return int(np.ceil(x / m) * m)


class Perspective2Panorama180_NoBlack:
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
                # If auto_canvas=True, output size = input_size * canvas_scale (rounded up to /64)
                "canvas_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.01}),
                # Manual size (used when auto_canvas=False)
                "pano_width": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 64}),
                "pano_height": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 64}),

                # Optional: make the projected content bigger + mask smaller:
                # crop to the "in-bounds" bbox (plus margin), then optionally resize back to pano size
                "tight_crop": ("BOOLEAN", {"default": True}),
                "crop_margin": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_mode": (["crop_and_resize", "crop_only"], {"default": "crop_and_resize"}),
            }
        }

    # Still outputs a real ComfyUI MASK type (green port), but the panorama is *filled everywhere*
    # so you don't get black holes baked into the pixels.
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
        tight_crop,
        crop_margin,
        crop_mode,
    ):
        # Basic validation
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        B, H, W, C = images.shape

        # Output canvas size
        if bool(auto_canvas):
            pano_w = _round_to_multiple(int(W * float(canvas_scale)), 64)
            pano_h = _round_to_multiple(int(H * float(canvas_scale)), 64)
        else:
            pano_w = int(pano_width)
            pano_h = int(pano_height)

        # 180° pano for hemisphere: force square output (1:1)
        side = int(max(pano_w, pano_h))
        pano_w = side
        pano_h = side

        # Convert to numpy uint8
        imgs = images.detach().float().clamp(0, 1).cpu().numpy()
        frames = (imgs * 255.0).astype(np.uint8)

        # Precompute 1:1 180° hemisphere sphere dirs (square canvas)
        ys = np.linspace(0, pano_h - 1, pano_h, dtype=np.float32)
        xs = np.linspace(0, pano_w - 1, pano_w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)

        # lon, lat in [-π/2, +π/2]
        lon = (xv / pano_w) * (np.pi) - (np.pi / 2.0)
        lat = (yv / pano_h) * (np.pi) - (np.pi / 2.0)

        cos_lat = np.cos(lat)
        dirs = np.stack(
            [
                cos_lat * np.sin(lon),
                np.sin(lat),
                cos_lat * np.cos(lon),
            ],
            axis=-1,
        ).astype(np.float32)  # [pano_h,pano_w,3]

        # Rotation (roll Z, pitch X, yaw fixed 0)
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
        cx0 = W / 2.0
        cy0 = H / 2.0

        # Rotate dirs into camera coords
        vcam = dirs @ R.T
        x = vcam[..., 0]
        y = vcam[..., 1]
        z = vcam[..., 2]

        eps = 1e-6
        # Hemisphere should be facing forward (z>=0); keep a tiny epsilon for numerical safety
        valid = z > eps

        u = focal * (x / (z + eps)) + cx0
        v = focal * (y / (z + eps)) + cy0

        # "in_bounds" tells you where the pano samples *truly* map inside the input image.
        # We'll still output this as a mask, but we will NOT bake black pixels into the pano.
        in_bounds = (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))
        content_mask = valid & in_bounds

        # Clamp sampling coords so EVERY pano pixel is filled (edge-stretch instead of black)
        ui = np.clip(np.rint(u).astype(np.int32), 0, W - 1)
        vi = np.clip(np.rint(v).astype(np.int32), 0, H - 1)

        panos = []
        for i in range(B):
            pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
            # Fill all valid hemisphere pixels with clamped samples (no black holes)
            pano[valid] = frames[i][vi[valid], ui[valid]]
            panos.append(pano)

        pano_np = (np.stack(panos, axis=0).astype(np.float32) / 255.0)
        pano_t = torch.from_numpy(pano_np).to(images.device, dtype=images.dtype)

        # Mask still indicates "holes" (white = needs fill, black = keep)
        hole_mask = (~content_mask).astype(np.float32)  # [H,W]
        hole_mask_b = np.repeat(hole_mask[None, ...], B, axis=0)  # [B,H,W]
        mask_t = torch.from_numpy(hole_mask_b).to(images.device, dtype=torch.float32)

        # --- Optional tight crop to shrink mask area (based on in-bounds content) ---
        if bool(tight_crop):
            ys2, xs2 = np.where(content_mask)
            if ys2.size > 0 and xs2.size > 0:
                y0, y1 = int(ys2.min()), int(ys2.max())
                x0, x1 = int(xs2.min()), int(xs2.max())

                bh = max(1, y1 - y0 + 1)
                bw = max(1, x1 - x0 + 1)
                my = int(bh * float(crop_margin))
                mx = int(bw * float(crop_margin))

                y0 = max(0, y0 - my)
                y1 = min(pano_h - 1, y1 + my)
                x0 = max(0, x0 - mx)
                x1 = min(pano_w - 1, x1 + mx)

                pano_crop = pano_t[:, y0:y1 + 1, x0:x1 + 1, :]
                mask_crop = mask_t[:, y0:y1 + 1, x0:x1 + 1]

                if crop_mode == "crop_and_resize":
                    target_h, target_w = pano_h, pano_w

                    # IMAGE resize (expects NCHW)
                    pano_nchw = pano_crop.permute(0, 3, 1, 2)
                    pano_nchw = torch.nn.functional.interpolate(
                        pano_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                    pano_t = pano_nchw.permute(0, 2, 3, 1)

                    # MASK resize (N1HW)
                    mask_n1hw = mask_crop.unsqueeze(1)
                    mask_n1hw = torch.nn.functional.interpolate(
                        mask_n1hw, size=(target_h, target_w), mode="nearest"
                    )
                    mask_t = mask_n1hw[:, 0, :, :]
                else:
                    pano_t = pano_crop
                    mask_t = mask_crop

        info = (
            f"OK (no-black): input={W}x{H} roll={roll:.2f} pitch={pitch:.2f} vfov={vfov:.2f} "
            f"output={pano_w}x{pano_h} auto_canvas={bool(auto_canvas)} scale={float(canvas_scale):.2f} "
            f"tight_crop={bool(tight_crop)} mode={crop_mode} margin={float(crop_margin):.2f}"
        )
        return (pano_t, mask_t, info)


NODE_CLASS_MAPPINGS = {"Perspective2Panorama180_NoBlack": Perspective2Panorama180_NoBlack}
NODE_DISPLAY_NAME_MAPPINGS = {"Perspective2Panorama180_NoBlack": "Perspective → 180° Pano (No Black Fill)"}