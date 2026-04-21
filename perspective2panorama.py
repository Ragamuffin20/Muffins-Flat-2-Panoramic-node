import numpy as np
import torch

def _round_to_multiple(x: int, m: int) -> int:
    return int(np.ceil(x / m) * m)


def _resolve_output_size(
    auto_canvas: bool,
    canvas_scale: float,
    pano_width: int,
    pano_height: int,
    input_width: int,
    input_height: int,
    output_projection: str,
):
    if bool(auto_canvas):
        raw_w = _round_to_multiple(int(input_width * float(canvas_scale)), 64)
        raw_h = _round_to_multiple(int(input_height * float(canvas_scale)), 64)
    else:
        raw_w = int(pano_width)
        raw_h = int(pano_height)

    if output_projection == "fisheye_1_1":
        side = int(max(raw_w, raw_h))
        return side, side

    target_h = int(max(raw_h, int(np.ceil(raw_w / 2.0))))
    return target_h * 2, target_h


def _build_output_dirs(pano_w: int, pano_h: int, output_projection: str):
    ys = np.linspace(0, pano_h - 1, pano_h, dtype=np.float32)
    xs = np.linspace(0, pano_w - 1, pano_w, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    if output_projection == "fisheye_1_1":
        cx = (pano_w - 1) / 2.0
        cy = (pano_h - 1) / 2.0
        radius = max(1.0, min(pano_w, pano_h) - 1) / 2.0

        nx = (xv - cx) / radius
        ny = (yv - cy) / radius
        r = np.sqrt(nx * nx + ny * ny)
        lens_valid = r <= 1.0

        # Equidistant fisheye over the forward hemisphere.
        theta = np.clip(r, 0.0, 1.0) * (np.pi / 2.0)
        phi = np.arctan2(ny, nx)
        sin_theta = np.sin(theta)

        dirs = np.stack(
            [
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)

        return dirs, lens_valid

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
    ).astype(np.float32)

    return dirs, np.ones((pano_h, pano_w), dtype=bool)


def _fill_holes_for_conditioning(panos, hole_mask):
    if not np.any(hole_mask):
        return panos

    valid_mask = ~hole_mask
    filled = []
    for pano in panos:
        fallback = pano.copy()
        valid_pixels = pano[valid_mask]
        if valid_pixels.size:
            fill_color = np.median(valid_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
        else:
            fill_color = np.array([127, 127, 127], dtype=np.uint8)
        fallback[hole_mask] = fill_color
        filled.append(fallback)
    return filled


class Perspective2Panorama:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.01}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -89.9, "max": 89.9, "step": 0.01}),
                "vfov": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 179.0, "step": 0.01}),
                "output_projection": (
                    ["fisheye_1_1", "panorama_2_1"],
                    {"default": "fisheye_1_1"},
                ),
                "auto_canvas": ("BOOLEAN", {"default": False}),
                "canvas_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.01}),
                "pano_width": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 64}),
                "pano_height": ("INT", {"default": 1024, "min": 256, "max": 16384, "step": 64}),
                "tight_crop": ("BOOLEAN", {"default": True}),
                "crop_margin": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_mode": (["crop_and_resize", "crop_only"], {"default": "crop_and_resize"}),
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
        output_projection,
        auto_canvas,
        canvas_scale,
        pano_width,
        pano_height,
        tight_crop,
        crop_margin,
        crop_mode,
    ):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        B, H, W, C = images.shape

        pano_w, pano_h = _resolve_output_size(
            auto_canvas=bool(auto_canvas),
            canvas_scale=float(canvas_scale),
            pano_width=int(pano_width),
            pano_height=int(pano_height),
            input_width=W,
            input_height=H,
            output_projection=output_projection,
        )

        imgs = images.detach().float().clamp(0, 1).cpu().numpy()
        frames = (imgs * 255.0).astype(np.uint8)

        dirs, output_valid = _build_output_dirs(
            pano_w=pano_w,
            pano_h=pano_h,
            output_projection=output_projection,
        )

        rz = np.deg2rad(float(roll))
        rx = np.deg2rad(float(pitch))

        cz, sz = np.cos(rz), np.sin(rz)
        cx, sx = np.cos(rx), np.sin(rx)

        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        R = (Rx @ Rz).astype(np.float32)

        vfov_rad = np.deg2rad(float(vfov))
        focal = H / (2.0 * np.tan(vfov_rad / 2.0))
        cx0 = W / 2.0
        cy0 = H / 2.0

        vcam = dirs @ R.T
        x = vcam[..., 0]
        y = vcam[..., 1]
        z = vcam[..., 2]

        eps = 1e-6
        valid = output_valid & (z > eps)

        u = focal * (x / (z + eps)) + cx0
        v = focal * (y / (z + eps)) + cy0

        in_bounds = (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))
        content_mask = valid & in_bounds

        ui = np.clip(np.rint(u).astype(np.int32), 0, W - 1)
        vi = np.clip(np.rint(v).astype(np.int32), 0, H - 1)

        panos = []
        for i in range(B):
            pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
            pano[content_mask] = frames[i][vi[content_mask], ui[content_mask]]
            panos.append(pano)

        pano_np = np.stack(panos, axis=0).astype(np.float32) / 255.0
        pano_t = torch.from_numpy(pano_np).to(images.device, dtype=images.dtype)

        hole_mask = (~content_mask).astype(np.float32)
        hole_mask_b = np.repeat(hole_mask[None, ...], B, axis=0)
        mask_t = torch.from_numpy(hole_mask_b).to(images.device, dtype=torch.float32)

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

                    pano_nchw = pano_crop.permute(0, 3, 1, 2)
                    pano_nchw = torch.nn.functional.interpolate(
                        pano_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                    pano_t = pano_nchw.permute(0, 2, 3, 1)

                    mask_n1hw = mask_crop.unsqueeze(1)
                    mask_n1hw = torch.nn.functional.interpolate(
                        mask_n1hw, size=(target_h, target_w), mode="nearest"
                    )
                    mask_t = mask_n1hw[:, 0, :, :]
                else:
                    pano_t = pano_crop
                    mask_t = mask_crop

        info = (
            f"OK: input={W}x{H} roll={roll:.2f} pitch={pitch:.2f} vfov={vfov:.2f} "
            f"projection={output_projection} output={pano_w}x{pano_h} "
            f"auto_canvas={bool(auto_canvas)} scale={float(canvas_scale):.2f} "
            f"tight_crop={bool(tight_crop)} mode={crop_mode} margin={float(crop_margin):.2f}"
        )
        return (pano_t, mask_t, info)


NODE_CLASS_MAPPINGS = {"Perspective2Panorama": Perspective2Panorama}
NODE_DISPLAY_NAME_MAPPINGS = {"Perspective2Panorama": "Perspective -> Panorama (Universal)"}
