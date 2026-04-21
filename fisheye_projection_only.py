import numpy as np
import torch


def _bilinear_sample(frame, u, v):
    h, w, _ = frame.shape

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)

    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]

    top = frame[v0, u0] * (1.0 - du) + frame[v0, u1] * du
    bottom = frame[v1, u0] * (1.0 - du) + frame[v1, u1] * du
    return top * (1.0 - dv) + bottom * dv


class FisheyeProjectionOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.01}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -89.9, "max": 89.9, "step": 0.01}),
                "vfov": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 179.0, "step": 0.01}),
                "use_input_size": ("BOOLEAN", {"default": True}),
                "output_side": ("INT", {"default": 1024, "min": 256, "max": 16384, "step": 64}),
                "fill_mode": (["edge", "black"], {"default": "edge"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("fisheye_images", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(self, images, roll, pitch, vfov, use_input_size, output_side, fill_mode):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        batch, in_h, in_w, _ = images.shape
        side = max(int(in_w), int(in_h)) if bool(use_input_size) else int(output_side)
        side = max(1, side)

        ys = np.linspace(0, side - 1, side, dtype=np.float32)
        xs = np.linspace(0, side - 1, side, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)

        cx = (side - 1) / 2.0
        cy = (side - 1) / 2.0
        radius = max(1.0, side - 1) / 2.0

        nx = (xv - cx) / radius
        ny = (yv - cy) / radius
        r = np.sqrt(nx * nx + ny * ny)
        inside_lens = r <= 1.0

        # This node is projection-only for a second pass. With edge fill, pixels
        # outside the fisheye circle are clamped to the horizon instead of being
        # turned into a new black padding region.
        r_for_theta = np.clip(r, 0.0, 1.0)
        theta = r_for_theta * (np.pi / 2.0)
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

        rz = np.deg2rad(float(roll))
        rx = np.deg2rad(float(pitch))

        cz, sz = np.cos(rz), np.sin(rz)
        cxr, sxr = np.cos(rx), np.sin(rx)

        rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rx_mat = np.array([[1, 0, 0], [0, cxr, -sxr], [0, sxr, cxr]], dtype=np.float32)
        rot = (rx_mat @ rz_mat).astype(np.float32)

        vfov_rad = np.deg2rad(float(vfov))
        focal = in_h / (2.0 * np.tan(vfov_rad / 2.0))
        input_cx = in_w / 2.0
        input_cy = in_h / 2.0

        camera_dirs = dirs @ rot.T
        x = camera_dirs[..., 0]
        y = camera_dirs[..., 1]
        z = camera_dirs[..., 2]

        eps = 1e-6
        in_front = z > eps
        u = focal * (x / (z + eps)) + input_cx
        v = focal * (y / (z + eps)) + input_cy

        in_bounds = (u >= 0) & (u <= (in_w - 1)) & (v >= 0) & (v <= (in_h - 1))
        valid = in_front & in_bounds
        if fill_mode == "black":
            valid = valid & inside_lens

        u = np.clip(u, 0, in_w - 1)
        v = np.clip(v, 0, in_h - 1)

        frames = images.detach().float().clamp(0, 1).cpu().numpy()
        output = []
        for i in range(batch):
            sampled = _bilinear_sample(frames[i], u, v).astype(np.float32)
            if fill_mode == "black":
                sampled[~valid] = 0.0
            output.append(sampled)

        out_np = np.stack(output, axis=0)
        out_t = torch.from_numpy(out_np).to(images.device, dtype=images.dtype).clamp(0, 1)

        info = (
            f"OK projection-only fisheye: input={in_w}x{in_h} output={side}x{side} "
            f"roll={float(roll):.2f} pitch={float(pitch):.2f} vfov={float(vfov):.2f} "
            f"use_input_size={bool(use_input_size)} fill_mode={fill_mode}"
        )
        return (out_t, info)


NODE_CLASS_MAPPINGS = {"FisheyeProjectionOnly": FisheyeProjectionOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"FisheyeProjectionOnly": "Fisheye Projection Only"}


class FisheyeLensWarpOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rectilinear_fov": ("FLOAT", {"default": 150.0, "min": 60.0, "max": 178.0, "step": 0.5}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "lens_radius": ("FLOAT", {"default": 0.98, "min": 0.25, "max": 1.5, "step": 0.01}),
                "edge_fade": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "vignette": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("fisheye_images", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(
        self,
        images,
        strength,
        rectilinear_fov,
        zoom,
        center_x,
        center_y,
        lens_radius,
        edge_fade,
        vignette,
    ):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        batch, h, w, _ = images.shape
        cx = float(center_x) * (w - 1)
        cy = float(center_y) * (h - 1)
        radius = max(1.0, min(w, h) * 0.5 * float(lens_radius))

        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)

        nx = (xv - cx) / radius
        ny = (yv - cy) / radius
        r = np.sqrt(nx * nx + ny * ny)
        angle = np.arctan2(ny, nx)

        max_theta = np.deg2rad(float(rectilinear_fov) * 0.5)
        max_theta = np.clip(max_theta, np.deg2rad(1.0), np.deg2rad(89.0))
        r_safe = np.clip(r, 0.0, 1.0)

        fisheye_r = np.tan(r_safe * max_theta) / np.tan(max_theta)
        src_r = (1.0 - float(strength)) * r_safe + float(strength) * fisheye_r
        src_r = src_r / max(float(zoom), 1e-6)

        src_x = cx + np.cos(angle) * src_r * radius
        src_y = cy + np.sin(angle) * src_r * radius
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        fade_start = max(0.0, 1.0 - float(edge_fade))
        if float(edge_fade) <= 0:
            edge_alpha = (r > 1.0).astype(np.float32)
        else:
            edge_alpha = np.clip((r - fade_start) / max(float(edge_fade), 1e-6), 0.0, 1.0)
            edge_alpha = edge_alpha * edge_alpha * (3.0 - 2.0 * edge_alpha)

        inner_vignette = np.clip(r_safe * r_safe * float(vignette), 0.0, 1.0)
        darken = np.clip(1.0 - inner_vignette, 0.0, 1.0)[..., None]
        edge_alpha = edge_alpha[..., None]

        frames = images.detach().float().clamp(0, 1).cpu().numpy()
        warped = []
        for i in range(batch):
            sampled = _bilinear_sample(frames[i], src_x, src_y).astype(np.float32)
            sampled = sampled * darken
            sampled = sampled * (1.0 - edge_alpha)
            warped.append(sampled)

        out_np = np.stack(warped, axis=0)
        out_t = torch.from_numpy(out_np).to(images.device, dtype=images.dtype).clamp(0, 1)
        info = (
            f"OK lens-warp only: input={w}x{h} output={w}x{h} "
            f"strength={float(strength):.2f} rectilinear_fov={float(rectilinear_fov):.1f} "
            f"zoom={float(zoom):.2f} center=({float(center_x):.3f},{float(center_y):.3f}) "
            f"lens_radius={float(lens_radius):.2f} edge_fade={float(edge_fade):.2f} "
            f"vignette={float(vignette):.2f}"
        )
        return (out_t, info)


NODE_CLASS_MAPPINGS["FisheyeLensWarpOnly"] = FisheyeLensWarpOnly
NODE_DISPLAY_NAME_MAPPINGS["FisheyeLensWarpOnly"] = "Fisheye Lens Warp Only"
