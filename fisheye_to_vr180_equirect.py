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


def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return (ry @ rx @ rz).astype(np.float32)


def _lens_radius(theta, max_theta, lens_model):
    theta = np.clip(theta, 0.0, max_theta)
    if lens_model == "equisolid":
        denom = max(np.sin(max_theta / 2.0), 1e-6)
        return np.sin(theta / 2.0) / denom
    if lens_model == "orthographic":
        denom = max(np.sin(max_theta), 1e-6)
        return np.sin(theta) / denom
    if lens_model == "stereographic":
        denom = max(np.tan(max_theta / 2.0), 1e-6)
        return np.tan(theta / 2.0) / denom
    return theta / max(max_theta, 1e-6)


class FisheyeToVR180Equirect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_mode": (
                    ["vr180_equirect_1_1", "padded_360_equirect_2_1"],
                    {"default": "vr180_equirect_1_1"},
                ),
                "use_input_size": ("BOOLEAN", {"default": True}),
                "output_height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "fisheye_fov": ("FLOAT", {"default": 180.0, "min": 90.0, "max": 220.0, "step": 0.1}),
                "lens_model": (
                    ["equidistant", "equisolid", "orthographic", "stereographic"],
                    {"default": "equidistant"},
                ),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "lens_radius": ("FLOAT", {"default": 0.98, "min": 0.25, "max": 1.5, "step": 0.001}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "fill_mode": (["black", "edge"], {"default": "black"}),
                "horizontal_flip": ("BOOLEAN", {"default": False}),
                "vertical_flip": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("vr180_equirect", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(
        self,
        images,
        output_mode,
        use_input_size,
        output_height,
        fisheye_fov,
        lens_model,
        center_x,
        center_y,
        lens_radius,
        yaw,
        pitch,
        roll,
        fill_mode,
        horizontal_flip,
        vertical_flip,
    ):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        batch, in_h, in_w, _ = images.shape
        out_h = max(1, int(max(in_w, in_h) if bool(use_input_size) else output_height))
        out_w = out_h if output_mode == "vr180_equirect_1_1" else out_h * 2

        ys = np.linspace(0, out_h - 1, out_h, dtype=np.float32)
        xs = np.linspace(0, out_w - 1, out_w, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)

        if output_mode == "vr180_equirect_1_1":
            lon = (xv / max(out_w - 1, 1) - 0.5) * np.pi
            front_valid = np.ones((out_h, out_w), dtype=bool)
        else:
            lon = (xv / max(out_w - 1, 1) - 0.5) * (2.0 * np.pi)
            front_valid = np.abs(lon) <= (np.pi / 2.0)

        lat = (0.5 - yv / max(out_h - 1, 1)) * np.pi
        if bool(horizontal_flip):
            lon = -lon
        if bool(vertical_flip):
            lat = -lat

        cos_lat = np.cos(lat)
        dirs = np.stack(
            [
                cos_lat * np.sin(lon),
                np.sin(lat),
                cos_lat * np.cos(lon),
            ],
            axis=-1,
        ).astype(np.float32)

        rot = _rotation_matrix(yaw, pitch, roll)
        dirs = dirs @ rot.T

        z = dirs[..., 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        max_theta = np.deg2rad(float(fisheye_fov) / 2.0)
        valid = front_valid & (theta <= max_theta + 1e-6) & (z >= -1e-6)

        sin_theta = np.sin(theta)
        safe_sin = np.where(np.abs(sin_theta) < 1e-6, 1.0, sin_theta)
        nx = dirs[..., 0] / safe_sin
        ny = -dirs[..., 1] / safe_sin
        nx = np.where(np.abs(sin_theta) < 1e-6, 0.0, nx)
        ny = np.where(np.abs(sin_theta) < 1e-6, 0.0, ny)

        rho = _lens_radius(theta, max_theta, lens_model)
        if fill_mode == "edge":
            rho = np.clip(rho, 0.0, 1.0)

        radius_px = max(1.0, min(in_w, in_h) * 0.5 * float(lens_radius))
        input_cx = float(center_x) * (in_w - 1)
        input_cy = float(center_y) * (in_h - 1)
        u = input_cx + nx * rho * radius_px
        v = input_cy + ny * rho * radius_px

        in_bounds = (u >= 0) & (u <= (in_w - 1)) & (v >= 0) & (v <= (in_h - 1))
        valid = valid & in_bounds
        u = np.clip(u, 0, in_w - 1)
        v = np.clip(v, 0, in_h - 1)

        frames = images.detach().float().clamp(0, 1).cpu().numpy()
        outputs = []
        for i in range(batch):
            sampled = _bilinear_sample(frames[i], u, v).astype(np.float32)
            if fill_mode == "black":
                sampled[~valid] = 0.0
            outputs.append(sampled)

        out_np = np.stack(outputs, axis=0)
        out_t = torch.from_numpy(out_np).to(images.device, dtype=images.dtype).clamp(0, 1)
        info = (
            f"OK fisheye->VR180 equirect: input={in_w}x{in_h} output={out_w}x{out_h} "
            f"mode={output_mode} fov={float(fisheye_fov):.1f} lens={lens_model} "
            f"center=({float(center_x):.3f},{float(center_y):.3f}) radius={float(lens_radius):.3f} "
            f"yaw={float(yaw):.1f} pitch={float(pitch):.1f} roll={float(roll):.1f} fill={fill_mode}"
        )
        return (out_t, info)


NODE_CLASS_MAPPINGS = {"FisheyeToVR180Equirect": FisheyeToVR180Equirect}
NODE_DISPLAY_NAME_MAPPINGS = {"FisheyeToVR180Equirect": "Fisheye 1:1 -> VR180 Equirect"}
