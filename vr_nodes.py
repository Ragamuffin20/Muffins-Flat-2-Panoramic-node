import math

import numpy as np
import torch

from .fisheye_to_vr180_equirect import _bilinear_sample, _rotation_matrix


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _wrap_degrees(value, limit):
    value = ((float(value) + 180.0) % 360.0) - 180.0
    return float(np.clip(value, -limit, limit))


def _sample_frame_indices(batch, sample_frames):
    count = max(1, min(int(sample_frames), int(batch)))
    if count == 1:
        return [batch // 2]
    return np.linspace(0, batch - 1, count).round().astype(np.int32).tolist()


def _smooth_1d(values, radius):
    values = np.asarray(values, dtype=np.float32)
    if radius <= 0:
        return values
    kernel_size = radius * 2 + 1
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(values, kernel, mode="same")


def _analysis_frame(frame, vr_format):
    h, w, _ = frame.shape
    if vr_format == "vr180_equirect_1_1" and w >= h * 1.75:
        return frame[:, : w // 2, :]
    return frame


def _estimate_single_orientation(frame, horizontal_degrees, vr_format):
    frame = _analysis_frame(frame, vr_format)
    gray = (
        frame[..., 0] * 0.299
        + frame[..., 1] * 0.587
        + frame[..., 2] * 0.114
    ).astype(np.float32)
    h, w = gray.shape

    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx * gx + gy * gy)

    central_y0 = max(0, int(h * 0.18))
    central_y1 = min(h, int(h * 0.82))
    central = slice(central_y0, central_y1)

    horizontal_profile = _smooth_1d(np.mean(edge[central, :], axis=0), max(1, w // 96))
    vertical_profile = _smooth_1d(np.mean(np.abs(gy), axis=1), max(1, h // 96))

    uniform_x = float(np.std(horizontal_profile) / (np.mean(horizontal_profile) + 1e-6))
    uniform_y = float(np.std(vertical_profile) / (np.mean(vertical_profile) + 1e-6))

    x_weights = horizontal_profile + 1e-6
    y_weights = vertical_profile + 1e-6
    x_centroid = float(np.sum(np.arange(w, dtype=np.float32) * x_weights) / np.sum(x_weights))
    y_centroid = float(np.sum(np.arange(h, dtype=np.float32) * y_weights) / np.sum(y_weights))

    yaw = (0.5 - (x_centroid + 0.5) / max(w, 1)) * horizontal_degrees * min(uniform_x, 1.0)
    pitch = (0.5 - (y_centroid + 0.5) / max(h, 1)) * 90.0 * min(uniform_y, 1.0)

    strong = edge > np.percentile(edge, 82.0)
    if np.count_nonzero(strong) < 32:
        return yaw, pitch, 0.0, 0.0

    gx_s = gx[strong]
    gy_s = gy[strong]
    line_angles = np.arctan2(gy_s, gx_s) + np.pi / 2.0
    line_angles = (line_angles + np.pi / 2.0) % np.pi - np.pi / 2.0
    horizontal_preference = np.exp(-((np.rad2deg(line_angles) / 35.0) ** 2))
    weights = edge[strong] * horizontal_preference

    if float(np.sum(weights)) <= 1e-6:
        roll = 0.0
    else:
        mean_angle = 0.5 * math.atan2(
            float(np.sum(weights * np.sin(2.0 * line_angles))),
            float(np.sum(weights * np.cos(2.0 * line_angles))),
        )
        roll = -math.degrees(mean_angle)

    confidence = float(np.clip(np.mean(edge[strong]) / (np.mean(edge) + 1e-6) / 10.0, 0.0, 1.0))
    if confidence < 0.08:
        yaw = 0.0
        pitch = 0.0
        roll = 0.0

    return yaw, pitch, roll, confidence


def _panoramic_prewarp(
    frames,
    output_mode,
    out_h,
    out_w,
    yaw,
    pitch,
    roll,
    horizontal_flip,
    vertical_flip,
):
    batch, in_h, in_w, _ = frames.shape
    horizontal_radians = 2.0 * np.pi if output_mode == "padded_360_equirect_2_1" else np.pi

    ys = np.arange(out_h, dtype=np.float32)
    xs = np.arange(out_w, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    lon = ((xv + 0.5) / max(out_w, 1) - 0.5) * horizontal_radians
    lat = (0.5 - (yv + 0.5) / max(out_h, 1)) * np.pi

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

    dirs = dirs @ _rotation_matrix(yaw, pitch, roll).T

    src_lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    src_lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))

    if output_mode == "padded_360_equirect_2_1":
        x_norm = np.mod(src_lon / (2.0 * np.pi) + 0.5, 1.0)
    else:
        x_norm = np.clip(src_lon / np.pi + 0.5, 0.0, 1.0)

    y_norm = np.clip(0.5 - src_lat / np.pi, 0.0, 1.0)

    # The pole scale is the panoramic pre-distortion: it narrows horizontal
    # detail near the top and bottom so a VR/equirect viewer unwraps it flat.
    pole_scale = np.clip(np.cos(src_lat), 0.0, 1.0)
    x_norm = 0.5 + (x_norm - 0.5) * pole_scale

    u = np.clip(x_norm * (in_w - 1), 0, in_w - 1)
    v = np.clip(y_norm * (in_h - 1), 0, in_h - 1)

    outputs = []
    for i in range(batch):
        sampled = _bilinear_sample(frames[i], u, v).astype(np.float32)
        outputs.append(sampled)
    return np.stack(outputs, axis=0)


class EstimateVideoOrientation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vr_format": (
                    ["vr180_equirect_1_1", "padded_360_equirect_2_1"],
                    {"default": "vr180_equirect_1_1"},
                ),
                "sample_frames": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("yaw", "pitch", "roll", "info")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(self, images, vr_format, sample_frames):
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        frames = images.detach().float().clamp(0, 1).cpu().numpy()
        batch, h, w, _ = frames.shape
        horizontal_degrees = 360.0 if vr_format == "padded_360_equirect_2_1" else 180.0

        estimates = []
        for index in _sample_frame_indices(batch, sample_frames):
            estimates.append(_estimate_single_orientation(frames[index], horizontal_degrees, vr_format))

        yaw = _wrap_degrees(np.median([item[0] for item in estimates]), 180.0)
        pitch = _wrap_degrees(np.median([item[1] for item in estimates]), 89.0)
        roll = _wrap_degrees(np.median([item[2] for item in estimates]), 45.0)
        confidence = float(np.mean([item[3] for item in estimates]))

        info = (
            f"OK estimated orientation: frames={len(estimates)}/{batch} input={w}x{h} "
            f"format={vr_format} yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f} "
            f"confidence={confidence:.2f}"
        )
        return (yaw, pitch, roll, info)


class ConvertToVR:
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
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.1}),
                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "horizontal_flip": ("BOOLEAN", {"default": False}),
                "vertical_flip": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("vr_frames", "info", "viewer_preview_first_frame", "legacy_preview_first_frame")
    FUNCTION = "run"
    CATEGORY = "video/utils"

    def run(
        self,
        images,
        output_mode,
        use_input_size,
        output_height,
        yaw,
        pitch,
        roll,
        horizontal_flip,
        vertical_flip,
    ):
        yaw = _as_float(yaw)
        pitch = _as_float(pitch)
        roll = _as_float(roll)

        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected IMAGE torch.Tensor, got {type(images)}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE shape [B,H,W,3], got {tuple(images.shape)}")

        _, in_h, in_w, _ = images.shape
        if output_mode == "padded_360_equirect_2_1":
            out_h = max(1, int(in_h if bool(use_input_size) else output_height))
            out_w = out_h * 2
        else:
            out_h = max(1, int(max(in_w, in_h) if bool(use_input_size) else output_height))
            out_w = out_h

        frames = images.detach().float().clamp(0, 1).cpu().numpy()
        out_np = _panoramic_prewarp(
            frames,
            output_mode,
            out_h,
            out_w,
            yaw,
            pitch,
            roll,
            horizontal_flip,
            vertical_flip,
        )

        vr_frames = torch.from_numpy(out_np).to(images.device, dtype=images.dtype).clamp(0, 1)
        preview = vr_frames[:1].clone()
        info = (
            f"OK panoramic prewarp only: input={in_w}x{in_h} output={out_w}x{out_h} "
            f"mode={output_mode} yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f}; "
            "no fisheye conversion, lens, crop, or masked fill"
        )
        return (vr_frames, info, preview, preview)


NODE_CLASS_MAPPINGS = {
    "EstimateVideoOrientation": EstimateVideoOrientation,
    "ConvertToVR": ConvertToVR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EstimateVideoOrientation": "Estimate Video Orientation",
    "ConvertToVR": "Convert To VR / Apply Panoramic",
}
