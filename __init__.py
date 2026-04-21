from .pano_outpaint_canvas import PanoOutpaintCanvas

NODE_CLASS_MAPPINGS = {
    "PanoOutpaintCanvas": PanoOutpaintCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoOutpaintCanvas": "Pano Outpaint Canvas",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
