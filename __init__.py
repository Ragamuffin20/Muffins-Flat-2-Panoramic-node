import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _register(name, import_func):
    try:
        loaded = import_func()
        if not isinstance(loaded, tuple):
            loaded = (loaded,)
        for class_name, cls, node_name in loaded:
            NODE_CLASS_MAPPINGS[class_name] = cls
            NODE_DISPLAY_NAME_MAPPINGS[class_name] = node_name
            print(f"[{class_name}] Loaded successfully.")
    except Exception:
        print(f"[{name}] FAILED TO IMPORT:")
        traceback.print_exc()


def _perspective_nodes():
    from .perspective2panorama import Perspective2Panorama
    from .perspective2panorama_180_nomaskfill import Perspective2Panorama180_NoBlack
    from .perspective2panorama_smallmask import Perspective2PanoramaSmallMask

    return (
        ("Perspective2Panorama", Perspective2Panorama, "Perspective -> Panorama (Universal)"),
        ("Perspective2PanoramaSmallMask", Perspective2PanoramaSmallMask, "Perspective -> Panorama (Small Mask, No Resize)"),
        ("Perspective2Panorama180_NoBlack", Perspective2Panorama180_NoBlack, "Perspective -> 180 Pano (No Black Fill)"),
    )


def _fisheye_nodes():
    from .fisheye_projection_only import FisheyeLensWarpOnly, FisheyeProjectionOnly
    from .fisheye_to_vr180_equirect import FisheyeToVR180Equirect
    from .vr_nodes import ConvertToVR, EstimateVideoOrientation

    return (
        ("FisheyeProjectionOnly", FisheyeProjectionOnly, "Fisheye Projection Only"),
        ("FisheyeLensWarpOnly", FisheyeLensWarpOnly, "Fisheye Lens Warp Only"),
        ("FisheyeToVR180Equirect", FisheyeToVR180Equirect, "apply panoramic"),
        ("ConvertToVR", ConvertToVR, "Convert To VR / Apply Panoramic"),
        ("EstimateVideoOrientation", EstimateVideoOrientation, "Estimate Video Orientation"),
    )


def _outpaint_nodes():
    from .masked_outpaint_guide_fill import MaskedOutpaintGuideFill
    from .pano_outpaint_canvas import PanoOutpaintCanvas

    return (
        ("MaskedOutpaintGuideFill", MaskedOutpaintGuideFill, "Masked Outpaint Guide Fill"),
        ("PanoOutpaintCanvas", PanoOutpaintCanvas, "Pano Outpaint Canvas"),
    )


_register("Perspective nodes", _perspective_nodes)
_register("Fisheye nodes", _fisheye_nodes)
_register("Outpaint nodes", _outpaint_nodes)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
