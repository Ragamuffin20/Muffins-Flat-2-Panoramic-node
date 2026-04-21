# Muffins Flat 2 Panoramic Node

ComfyUI custom nodes for preparing flat image or video frames for panoramic, fisheye, and VR180 workflows.

The main use case is an outpainting pipeline: place a flat frame on a larger panoramic or square fisheye canvas, generate the missing surroundings, optionally apply a fisheye crop/warp, then convert the final 1:1 fisheye result into a VR180 equirectangular video that can be viewed in a 180/360 viewer.

Workflow files are intentionally not included in this repository.

## Nodes

### Pano Outpaint Canvas

ComfyUI class name: `PanoOutpaintCanvas`

Prepares an image batch on a black panoramic canvas and outputs a mask for the new padded/outpaint area.

Inputs:

- `images`: input `IMAGE` batch
- `output_projection`: `panorama_2_1` or `fisheye_1_1`
- `canvas_width`: base canvas width
- `canvas_height`: base canvas height
- `source_scale`: scale for the original source inside the canvas
- `outpaint_scale`: expands the final canvas while preserving the selected projection ratio

Outputs:

- `canvas_images`: source images centered on the larger canvas
- `padding_mask`: mask where padded/outpaint regions are `1.0` and source pixels are `0.0`
- `info`: placement and scale summary

### Fisheye Lens Warp Only

ComfyUI class name: `FisheyeLensWarpOnly`

Applies a fisheye-style crop/warp to an image batch without adding new padding. This is useful as a second-pass refinement step after the first outpaint pass has already produced the square frame.

Inputs include:

- `images`
- `strength`
- `rectilinear_fov`
- `zoom`
- `center_x`
- `center_y`
- `lens_radius`
- `edge_fade`
- `vignette`

### Fisheye Projection Only

ComfyUI class name: `FisheyeProjectionOnly`

Projects frames into a square fisheye-style view for second-pass processing. It can clamp outside-lens pixels to the edge instead of creating a new black padding region.

### Fisheye 1:1 -> VR180 Equirect

ComfyUI class name: `FisheyeToVR180Equirect`

Converts a square 1:1 fisheye image batch into a VR180 equirectangular output.

Recommended output mode for a half-sphere 180 viewer:

- `vr180_equirect_1_1`

Alternate output mode for players that expect a full equirectangular container:

- `padded_360_equirect_2_1`

Useful controls:

- `fisheye_fov`: usually `180`
- `lens_model`: `equidistant`, `equisolid`, `orthographic`, or `stereographic`
- `center_x` / `center_y`: correct off-center fisheye frames
- `lens_radius`: match the circular fisheye radius inside the square frame
- `yaw`, `pitch`, `roll`: orientation correction
- `horizontal_flip` / `vertical_flip`: mirror correction

### Masked Outpaint Guide Fill

ComfyUI class name: `MaskedOutpaintGuideFill`

Fills masked guide areas for outpaint conditioning while preserving the original unmasked pixels.

### Perspective Nodes

The package also includes the earlier perspective-to-panorama helpers:

- `Perspective2Panorama`
- `Perspective2PanoramaSmallMask`
- `Perspective2Panorama180_NoBlack`

## Installation

Clone this repository into your ComfyUI `custom_nodes` folder:

```powershell
cd C:\AI\comfy\ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/Ragamuffin20/Muffins-Flat-2-Panoramic-node.git
```

Install requirements if needed:

```powershell
cd Muffins-Flat-2-Panoramic-node
python -m pip install -r requirements.txt
```

Restart ComfyUI after installing or updating.

## Basic Use

1. Add `Pano Outpaint Canvas` to your workflow.
2. Connect an `IMAGE` or video-frame image batch to `images`.
3. Choose `panorama_2_1` for a 2:1 panoramic canvas or `fisheye_1_1` for a square fisheye-style canvas.
4. Send `canvas_images` into your outpaint/inpaint path.
5. Send `padding_mask` to the mask input of the outpainting stage.
6. For a second pass, use `Fisheye Lens Warp Only` or `Fisheye Projection Only` without adding new padding.
7. To create viewer-ready VR180 output, send the final square fisheye frames into `Fisheye 1:1 -> VR180 Equirect`.

## Support

Support my work on Patreon: [The World of Anatnom](https://www.patreon.com/c/theworldofanatnom?vanity=user)
