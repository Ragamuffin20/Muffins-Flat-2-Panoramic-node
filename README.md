# Muffins Flat 2 Panoramic Node

A lightweight ComfyUI custom node for preparing flat image or video frames on a larger panoramic canvas.

This node is intended to be used in an outpainting workflow. It centers the incoming image batch inside a black canvas, scales it to the requested panoramic or fisheye target, and outputs a mask for the newly padded area so an inpaint/outpaint pipeline can generate the missing surroundings.

## Node

### Pano Outpaint Canvas

ComfyUI class name: `PanoOutpaintCanvas`

Inputs:

- `images`: input `IMAGE` batch
- `output_projection`: `panorama_2_1` or `fisheye_1_1`
- `canvas_width`: base canvas width
- `canvas_height`: base canvas height
- `source_scale`: how much of the final canvas the source image should occupy
- `outpaint_scale`: multiplier for expanding the final outpaint canvas

Outputs:

- `canvas_images`: the source images centered on the larger canvas
- `padding_mask`: mask where padded/outpaint regions are `1.0` and source pixels are `0.0`
- `info`: a short string describing the generated canvas, source placement, and scale

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

Restart ComfyUI after installing.

## Basic Use

1. Add `Pano Outpaint Canvas` to your workflow.
2. Connect an `IMAGE` or video-frame image batch to `images`.
3. Choose `panorama_2_1` for a 2:1 panoramic canvas or `fisheye_1_1` for a square fisheye-style canvas.
4. Send `canvas_images` into your outpaint/inpaint path.
5. Send `padding_mask` to the mask input of the outpainting stage.

The original workflow used to build this node is not included in this repository.

## Support

Support my work on Patreon: [The World of Anatnom](https://www.patreon.com/c/theworldofanatnom?vanity=user)
