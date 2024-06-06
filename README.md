# nerfview

nerfview is a minimalistic viewer for interactive NeRF rendering. It is largely inspired by [nerfstudio's viewer](https://github.com/nerfstudio-project/nerfstudio), but with a standalone packaging and simple API to quickly integrate into your own projects.

## Installation

For existing project, you can install it via pip:

```bash
pip install git+https://github.com/hangg7/nerfview.git
```

For examples, you can clone this repository and install it via pip:

```bash
git clone https://github.com/hangg7/nerfview
pip install -e ".[examples]"
```

## Usage

nerfview is built on [viser](https://viser.studio/latest/) and provides a simple API for interactive viewing. It supports two modes: rendering an existing NeRF model and rendering a progressive training process.
The canonical usage is as follows:

```python
from nerfview import CameraState, ViewerServer

def render_fn(
    camera_state: CameraState, img_wh: Tuple[int, int]
) -> UInt8[np.ndarray, "H W 3"]:
    img = your_rendering_logic(...)
    return img

server = ViewerServer(render_fn=render_fn)
```

It will start a viser server and render the image based on the camera state which you can interact with.

## Examples

We provide a few examples, for both rendering and training:

- Rendering a dummy scene.
- Rendering a dummy training process.
- Rendering a gsplat scene.
- Rendering a gsplat training process.

## Acknowledgement

This project cannot exist without the great work of [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [viser](https://viser.studio/latest/). We thank the authors for their great work and open-source spirit.
