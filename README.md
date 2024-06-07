# nerfview

nerfview is a minimal web viewer for interactive NeRF rendering. It is largely
inspired by [nerfstudio's
viewer](https://github.com/nerfstudio-project/nerfstudio), but with a
standalone packaging and simple API to quickly integrate into your own
projects.

## Installation

For existing project, you can install it via pip:

```bash
pip install git+https://github.com/hangg7/nerfview.git
```

To run our examples, you can clone this repository and then install it locally:

```bash
git clone https://github.com/hangg7/nerfview
# Install torch first.
pip install torch
# Then this repo and dependencies for running examples. Note that `gsplat`
# requires compilation and this will take some time for the first time.
pip install -e ".[examples]"
```

## Usage

nerfview is built on [viser](https://viser.studio/latest/) and provides a
simple API for interactive viewing. It supports two modes: rendering an
existing NeRF model and rendering a progressive training process. The canonical
usage is as follows:

```python
import numpy as np
from nerfview import CameraState, ViewerServer

def render_fn(
    camera_state: CameraState, img_wh: Tuple[int, int]
) -> UInt8[np.ndarray, "H W 3"]:
    # Parse camera state for camera-to-world matrix (c2w) and intrinsic matrix
    # (K) as float64 numpy arrays.
    fov = camera_state.fov
    c2w = camera_state.c2w
    W, H = img_wh
    focal_length = H / 2.0 / np.tan(fov / 2.0)
    K = np.array(
        [
            [focal_length, 0.0, W / 2.0],
            [0.0, focal_length, H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    img = your_rendering_logic(...)
    return img

server = ViewerServer(render_fn=render_fn)
```

It will start a viser server and render the image based on the camera state
which you can interact with.

## Examples

We provide a few examples ranging from toy rendering to real-world NeRF training
applications. Click on the dropdown to see more details. You can always ask for
help message by the `-h` flag.

<details>
<summary>Rendering a dummy scene.</summary>
<br>
    
https://github.com/hangg7/nerfview/assets/10098306/53a41fac-bce7-4820-be75-f90483bc22a0

This example is the best starting point to understand the basic API.

```bash
python examples/00_dummy_rendering.py
```

</details>

<details>
<summary>Rendering a dummy training process.</summary>
<br>
    
https://github.com/hangg7/nerfview/assets/10098306/8b13ca4a-6aaa-46a7-a333-b889c2a4ac15

This example is the best starting point to understand the API for training time
update.

```bash
python examples/01_dummy_training.py
```

</details>

<details>
<summary>Rendering a mesh scene.</summary>
<br>
    
https://github.com/hangg7/nerfview/assets/10098306/84c9993f-82a3-48fb-9786-b5205bffcd6f

This example showcases how to interactively render a mesh by directly serving
rendering results from <a href="https://nvlabs.github.io/nvdiffrast/">nvdiffrast</a>.

```bash
# Only need to run once the first time.
bash examples/assets/download_dragon_mesh.sh
CUDA_VISIBLE_DEVICES=0 python examples/02_mesh_rendering.py
```

</details>

<details>
<summary>Rendering a pretrained 3DGS scene.</summary>
<br>
    
https://github.com/hangg7/nerfview/assets/10098306/7b526105-8b6f-431c-9b49-10c821a3bd36

This example showcases how to render a pretrained 3DGS model using gsplat. The
scene is cropped such that it is smaller to download. It is essentially the [simple_viewer example](https://github.com/nerfstudio-project/gsplat/blob/v1.0/examples/simple_viewer.py), which we include here to be self-contained.

```bash
# Only need to run once the first time.
bash examples/assets/download_gsplat_ckpt.sh
CUDA_VISIBLE_DEVICES=0 python examples/03_gsplat_rendering.py \
    --ckpt results/garden/ckpts/ckpt_6999_crop.pt
```

</details>

<details>
<summary>Rendering a 3DGS training process.</summary>
<br>
    
https://github.com/hangg7/nerfview/assets/10098306/640d4067-e410-49aa-86b8-325140dd73a8

This example showcases how to render while training 3DGS on mip-NeRF's garden
scene using gsplat. It is essentially the [simple_trainer example](https://github.com/nerfstudio-project/gsplat/blob/v1.0/examples/simple_trainer.py), which we include here to be self-contained.

```bash
# Only need to run once the first time.
bash examples/assets/download_colmap_garden.sh
CUDA_VISIBLE_DEVICES=0 python examples/04_gsplat_training.py \
    --data_dir examples/assets/colmap_garden/ \
    --data_factor 8 \
    --result_dir results/garden/
```

</details>

## Acknowledgement

This project cannot exist without the great work of
[nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and
[viser](https://viser.studio/latest/). We rely on
[nvdiffrast](https://nvlabs.github.io/nvdiffrast/) for the mesh example and
[gsplat](https://docs.gsplat.studio/latest/) for the 3DGS examples. We thank
the authors for their great work and open-source spirit.
