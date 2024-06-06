import time
from typing import Tuple

import numpy as np
import tyro
from jaxtyping import UInt8

from nerfview import CameraState, ViewerServer


def main(port: int, rendering_latency: float = 0.0):
    """Rendering a dummy scene.

    This example allows injecting an artificial rendering latency to simulate
    real-world scenarios. The higher the latency, the lower the resolution of
    the rendered output during camera movement.

    Args:
        port (int): The port number for the viewer server.
        rendering_latency (float): The artificial rendering latency.
    """

    def render_fn(
        camera_state: CameraState, img_wh: Tuple[int, int]
    ) -> UInt8[np.ndarray, "H W 3"]:
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

        # Render a dummy image as a function of camera direction.
        camera_dirs = np.einsum(
            "ij,hwj->hwi",
            np.linalg.inv(K),
            np.pad(
                np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1)
                + 0.5,
                ((0, 0), (0, 0), (0, 1)),
                constant_values=1.0,
            ),
        )
        dirs = np.einsum("ij,hwj->hwi", c2w[:3, :3], camera_dirs)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        img = ((dirs + 1.0) / 2.0 * 255.0).astype(np.uint8)
        return img

    def delayed_render_fn(*args, **kwargs):
        # Inject an artificial rendering latency to simulate the real-world
        # scenario, e.g., rendering from a NGP model.
        time.sleep(rendering_latency)
        return render_fn(*args, **kwargs)

    # Initialize the viser server with our rendering function.
    server = ViewerServer(port=port, render_fn=delayed_render_fn, mode="rendering")
    # Optionally make world axes visible for better visualization in this
    # example. You don't need to do this in your own code.
    server.scene.world_axes.visible = True

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
