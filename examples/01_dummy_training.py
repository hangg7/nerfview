import time
from typing import Tuple

import numpy as np
import tyro
from jaxtyping import UInt8
from tqdm import tqdm

from nerfview import CameraState, ViewerServer


def main(port: int, max_steps: int = 200, rendering_latency: float = 0.0):
    """Rendering the training process of a dummy scene.

    This example allows injecting an artificial rendering latency to simulate
    real-world scenarios. The higher the latency, the lower the resolution of
    the rendered output during camera movement.

    Args:
        port (int): The port number for the viewer server.
        rendering_latency (float): The artificial rendering latency.
    """

    step: int = 0

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

        # Change the rendering based on the training step.
        progress = (step + 1) / max_steps
        img = ((dirs + 1.0) / 2.0 * progress * 255.0).astype(np.uint8)
        return img

    def delayed_render_fn(*args, **kwargs):
        # Inject an artificial rendering latency to simulate the real-world
        # scenario, e.g., rendering from a NGP model.
        time.sleep(rendering_latency)
        return render_fn(*args, **kwargs)

    def training_step():
        # Do some training logic here.
        time.sleep(0.1)
        # Get the number of training rays for each step. This will be used for
        # determine how frequent should the rendering scene be updated even when
        # there is no camera movement.
        num_train_rays_per_step = 512 * 512
        return num_train_rays_per_step

    # Initialize the viser server with the rendering function.
    server = ViewerServer(port=port, render_fn=delayed_render_fn, mode="training")
    # Optionally make world axes visible for better visualization in this
    # example. You don't need to do this in your own code.
    server.scene.world_axes.visible = True

    for step in tqdm(range(max_steps)):
        # Allow user to pause the training process.
        while server.state.status == "paused":
            time.sleep(0.01)
        # Do the training step and compute the number of training rays per second.
        tic = time.time()
        with server.lock:
            num_train_rays_per_step = training_step()
        num_train_steps_per_sec = 1.0 / (time.time() - tic)
        num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
        # Update the viewer stats.
        server.state.num_train_rays_per_sec = num_train_rays_per_sec
        # Update the scene.
        server.update(step, num_train_rays_per_step)
    server.complete()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
