import time
from typing import Tuple

import numpy as np
import tyro
import viser
from jaxtyping import UInt8
from tqdm import tqdm

import nerfview


def main(port: int = 8080, max_steps: int = 50, rendering_latency: float = 0.0):
    """Rendering a dummy training process.

    This example is the best starting point to understand the API for training
    time update.

    You can inject an artificial rendering latency to simulate real-world
    scenarios. The higher the latency, the lower the resolution of the rendered
    output during camera movement.

    Args:
        port (int): The port number for the viewer server.
        max_steps (int): The maximum number of training steps.
        rendering_latency (float): The artificial rendering latency.
    """

    step: int = 0

    def render_fn(
        camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ) -> UInt8[np.ndarray, "H W 3"]:
        # Get camera parameters.
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)

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

    # Initialize the viser server and our viewer.
    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=delayed_render_fn,
        mode="training",
    )
    # Optionally make world axes visible for better visualization in this
    # example. You don't need to do this in your own code.
    server.scene.world_axes.visible = True
    # Optionally make the training utility lower such that we update the scene
    # more frequently in this example. You dont need to do this in your own
    # code.
    viewer._train_util_slider.value = 0.5

    for step in tqdm(range(max_steps)):
        # Allow user to pause the training process.
        while viewer.state.status == "paused":
            time.sleep(0.01)
        # Do the training step and compute the number of training rays per second.
        tic = time.time()
        with viewer.lock:
            num_train_rays_per_step = training_step()
        num_train_steps_per_sec = 1.0 / (time.time() - tic)
        num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
        # Update the viewer state.
        viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
        # Update the scene.
        viewer.update(step, num_train_rays_per_step)
    viewer.complete()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
