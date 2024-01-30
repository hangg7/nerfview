#!/usr/bin/env python3
#
# File   : basic.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 12/26/2023
#
# Distributed under terms of the MIT license.

import time

import numpy as np

from nerfview import CameraState, ViewerServer, ViewerStats, with_view_lock


def render_fn(camera_state: CameraState, img_wh: tuple[int, int]):
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

    camera_dirs = np.einsum(
        "ij,hwj->hwi",
        np.linalg.inv(K),
        np.pad(
            np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1) + 0.5,
            ((0, 0), (0, 0), (0, 1)),
            constant_values=1.0,
        ),
    )
    dirs = np.einsum("ij,hwj->hwi", c2w[:3, :3], camera_dirs)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    img = ((dirs + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return img


@with_view_lock
def train_step(step: int, budget: float = 0.1):
    print(f"Training step {step}...")
    time.sleep(budget)
    print("Done.")


if __name__ == "__main__":
    # NOTE(Hang Gao @ 01/26): Debug why this not working.
    #  # Use case 1: Just serving the images -- useful when inspecting a
    #  # pretrained checkpoint.
    #  server = ViewerServer(port=30108, render_fn=render_fn)
    #  while True:
    #      with server.lock:
    #          time.sleep(1.0)
    #      time.sleep(0.5)

    #  # Use case 1: Just serving the images -- useful when inspecting a
    #  # pretrained checkpoint.
    #  server = ViewerServer(port=30108, render_fn=render_fn)
    #  while True:
    #      time.sleep(1.0)

    # Use case 2: Periodically update the renderer -- useful when training.
    viewer_stats = ViewerStats()
    server = ViewerServer(port=30108, render_fn=render_fn)
    stats = server.stats
    max_steps = 10000
    num_train_rays_per_step = 512 * 512
    for step in range(max_steps):
        while server.training_state == "paused":
            time.sleep(0.01)

        train_step(step)
        num_train_steps_per_sec = 10.0
        num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
        tic = time.time()

        # Update viewer stats.
        stats.num_train_rays_per_sec = num_train_rays_per_sec
        # Update viewer.
        server.update(step, num_train_rays_per_step)
    server.complete()
