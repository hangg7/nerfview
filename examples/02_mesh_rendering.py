import os.path as osp
import time
from typing import Tuple, cast

import numpy as np
import nvdiffrast.torch as dr
import roma
import torch
import torch.nn.functional as F
import trimesh
import tyro
import viser
from jaxtyping import UInt8

import nerfview

CUDA_CTX = dr.RasterizeCudaContext()
GL_CTX = dr.RasterizeGLContext()


def get_proj_mat(
    K: torch.Tensor,
    img_wh: Tuple[int, int],
    znear: float = 0.001,
    zfar: float = 1000.0,
) -> torch.Tensor:
    """Get the OpenGL-style projection matrix.

    Args:
        K (torch.Tensor): (3, 3) camera intrinsic matrix.
        img_wh (Tuple[int, int]): Image width and height.
        znear (float): Near plane.
        zfar (float): Far plane.

    Returns:
        proj_mat (torch.Tensor): (4, 4) projection matrix.
    """
    W, H = img_wh
    # Assume a camera model without distortion.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fovx = 2.0 * torch.arctan(W / (2.0 * fx)).item()
    fovy = 2.0 * torch.arctan(H / (2.0 * fy)).item()
    t = znear * np.tan(0.5 * fovy).item()
    b = -t
    r = znear * np.tan(0.5 * fovx).item()
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (cx - W / 2) / W * 2, 0.0],
            [0.0, 2 * n / (t - b), (cy - H / 2) / H * 2, 0.0],
            [0.0, 0.0, (f + n) / (f - n), -f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=K.device,
    )


@torch.inference_mode()
def render_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    w2c: torch.Tensor,
    proj_mat: torch.Tensor,
    img_wh: Tuple[int, int],
    is_opencv_camera: bool = True,
) -> dict:
    """Render the mesh with normal and mask.

    Args:
        verts (torch.Tensor): (V, 3) mesh vertices.
        faces (torch.Tensor): (F, 3) mesh faces.
        w2c (torch.Tensor): (4, 4) world-to-camera matrix.
        proj_mat (torch.Tensor): (4, 4) projection matrix.
        img_wh (Tuple[int, int]): Image width and height.
        is_opencv_camera (bool): Whether the camera is in OpenCV coordinate.

    Returns:
        dict: A dictionray containing rendered normal and mask.
    """
    W, H = img_wh
    if max(W, H) > 2048:
        ctx = GL_CTX
    else:
        ctx = CUDA_CTX

    # Maintain two sets of typed faces for different ops.
    faces_int32 = faces.to(torch.int32)
    faces_int64 = faces.to(torch.int64)

    mvp = proj_mat @ w2c
    verts_clip = torch.einsum(
        "ij,nj->ni", mvp, F.pad(verts, pad=(0, 1), value=1.0)
    ).contiguous()
    rast, _ = cast(tuple, dr.rasterize(ctx, verts_clip[None], faces_int32, (H, W)))

    # Render mask.
    mask = (rast[..., -1:] > 0).to(torch.float32)
    mask = cast(torch.Tensor, dr.antialias(mask, rast, verts_clip, faces_int32))[
        0
    ].clamp(0, 1)

    # Render normal in camera space.
    i0, i1, i2 = faces_int64[:, 0], faces_int64[:, 1], faces_int64[:, 2]
    v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    face_normals = torch.einsum("ij,nj->ni", w2c[:3, :3], face_normals)
    vert_normals = torch.zeros_like(verts)
    vert_normals.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vert_normals.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vert_normals.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
    vert_normals = torch.where(
        torch.sum(vert_normals * vert_normals, -1, keepdim=True) > 1e-20,
        vert_normals,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vert_normals.device),
    )
    normal = F.normalize(
        cast(
            torch.Tensor,
            dr.interpolate(vert_normals[None].contiguous(), rast, faces_int32),
        )[0],
        dim=-1,
    )
    normal = cast(torch.Tensor, dr.antialias(normal, rast, verts_clip, faces_int32))[
        0
    ].clamp(-1, 1)
    # Align normal coordinate to get to the blue-pinkish side of normals.
    if is_opencv_camera:
        normal[..., [1, 2]] *= -1.0
    normal = mask * (normal + 1.0) / 2.0 + (1.0 - mask)

    return {
        # (H, W, 3).
        "normal": normal,
        # (H, W, 1).
        "mask": mask,
    }


def main(port: int = 8080):
    """Rendering a mesh scene.

    This example showcase how to interactively render a mesh by directly
    serving rendering results from nvdiffrast.

    Note that the server is only serving the images rather than passing the
    mesh object to ThreeJS for rendering.

    Args:
        port (int): The port number for the viewer server.
    """
    device = "cuda"

    mesh_path = osp.join(osp.dirname(__file__), "assets/dragon.obj")
    if not osp.exists(mesh_path):
        raise FileNotFoundError(
            f"Mesh not found, please run `bash examples/assets/download_dragon_mesh.sh`"
        )
    mesh = cast(trimesh.Trimesh, trimesh.load(mesh_path))
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.int64, device=device)

    # Normalize and orient the mesh such that it is centered at the origin and
    # has +z as the up direction.
    verts -= verts.mean(0)
    verts *= 0.05
    verts = torch.einsum(
        "ij,nj->ni",
        roma.rotvec_to_rotmat(torch.tensor([np.pi / 2, 0.0, 0.0], device=device)),
        verts,
    )

    def render_fn(
        camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ) -> UInt8[np.ndarray, "H W 3"]:
        # nvdiffrast requires the image size to be multiples of 8.
        img_wh = (img_wh[0] // 8 * 8, img_wh[1] // 8 * 8)

        # Get camera parameters.
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)

        # Compute the normal map.
        w2c = torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32, device=device))
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        normal = (
            render_mesh(
                verts,
                faces,
                w2c,
                get_proj_mat(K, img_wh),
                img_wh,
            )["normal"]
            .cpu()
            .numpy()
        )

        return normal

    # Initialize the viser server and our viewer.
    server = viser.ViserServer(port=port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=render_fn,
        mode="rendering",
    )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
