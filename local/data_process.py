# -----------------------------------------------------------------------------
# Rewritten and modified by Wentao Yu (2026)
# Based on: https://github.com/filaPro/cad-recode/blob/main/demo.ipynb
# -----------------------------------------------------------------------------


import os

import trimesh
import torch
import numpy as np

# open3d and skimage are imported inside render() rather than here. Only the
# rendering helpers use them, training never calls those, and importing open3d
# at module scope pulls in libGLdispatch from /.singularity.d/libs -- the host's
# OpenGL stack, injected by --nv. On a node whose host glibc is newer than the
# image's (one node, glibc 2.38 vs the image's 2.35) that fails to load and takes
# the whole training job with it, eleven seconds in, for a library it never uses.

# pytorch3d ships as a compiled extension linked against one specific torch ABI,
# so it cannot follow torch to a newer build -- and a newer build is exactly what
# an sm_120 card needs. FPS is a dozen lines, so the dependency is optional: use
# the fast kernel when it loads, fall back to torch when it does not.
try:
    from pytorch3d.ops import sample_farthest_points as _p3d_fps
except Exception:                                   # ImportError, or ABI mismatch
    _p3d_fps = None


def _fps_torch(points, K):
    """Farthest point sampling, matching pytorch3d's random_start_point=False path.

    Same greedy loop and the same starting index (0), so the selected points are
    identical to the compiled kernel's. That equality matters: this runs on every
    training sample, and a different point cloud is a different dataset -- runs on
    two card types would no longer be comparable.
    """
    pts = points[0]                                  # (P, 3); callers pass batch 1
    P = pts.shape[0]
    k = min(K, P)
    ids = torch.empty(k, dtype=torch.int64, device=pts.device)
    ids[0] = 0
    closest = torch.full((P,), float("inf"), dtype=pts.dtype, device=pts.device)
    for i in range(1, k):
        last = pts[ids[i - 1]]
        closest = torch.minimum(closest, ((pts - last) ** 2).sum(-1))
        ids[i] = torch.argmax(closest)
    return pts[ids].unsqueeze(0), ids.unsqueeze(0)


def sample_farthest_points(points, K):
    """FPS through pytorch3d when available, otherwise the torch fallback.

    CAD_FPS_TORCH=1 forces the fallback even when pytorch3d loads -- used to prove
    the two agree before trusting the fallback on a run that matters.
    """
    if _p3d_fps is not None and not os.environ.get("CAD_FPS_TORCH"):
        return _p3d_fps(points, K=K)
    return _fps_torch(points, K)

def mesh_to_point_cloud(mesh, n_points=256, n_pre_points=8192):
    """
    Convert a 3D mesh into a fixed-size point cloud using surface sampling
    followed by farthest point sampling (FPS).

    Args:
        mesh (trimesh.Trimesh): Input 3D mesh.
        n_points (int, optional): Number of points in the output point cloud.
            Default is 256.
        n_pre_points (int, optional): Number of points initially sampled
            uniformly from the mesh surface before FPS. Default is 8192.

    Returns:
        np.ndarray: Array of shape (n_points, 3) representing the sampled
            point cloud.
    """
    # A degenerate solid -- zero-area faces, NaN vertices from a failed boolean --
    # makes sample_surface return non-finite coordinates. Nothing downstream checks,
    # so those NaNs reach the point encoder, then the loss, then the weights, and the
    # run is dead within one optimizer step with no error message. This cost two runs
    # (joint at ~48k steps, sweep at ~3k) before it was found. Raising here is enough:
    # the dataset already treats an exception from this function as "skip the sample".
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh has non-finite vertices")

    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    if not np.isfinite(vertices).all():
        raise ValueError("surface sampling produced non-finite points")

    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    out = np.asarray(vertices[ids])
    if not np.isfinite(out).all():
        raise ValueError("farthest-point sampling produced non-finite points")
    return out   



# def render(geometry, image_size=128, camera_distance=-1.8, image_path='/tmp/tmp.png'):
#     """
#     Render a 3D geometry to a 2D image using Open3D with a fixed camera setup.

#     The function sets a custom camera extrinsic matrix based on a predefined
#     look-at point, viewing direction, and up vector, captures the rendered
#     image, and resizes it to the target resolution.

#     Args:
#         geometry (open3d.geometry.Geometry): Input 3D geometry to render.
#         image_size (int, optional): Output image size (image_size x image_size).
#             Default is 128.
#         camera_distance (float, optional): Distance of the camera from the
#             look-at point along the viewing direction. Default is -1.8.
#         image_path (str, optional): Temporary path to save the rendered image.
#             Default is '/tmp/tmp.png'.

#     Returns:
#         np.ndarray: Rendered RGB image of shape (image_size, image_size, 3)
#             with dtype uint8.
#     """
#     visualizer = open3d.visualization.Visualizer()
#     visualizer.create_window(width=image_size * 4, height=image_size * 4)
#     visualizer.add_geometry(geometry)

#     lookat = np.array([0, 0, 0], dtype=np.float32)
#     front = np.array([1, 1, 1], dtype=np.float32)
#     up = np.array([0, 1, 0], dtype=np.float32)
#     eye = lookat + front * camera_distance
#     right = np.cross(up, front)
#     right /= np.linalg.norm(right)
#     true_up = np.cross(front, right)
#     rotation_matrix = np.column_stack((right, true_up, front)).T
#     extrinsic = np.eye(4)
#     extrinsic[:3, :3] = rotation_matrix
#     extrinsic[:3, 3] = -rotation_matrix @ eye

#     view_control = visualizer.get_view_control()
#     camera_params = view_control.convert_to_pinhole_camera_parameters()
#     camera_params.extrinsic = extrinsic
#     view_control.convert_from_pinhole_camera_parameters(camera_params)

#     visualizer.poll_events()
#     visualizer.update_renderer()
#     visualizer.capture_screen_image(image_path)
#     visualizer.destroy_window()

#     image = skimage.io.imread(image_path)
#     image = skimage.transform.resize(
#         image,
#         output_shape=(image_size, image_size),
#         order=2,
#         anti_aliasing=True,
#         preserve_range=True).astype(np.uint8)
#     return image


def render(geometry, image_size=128, camera_distance=None, image_path='/tmp/tmp.png'):
    """改进的渲染函数，支持自动相机距离"""
    # Imported here, not at module scope -- see the note at the top of the file.
    import open3d
    import skimage

    # 如果未指定相机距离，自动计算
    if camera_distance is None:
        bbox = geometry.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_extent = np.max(extent)
        camera_distance = -max_extent * 2.5  # 自适应距离
    
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window(width=image_size * 4, height=image_size * 4, visible=False)
    visualizer.add_geometry(geometry)
    
    lookat = geometry.get_center()  # 改用点云中心作为观察点
    front = np.array([1, 1, 1], dtype=np.float32)
    front = front / np.linalg.norm(front)  # 归一化
    up = np.array([0, 1, 0], dtype=np.float32)
    
    eye = lookat + front * camera_distance
    right = np.cross(up, front)
    right /= np.linalg.norm(right)
    true_up = np.cross(front, right)
    
    rotation_matrix = np.column_stack((right, true_up, front)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye
    
    view_control = visualizer.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(image_path)
    visualizer.destroy_window()
    
    image = skimage.io.imread(image_path)
    image = skimage.transform.resize(
        image, 
        output_shape=(image_size, image_size), 
        order=2, 
        anti_aliasing=True, 
        preserve_range=True
    ).astype(np.uint8)
    
    return image