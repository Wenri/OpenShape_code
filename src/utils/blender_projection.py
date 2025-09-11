import numpy as np
import torch
from typing import Tuple, Optional, Union
from utils.projection import points_to_pixels, create_camera_pose

def blender_camera_matrix(
    lens_mm: float = 35.0,
    sensor_width_mm: float = 32.0,
    image_width: int = 512,
    image_height: int = 512
) -> np.ndarray:
    """
    Create camera intrinsic matrix matching Blender's camera setup from render_single_glb.py.

    Args:
        lens_mm: Focal length in millimeters (default 35mm from setup_camera())
        sensor_width_mm: Sensor width in millimeters (default 32mm from setup_camera())
        image_width: Image width in pixels (default 512 from render settings)
        image_height: Image height in pixels (default 512 from render settings)

    Returns:
        camera_matrix: 3x3 camera intrinsic matrix matching Blender setup
    """
    # Calculate focal length in pixels based on sensor size and image resolution
    # This matches Blender's internal camera projection calculations
    focal_length_px = (lens_mm / sensor_width_mm) * image_width

    # Principal point at image center
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Blender uses square pixels, so fx = fy
    fx = fy = focal_length_px

    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ], dtype=np.float32)

    return camera_matrix


def sample_blender_camera_pose(phi: float, theta: float, radius: float = 1.2) -> np.ndarray:
    """
    Create camera pose matching the sampling pattern from render_single_glb.py.

    Args:
        phi: Elevation angle in radians
        theta: Azimuth angle in radians
        radius: Camera distance from origin (default 1.2 from --camera_dist)

    Returns:
        camera_pose: 4x4 camera-to-world transformation matrix
    """
    # Sample camera location using spherical coordinates (matches sample_camera_loc)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    camera_position = np.array([x, y, z])

    # Camera looks at origin (matches the constraint setup)
    target = np.array([0.0, 0.0, 0.0])

    # Use Z-up coordinate system (matches Blender default)
    up = np.array([0.0, 0.0, 1.0])

    return create_camera_pose(camera_position, target, up)


def get_blender_default_viewpoints() -> list:
    """
    Get the 12 default viewpoints used in render_single_glb.py.

    Returns:
        list of (phi, theta) tuples representing the camera viewpoints
    """
    views = [
        [np.pi / 3, np.pi / 6],      # View 0
        [np.pi / 3, np.pi / 6 * 4],  # View 1
        [np.pi / 3, np.pi / 6 * 7],  # View 2
        [np.pi / 3, np.pi / 6 * 10], # View 3
        [np.pi / 2, np.pi / 6 * 2],  # View 4
        [np.pi / 2, np.pi / 6 * 5],  # View 5
        [np.pi / 2, np.pi / 6 * 8],  # View 6
        [np.pi / 2, np.pi / 6 * 11], # View 7
        [np.pi / 3 * 2, 0],          # View 8
        [np.pi / 3 * 2, np.pi / 2],  # View 9
        [np.pi / 3 * 2, np.pi],      # View 10
        [np.pi / 3 * 2, np.pi / 2 * 3] # View 11
    ]
    return views


def project_points_blender_style(
    points_3d: Union[np.ndarray, torch.Tensor],
    colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    view_index: int = 0,
    camera_distance: float = 1.2,
    image_size: Tuple[int, int] = (512, 512)
) -> dict:
    """
    Project 3D points using the exact same camera setup as render_single_glb.py.

    Args:
        points_3d: 3D points in normalized object coordinates, shape (N, 3)
        colors: Optional RGB colors for each point, shape (N, 3)
        view_index: Which of the 12 standard views to use (0-11)
        camera_distance: Camera distance from object center (default 1.2)
        image_size: Image resolution (height, width), default (512, 512)

    Returns:
        dict with projected point information matching Blender rendering
    """
    is_torch = isinstance(points_3d, torch.Tensor)

    # Get Blender camera parameters
    K = blender_camera_matrix(image_width=image_size[1], image_height=image_size[0])

    # Get specific viewpoint
    views = get_blender_default_viewpoints()
    if view_index >= len(views):
        raise ValueError(f"view_index {view_index} must be < {len(views)}")

    phi, theta = views[view_index]
    camera_pose = sample_blender_camera_pose(phi, theta, camera_distance)

    if is_torch:
        K = torch.from_numpy(K).to(points_3d.device)
        camera_pose = torch.from_numpy(camera_pose).to(points_3d.device)

    # Project points
    pixels, valid_indices, depths = points_to_pixels(
        points_3d, K, camera_pose, image_size,
        return_depth=True, filter_behind_camera=False, filter_outside_image=False
    )

    # Convert to integer coordinates for image indexing
    if is_torch:
        image_coords = torch.round(pixels).long()
        image_coords[:, 0] = torch.clamp(image_coords[:, 0], 0, image_size[1] - 1)
        image_coords[:, 1] = torch.clamp(image_coords[:, 1], 0, image_size[0] - 1)
    else:
        image_coords = np.round(pixels).astype(int)
        image_coords[:, 0] = np.clip(image_coords[:, 0], 0, image_size[1] - 1)
        image_coords[:, 1] = np.clip(image_coords[:, 1], 0, image_size[0] - 1)

    result = {
        'pixels': pixels,                    # Continuous pixel coordinates
        'image_coords': image_coords,        # Integer pixel coordinates
        'depths': depths,                    # Depth values
        'valid_indices': valid_indices,     # Indices of visible points
        'view_index': view_index,           # Which view was used
        'camera_pose': camera_pose,         # Camera transformation matrix
        'camera_matrix': K,                 # Camera intrinsic matrix
        'phi': phi,                         # Elevation angle
        'theta': theta                      # Azimuth angle
    }

    if colors is not None:
        result['colors'] = colors[valid_indices]

    return result


def create_depth_image(projection_result: dict, image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Create a depth image from projection results, similar to Blender's depth output.

    Args:
        projection_result: Result from project_points_blender_style()
        image_size: Image resolution (height, width)

    Returns:
        depth_image: 2D depth image with same format as Blender depth output
    """
    is_torch = isinstance(projection_result['depths'], torch.Tensor)

    if is_torch:
        depth_image = torch.zeros(image_size, device=projection_result['depths'].device)
        coords = projection_result['image_coords']
        depths = projection_result['depths']

        # Handle multiple points projecting to same pixel by keeping closest depth
        for i in range(len(coords)):
            y, x = coords[i, 1], coords[i, 0]
            if depth_image[y, x] == 0 or depths[i] < depth_image[y, x]:
                depth_image[y, x] = depths[i]

        depth_image = depth_image.cpu().numpy()
    else:
        depth_image = np.zeros(image_size)
        coords = projection_result['image_coords']
        depths = projection_result['depths']

        # Handle multiple points projecting to same pixel by keeping closest depth
        for i in range(len(coords)):
            y, x = coords[i, 1], coords[i, 0]
            if depth_image[y, x] == 0 or depths[i] < depth_image[y, x]:
                depth_image[y, x] = depths[i]

    return depth_image


def print_blender_camera_info():
    """Print camera information matching the Blender setup for debugging."""
    K = blender_camera_matrix()
    print("=== Blender Camera Setup (from render_single_glb.py) ===")
    print(f"Lens: 35mm")
    print(f"Sensor width: 32mm")
    print(f"Image resolution: 512x512")
    print(f"Focal length in pixels: {K[0, 0]:.1f}")
    print(f"Principal point: ({K[0, 2]:.1f}, {K[1, 2]:.1f})")
    print(f"Camera intrinsic matrix:\n{K}")

    print(f"\nDefault camera distance: 1.2")
    print(f"Number of viewpoints: {len(get_blender_default_viewpoints())}")

    views = get_blender_default_viewpoints()
    for i, (phi, theta) in enumerate(views):
        pose = sample_blender_camera_pose(phi, theta)
        pos = pose[:3, 3]
        print(f"View {i:2d}: phi={phi:.3f}, theta={theta:.3f}, pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")


if __name__ == "__main__":
    print_blender_camera_info()