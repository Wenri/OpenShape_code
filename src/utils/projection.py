import numpy as np
import torch
from typing import Tuple, Optional, Union

def points_to_pixels(
    points_3d: Union[np.ndarray, torch.Tensor],
    camera_matrix: Union[np.ndarray, torch.Tensor],
    camera_pose: Union[np.ndarray, torch.Tensor],
    image_size: Tuple[int, int] = (512, 512),
    return_depth: bool = False,
    filter_behind_camera: bool = True,
    filter_outside_image: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Map 3D points to 2D pixels using camera projection.
    
    Args:
        points_3d: 3D points in world coordinates, shape (N, 3)
        camera_matrix: Camera intrinsic matrix K, shape (3, 3)
                      [[fx, 0,  cx],
                       [0,  fy, cy], 
                       [0,  0,  1 ]]
        camera_pose: Camera extrinsic matrix (world to camera transform), shape (4, 4)
                    Can be camera-to-world (will be inverted) or world-to-camera
        image_size: Image dimensions (height, width)
        return_depth: Whether to return depth values
        filter_behind_camera: Filter out points behind the camera (negative depth)
        filter_outside_image: Filter out points outside image bounds
        
    Returns:
        pixels: 2D pixel coordinates, shape (M, 2) where M <= N
        valid_indices: Indices of valid points from input
        depths: (optional) Depth values for each valid point, shape (M,)
    """
    is_torch = isinstance(points_3d, torch.Tensor)
    
    if is_torch:
        points_3d = points_3d.float()
        camera_matrix = camera_matrix.float()
        camera_pose = camera_pose.float()
        device = points_3d.device
        np_func = torch
        concat_func = torch.cat
        ones_func = torch.ones
        zeros_func = torch.zeros
        arange_func = torch.arange
    else:
        np_func = np
        concat_func = np.concatenate
        ones_func = np.ones
        zeros_func = np.zeros
        arange_func = np.arange
    
    # Ensure points are in homogeneous coordinates
    if points_3d.shape[1] == 3:
        if is_torch:
            ones = ones_func((points_3d.shape[0], 1), device=device)
        else:
            ones = ones_func((points_3d.shape[0], 1))
        points_3d_hom = concat_func([points_3d, ones], axis=1)  # (N, 4)
    else:
        points_3d_hom = points_3d
    
    # Transform points to camera coordinate system
    # If camera_pose is camera-to-world, invert it to get world-to-camera
    if is_torch:
        camera_to_world = camera_pose
        world_to_camera = torch.inverse(camera_to_world)
    else:
        camera_to_world = camera_pose
        world_to_camera = np.linalg.inv(camera_to_world)
    
    # Transform points to camera coordinates
    if is_torch:
        points_camera = torch.matmul(points_3d_hom, world_to_camera.T)  # (N, 4)
    else:
        points_camera = np.dot(points_3d_hom, world_to_camera.T)  # (N, 4)
    
    # Extract XYZ in camera coordinates
    points_camera_xyz = points_camera[:, :3]  # (N, 3)
    
    # Project to image plane using camera intrinsics
    if is_torch:
        points_2d_hom = torch.matmul(points_camera_xyz, camera_matrix.T)  # (N, 3)
    else:
        points_2d_hom = np.dot(points_camera_xyz, camera_matrix.T)  # (N, 3)
    
    # Convert from homogeneous to pixel coordinates
    depths = points_2d_hom[:, 2]  # Z values (depth)
    
    # Avoid division by zero
    if is_torch:
        valid_depth_mask = depths > 1e-6
        pixels = torch.zeros((points_3d.shape[0], 2), device=device)
        pixels[valid_depth_mask, 0] = points_2d_hom[valid_depth_mask, 0] / depths[valid_depth_mask]  # u (x)
        pixels[valid_depth_mask, 1] = points_2d_hom[valid_depth_mask, 1] / depths[valid_depth_mask]  # v (y)
    else:
        valid_depth_mask = depths > 1e-6
        pixels = np.zeros((points_3d.shape[0], 2))
        pixels[valid_depth_mask, 0] = points_2d_hom[valid_depth_mask, 0] / depths[valid_depth_mask]  # u (x)
        pixels[valid_depth_mask, 1] = points_2d_hom[valid_depth_mask, 1] / depths[valid_depth_mask]  # v (y)
    
    # Create validity mask
    if is_torch:
        valid_mask = torch.ones(points_3d.shape[0], dtype=torch.bool, device=device)
    else:
        valid_mask = np.ones(points_3d.shape[0], dtype=bool)
    
    # Filter points behind camera
    if filter_behind_camera:
        valid_mask = valid_mask & (depths > 1e-6)
    
    # Filter points outside image bounds
    if filter_outside_image:
        height, width = image_size
        valid_mask = valid_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & \
                     (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    
    # Get valid indices
    if is_torch:
        valid_indices = torch.where(valid_mask)[0]
    else:
        valid_indices = np.where(valid_mask)[0]
    
    # Return only valid pixels
    valid_pixels = pixels[valid_mask]
    valid_depths = depths[valid_mask]
    
    if return_depth:
        return valid_pixels, valid_indices, valid_depths
    else:
        return valid_pixels, valid_indices


def create_camera_matrix(
    fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """
    Create camera intrinsic matrix from focal lengths and principal point.
    
    Args:
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)  
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
    """
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ], dtype=np.float32)


def create_camera_pose(
    position: Union[list, np.ndarray],
    target: Union[list, np.ndarray], 
    up: Union[list, np.ndarray] = [0, 0, 1]
) -> np.ndarray:
    """
    Create camera pose matrix from position, target, and up vector.
    
    Args:
        position: Camera position in world coordinates [x, y, z]
        target: Point the camera is looking at [x, y, z]
        up: Up vector [x, y, z], default [0, 0, 1]
        
    Returns:
        camera_pose: 4x4 camera-to-world transformation matrix
    """
    position = np.array(position, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    # Create coordinate system
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / np.linalg.norm(up_corrected)
    
    # Create camera-to-world matrix
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up_corrected  
    camera_pose[:3, 2] = -forward  # Camera looks in negative Z
    camera_pose[:3, 3] = position
    
    return camera_pose


def project_point_cloud_to_image(
    points_3d: Union[np.ndarray, torch.Tensor],
    colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    fx: float = 525.0,
    fy: float = 525.0, 
    cx: float = 256.0,
    cy: float = 256.0,
    camera_position: list = [1.2, 0, 0],
    camera_target: list = [0, 0, 0],
    image_size: Tuple[int, int] = (512, 512)
) -> dict:
    """
    High-level function to project a point cloud onto an image plane.
    
    Args:
        points_3d: 3D points, shape (N, 3)
        colors: Optional RGB colors for each point, shape (N, 3), values in [0, 1]
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates
        camera_position: Camera position [x, y, z]
        camera_target: Point camera looks at [x, y, z] 
        image_size: Output image size (height, width)
        
    Returns:
        dict with keys:
            'pixels': 2D pixel coordinates (M, 2)
            'depths': Depth values (M,)
            'colors': Colors if provided (M, 3)
            'valid_indices': Original indices of valid points (M,)
            'image_coords': Integer pixel coordinates for indexing (M, 2)
    """
    is_torch = isinstance(points_3d, torch.Tensor)
    
    # Create camera matrices
    K = create_camera_matrix(fx, fy, cx, cy)
    pose = create_camera_pose(camera_position, camera_target)
    
    if is_torch:
        K = torch.from_numpy(K).to(points_3d.device)
        pose = torch.from_numpy(pose).to(points_3d.device)
    
    # Project points
    pixels, valid_indices, depths = points_to_pixels(
        points_3d, K, pose, image_size, return_depth=True
    )
    
    # Get integer coordinates for image indexing
    if is_torch:
        image_coords = torch.round(pixels).long()
        image_coords[:, 0] = torch.clamp(image_coords[:, 0], 0, image_size[1] - 1)
        image_coords[:, 1] = torch.clamp(image_coords[:, 1], 0, image_size[0] - 1)
    else:
        image_coords = np.round(pixels).astype(int)
        image_coords[:, 0] = np.clip(image_coords[:, 0], 0, image_size[1] - 1)
        image_coords[:, 1] = np.clip(image_coords[:, 1], 0, image_size[0] - 1)
    
    result = {
        'pixels': pixels,
        'depths': depths,
        'valid_indices': valid_indices,
        'image_coords': image_coords
    }
    
    # Add colors if provided
    if colors is not None:
        valid_colors = colors[valid_indices]
        result['colors'] = valid_colors
        
    return result