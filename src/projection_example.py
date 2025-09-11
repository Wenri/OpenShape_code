import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from utils.projection import project_point_cloud_to_image, points_to_pixels, create_camera_matrix, create_camera_pose
from utils.data import normalize_pc
import sys
from PIL import Image

def load_demo_point_cloud():
    """Load demo point cloud or create a simple one"""
    try:
        # Try to load the demo point cloud
        pcd = o3d.io.read_point_cloud("demo/pc.ply")
        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        
        # Sample 1000 points for visualization
        if len(xyz) > 1000:
            idx = np.random.choice(len(xyz), 1000, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
            
    except:
        print("Demo point cloud not found, creating a simple cube...")
        # Create a simple colored cube
        xyz = []
        rgb = []
        for x in np.linspace(-0.5, 0.5, 10):
            for y in np.linspace(-0.5, 0.5, 10):
                for z in np.linspace(-0.5, 0.5, 10):
                    if abs(x) > 0.3 or abs(y) > 0.3 or abs(z) > 0.3:  # Hollow cube
                        xyz.append([x, y, z])
                        rgb.append([abs(x), abs(y), abs(z)])
        xyz = np.array(xyz)
        rgb = np.array(rgb)
    
    # Normalize point cloud
    xyz = normalize_pc(xyz)
    
    return xyz, rgb

def example_basic_projection():
    """Basic example of projecting points to pixels"""
    print("=== Basic Point-to-Pixel Projection ===")
    
    # Load point cloud
    xyz, rgb = load_demo_point_cloud()
    print(f"Loaded point cloud with {len(xyz)} points")
    
    # Set up camera parameters (similar to render_single_glb.py)
    fx = fy = 525.0  # Focal length in pixels
    cx = cy = 256.0  # Principal point (image center for 512x512)
    camera_position = [1.2, 0, 0]  # Camera position
    camera_target = [0, 0, 0]      # Look at origin
    image_size = (512, 512)        # Height, width
    
    # Project points using high-level function
    result = project_point_cloud_to_image(
        xyz, rgb, fx, fy, cx, cy, 
        camera_position, camera_target, image_size
    )
    
    print(f"Projected {len(result['pixels'])} / {len(xyz)} points to image")
    print(f"Pixel coordinates range: x=[{result['pixels'][:, 0].min():.1f}, {result['pixels'][:, 0].max():.1f}], "
          f"y=[{result['pixels'][:, 1].min():.1f}, {result['pixels'][:, 1].max():.1f}]")
    print(f"Depth range: [{result['depths'].min():.3f}, {result['depths'].max():.3f}]")
    
    return result, xyz, rgb

def example_torch_projection():
    """Example using PyTorch tensors"""
    print("\n=== PyTorch Tensor Projection ===")
    
    # Load point cloud
    xyz, rgb = load_demo_point_cloud()
    
    # Convert to torch tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xyz_torch = torch.from_numpy(xyz).float().to(device)
    rgb_torch = torch.from_numpy(rgb).float().to(device)
    
    print(f"Using device: {device}")
    
    # Project using torch
    result = project_point_cloud_to_image(
        xyz_torch, rgb_torch, 
        fx=525.0, fy=525.0, cx=256.0, cy=256.0,
        camera_position=[1.2, 0, 0], camera_target=[0, 0, 0]
    )
    
    print(f"Projected {len(result['pixels'])} points using PyTorch")
    
    return result

def example_multiple_viewpoints():
    """Example with multiple camera viewpoints"""
    print("\n=== Multiple Viewpoints ===")
    
    xyz, rgb = load_demo_point_cloud()
    
    # Multiple camera positions around the object
    viewpoints = [
        [1.2, 0, 0],    # Front
        [0, 1.2, 0],    # Side  
        [0, 0, 1.2],    # Top
        [0.8, 0.8, 0.8] # Diagonal
    ]
    
    results = []
    for i, cam_pos in enumerate(viewpoints):
        result = project_point_cloud_to_image(
            xyz, rgb, camera_position=cam_pos
        )
        results.append(result)
        print(f"View {i+1}: {len(result['pixels'])} visible points from position {cam_pos}")
    
    return results

def visualize_projection(result, xyz, rgb, save_path="projection_demo.png"):
    """Visualize the projection result"""
    print(f"\n=== Visualizing Projection ===")
    
    # Create image
    image_size = (512, 512)
    image = np.zeros((image_size[0], image_size[1], 3))
    
    # Place points in image
    pixels = result['image_coords']
    colors = result.get('colors', np.ones((len(pixels), 3)) * 0.5)
    
    for i in range(len(pixels)):
        x, y = pixels[i]
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            image[y, x] = colors[i]
    
    # Save and show
    plt.figure(figsize=(10, 5))
    
    # Original 3D points
    plt.subplot(1, 2, 1)
    plt.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=1)
    plt.title("Original 3D Points (X-Y view)")
    plt.axis('equal')
    
    # Projected 2D image
    plt.subplot(1, 2, 2)
    plt.imshow(image, origin='upper')
    plt.title("Projected 2D Image")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    return image

def example_advanced_usage():
    """Advanced usage with manual camera setup"""
    print("\n=== Advanced Manual Camera Setup ===")
    
    xyz, rgb = load_demo_point_cloud()
    
    # Create camera matrices manually
    K = create_camera_matrix(fx=600, fy=600, cx=320, cy=240)  # Different resolution
    pose = create_camera_pose(
        position=[0.5, 0.5, 1.0],
        target=[0, 0, 0],
        up=[0, 0, 1]
    )
    
    # Project with manual setup
    pixels, valid_indices, depths = points_to_pixels(
        xyz, K, pose, 
        image_size=(480, 640),  # Different resolution
        return_depth=True
    )
    
    print(f"Manual projection: {len(pixels)} valid points")
    print(f"Camera matrix:\n{K}")
    print(f"Camera pose shape: {pose.shape}")
    
    return pixels, valid_indices, depths

if __name__ == "__main__":
    print("Point Cloud to Image Projection Examples")
    print("=" * 50)
    
    # Run examples
    result, xyz, rgb = example_basic_projection()
    
    torch_result = example_torch_projection()
    
    multi_results = example_multiple_viewpoints()
    
    advanced_result = example_advanced_usage()
    
    # Visualize main result
    try:
        image = visualize_projection(result, xyz, rgb)
        print("\nExample completed successfully!")
        print("Check the generated 'projection_demo.png' file.")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("But projection functions work correctly!")
    
    print("\nUsage in your code:")
    print("""
    from utils.projection import project_point_cloud_to_image
    
    # Simple usage
    result = project_point_cloud_to_image(
        points_3d=your_xyz,  # (N, 3) array
        colors=your_rgb,     # (N, 3) array [optional]
        fx=525.0, fy=525.0,  # Focal lengths
        cx=256.0, cy=256.0,  # Principal point
        camera_position=[1.2, 0, 0],
        camera_target=[0, 0, 0]
    )
    
    # Access results
    pixels = result['pixels']          # (M, 2) pixel coordinates
    indices = result['valid_indices']  # (M,) indices of visible points
    depths = result['depths']          # (M,) depth values
    """)